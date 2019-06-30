(* Wolfram Language Package *)

BeginPackage["IMAPLink`"]
IMAPExecute
IMAPConnect
IMAPLink`Handles
Begin["`Private`"] (* Begin Private Context *) 

Needs["CURLLink`"]
Needs["CURLLink`Utilities`"]
Needs["IMAPLink`Utilities`"]
Needs["MailLink`"]
Dataset;
TypeSystem`PackageScope`$ElisionEnabled = False;
IMAPConnect[data_?AssociationQ]:=
Module[
	{handle,result,url = Lookup[data,"URL",None]},
	handle = setConnectionOptions[data];
	result = establishConnection[handle];
	result=Join[result,<|"URL"->url,"Type"->"Server"|>];
	result
	
]

$internalID = 0;
IMAPLink`Handles=<||>
lookupCURLHandle[assoc_Association]:= 
(
Lookup[
		Lookup[IMAPLink`Handles,Lookup[assoc,"InternalID"]],
		"CURLHandle"
	  ]
)
lookupTask[assoc_Association]:= 
(
Lookup[
		Lookup[IMAPLink`Handles,Lookup[assoc,"InternalID"]],
		"Task"
	  ]
)
IMAPExecute[uassoc_Association,"Select Folder",folderName_?StringQ]:=
Module[
	{res,handle,assoc=uassoc,rawres, maxSearchResult, prettyFolderName},
	
	handle = lookupCURLHandle[assoc];
	
	CURLLink`CURLReset[handle];
	
	assoc = Join[assoc,<|"Command"->"SELECT \""<>folderName<>"\""|>];
	
	rawres = IMAPExecute[assoc];
	
	assoc = Join[assoc,<|"Command"->"SEARCH RETURN (COUNT) UNSEEN"|>];
	
	rawres = StringJoin[{rawres,IMAPExecute[assoc] }];
	
	res = IMAPLink`Utilities`import[rawres,"SelectFolder"];
	
	prettyFolderName = IMAPLink`Utilities`import[folderName,"PrettyFolderName"];
	
	maxSearchResult = Lookup[res,"TotalMessageCount"];
	
	res = Join[res,<|"ObjectType"->"MailFolder","Path"->folderName,"Folder"->prettyFolderName,"InternalID"->Lookup[uassoc,"InternalID"]|>];
	
	res
]

IMAPExecute[uassoc_Association,"Examine Folder",folderName_?StringQ]:=
Module[
	{res,handle,assoc=uassoc,rawres},
	
	handle = lookupCURLHandle[assoc];
	
	CURLLink`CURLReset[handle];
	
	assoc = Join[assoc,<|"Command"->"STATUS \""<>folderName<>"\" (UNSEEN MESSAGES)"|>];
	
	rawres = IMAPExecute[assoc];
	
	res=IMAPLink`Utilities`import[rawres,"ExamineFolder"];
	
	res = Join[res,<|"ObjectType"->"MailFolder","InternalID"->Lookup[uassoc,"InternalID"]|>];
	
	
	res
]


IMAPExecute[assoc_Association,ids_List,"FetchMailSummary"]:= 
Module[
	{data,fullData,messageData,commandAssocs,envelopeData,bodyStructureData,flagData,searchResult},
	
	envelopeData = Table[
						  data = Join[assoc,<|"Command"->"FETCH "<>ids[[i]]<>" (ENVELOPE)"|>];
						  IMAPLink`Utilities`import[sendCommand[data],"SearchResults","ENVELOPE"]
						  ,
						  {i,1,Length[ids]}
					    ];
	envelopeData = Flatten[envelopeData];
	
	bodyStructureData = Table[
							  data = Join[assoc,<|"Command"->"FETCH "<>ids[[i]]<>" (BODYSTRUCTURE)"|>];
							  IMAPLink`Utilities`import[sendCommand[data],"SearchResults","BODYSTRUCTURE"]
							  ,
							  {i,1,Length[ids]}
						     ];
	bodyStructureData = Flatten[bodyStructureData];
	
	flagData = Table[	
					 data = Join[assoc,<|"Command"->"FETCH "<>ids[[i]]<>" (FLAGS)"|>];
					 IMAPLink`Utilities`import[sendCommand[data],"SearchResults","FLAGS"]
					 ,
					 {i,1,Length[ids]}
					];
	flagData = Flatten[flagData];
	
	commandAssocs = DeleteCases[Map[makeCommandAssoc,bodyStructureData],Null];
	
	messageData = Map[IMAPLink`Utilities`import[sendCommand[Join[assoc,#]],#["Encoding"],#["Charset"],#["ContentType"],"Body"]&,commandAssocs];

	fullData = JoinAcross[envelopeData, bodyStructureData, Key["MessagePosition"]];
	
	fullData = JoinAcross[fullData, flagData, Key["MessagePosition"]];
	
	searchResult = JoinAcross[fullData,messageData,Key["MessagePosition"]];
	
	searchResult = Map[Append[#,"Path"->assoc["Path"]]&,searchResult];

	searchResult = Map[Append[#,makeMailItem[KeyTake[#,{"Path","MessageID","From","FromName","FromAddress","Flags","MessagePosition","OriginatingDate","Subject","ToList","ToAddressList","ToNameList","Attachments","CcList","CcNameList","CcAddressList","Body"}],data["InternalID"]]]&,searchResult]
	
]

makeCommandAssoc=
Function[assoc,
	If[
		StringQ[assoc["PLAIN"]["Command"]],
		<|"Charset"->assoc["PLAIN"]["Charset"],"Encoding"->assoc["PLAIN"]["Encoding"],"Command"->assoc["PLAIN"]["Command"],"ContentType"->"PLAIN"|>
		,
		If[
			StringQ[assoc["HTML"]["Command"]],
			<|"Charset"->assoc["HTML"]["Charset"],"Encoding"->assoc["HTML"]["Encoding"],"Command"->assoc["HTML"]["Command"],"ContentType"->"HTML"|>]
		   ]
		 ]

makeMailItem[assoc_,internalID_]:="MailItem"->System`MailItem[Append[assoc,"InternalID"->internalID]];



IMAPExecute[assoc_Association,"SetFlag",Rule[flag_,bit_]]:=
Module[
	{data,raw,res},
	If[MemberQ[IMAPLink`Utilities`$permFlags,flag],
		data = Append[assoc,"Command"->"STORE "<>ToString@assoc["MessagePosition"]<>" "<>If[bit,"+","-"]<>"FLAGS"<>" (\\"<>flag<>")"];
		,
		data=Append[assoc,"Command"->"STORE "<>ToString@assoc["MessagePosition"]<>" "<>If[bit,"+","-"]<>"FLAGS"<>" ("<>flag<>")"];
	];
	raw=sendCommand[data];
	res = IMAPLink`Utilities`import[raw,"SearchResults","FLAGS"];
	Append[assoc,IMAPLink`Utilities`import[raw,"SearchResults","FLAGS"][[1]]]
]
IMAPExecute[assoc_Association,"SetFlags",flags_Association]:=
Module[
	{data, raw, res, removeflags, addflags, newflags, removeQ = False, addQ = False},
	
	newflags = KeyMap[If[MemberQ[IMAPLink`Utilities`$permFlags,#],"\\"<>#,#]&,flags];
	
	removeflags = If[Keys[Select[newflags, ! # &]] =!= {},removeQ = True; "-FLAGS ("<>StringJoin@Riffle[Keys[Select[newflags, ! # &]]," "]<>")", ""];
	
	If[removeQ,
			data = Append[assoc,"Command"->"STORE "<>ToString@assoc["MessagePosition"]<>" "<>removeflags];
			raw = sendCommand[data];
	];
	
	addflags = If[Keys[Select[newflags,# &]] =!= {},addQ = True; "+FLAGS ("<>StringJoin@Riffle[Keys[Select[newflags,  # &]]," "]<>")", ""];
	
	If[addQ,
			data = Append[assoc,"Command"->"STORE "<>ToString@assoc["MessagePosition"]<>" "<>addflags];
			raw = sendCommand[data];
	];
	
	res = IMAPLink`Utilities`import[raw,"SearchResults","FLAGS"];
	
	If[Head[res] === List && Length[res] >= 1, 
			Append[assoc,IMAPLink`Utilities`import[raw,"SearchResults","FLAGS"][[1]]]
			,
			assoc
	]
]

IMAPExecute[assocs_List,"SetFlags",flags_Association]:=
Module[
	{tassoc,idsString,newassocs,data,raw,res,removeflags,addflags,newflags,removeQ=False,addQ=False,assoc,ids},
	newassocs = Map[KeyTake[Extract[#, 1], {"MessagePosition", "InternalID"}] &, assocs];
	assoc = Merge[Map[Rule[#[[2]], #[[1]]] &, Map[Values, newassocs]], Identity];
	ids=Values[assoc][[1]];
	idsString=ToString/@ids;
	idsString=StringJoin@Riffle[idsString,","];
	
	newflags = KeyMap[If[MemberQ[IMAPLink`Utilities`$permFlags,#],"\\"<>#,#]&,flags];
	
	removeflags=If[Keys[Select[newflags, ! # &]]=!={},removeQ=True;"-FLAGS ("<>StringJoin@Riffle[Keys[Select[newflags, ! # &]]," "]<>")", ""];
	If[removeQ,
		tassoc=<|"InternalID"->Keys[assoc][[1]]|>;
		data = Append[tassoc,"Command"->"STORE "<>idsString<>" "<>removeflags];
		raw=sendCommand[data];
	];
	
	addflags=If[Keys[Select[newflags,# &]]=!={},addQ=True;"+FLAGS ("<>StringJoin@Riffle[Keys[Select[newflags,  # &]]," "]<>")",""];
	If[addQ,
		tassoc=<|"InternalID"->Keys[assoc][[1]]|>;
		data = Append[tassoc,"Command"->"STORE "<>idsString<>" "<>addflags];
		raw=sendCommand[data];
		];
	
	res = IMAPLink`Utilities`import[raw,"SearchResults","FLAGS"];
	Append[assoc,IMAPLink`Utilities`import[raw,"SearchResults","FLAGS"][[1]]]
]

IMAPExecute[assoc_Association,"FetchMail",seqID_]/;(MatchQ[seqID,_Integer]||Quiet@StringMatchQ[seqID,NumberString]):= 
Module[
	{data,raw,res},
	data = Join[assoc,<|"Command"->"FETCH "<>ToString[seqID]<>" (BODY.PEEK[])"|>];
	raw = sendCommand[data];
	res = IMAPLink`Utilities`import[raw,"EML"];
	Append[assoc,res]
]



IMAPExecute[assoc_Association,"FetchHeaders",seqID_]/;(MatchQ[seqID,_Integer]||Quiet@StringMatchQ[seqID,NumberString]):= 
Module[
	{data,res},
	
	data = Join[assoc,<|"Command"->"FETCH "<>ToString[seqID]<>" (BODY[HEADER.FIELDS (DATE FROM TO SUBJECT)])"|>];
	res = IMAPLink`Utilities`import[sendCommand[data],"FetchHeaders"];
	
]

IMAPExecute[assoc_Association,"FetchFlags",seqID_]/;(MatchQ[seqID,_Integer]||Quiet@StringMatchQ[seqID,NumberString]):= 
Module[
	{data,raw,res},
	data = Join[assoc,<|"Command"->"FETCH "<>ToString[seqID]<>" FLAGS"|>];
	raw = sendCommand[data];
	res = IMAPLink`Utilities`import[raw,"SearchResults","FLAGS"];
	JoinAcross[res,{assoc},"MessagePosition"][[1]]
]

IMAPExecute[assoc_Association,"Copy",destinationFolder_String,seqID_]/;(MatchQ[seqID,_Integer]||Quiet@StringMatchQ[seqID,NumberString]):= 
Module[
	{data,raw},
	data = Join[assoc,<|"Command"->"COPY "<>ToString[seqID]<>" "<>destinationFolder|>];
	raw = sendCommand[data];
	assoc
]

IMAPExecute[assoc_Association,"Search",query_String]:= 
Module[
	{data,ids,res={}},
	
	data = Join[assoc,<|"Command"->"SEARCH TEXT \""<>query<>"\"","Search"->True|>];
	ids = IMAPLink`Utilities`import[sendCommand[data],"SearchIds"];
	If[ids=!={},res =makePrettyDataset[IMAPExecute[assoc,ids,"FetchMailSummary"]][Reverse]];
	res
	
]	 	


IMAPExecute[assoc_Association,"Search",query_Association]:= 
Module[
	{data,ids,command,raw,data2,maxItems},
	command = makeFullCommand[query];
	data = Join[assoc,<|"Command"->command,"Search"->True|>];
	raw=sendCommand[data];
	maxItems=Lookup[query,MaxItems,Automatic];
	ids = IMAPLink`Utilities`import[raw,maxItems,"SearchIds"];
	data2 = IMAPExecute[assoc,ids,"FetchMailSummary"];
	If[TrueQ[Lookup[query,"Return"]==="MailItem"],
		If[TrueQ[AssociationQ[data2[[1]]]],data2[[1]]["MailItem"],<||>],
		If[ids==={},Dataset[{}],makePrettyDataset[data2,Lookup[query,"Property",{}]][Reverse]]
	]
	
]	 	
IMAPExecute[assoc_Association,"Expunge"]:= 
Module[
	{data,str},
	data = Join[assoc,<|"Command"->"EXPUNGE"|>];
	str=sendCommand[data];
	IMAPLink`Utilities`import[str,"Expunge"]
]
IMAPExecute[assoc_Association,command_?StringQ]:= 
Module[
	{data},
	data = Join[assoc,<|"Command"->command|>];
	sendCommand[data]
]

IMAPExecute[uassoc_Association]:= 
Module[
	{assoc=uassoc},
	sendCommand[assoc]
]
setConnectionOptions[data_?AssociationQ]:=
Module[
	{handle,port,timeout},
	(*this may be evaluated many times*)
	port = findPort[Lookup[data,"URL"],Lookup[data,"PortNumber"]];
	timeout=CURLLink`Utilities`toIntegerMilliseconds[Lookup[data,TimeConstraint,60]];
	handle = CURLLink`CURLHandleLoad[];
	CURLLink`CURLOption[handle, "CURLOPT_ERRORBUFFER", 1];
	CURLLink`CURLOption[handle, "CURLOPT_HEADERFUNCTION","WRITE_HEADER"];
	CURLLink`CURLOption[handle, "CURLOPT_WRITEHEADER", "MemoryPointer"];
	CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "WRITE_MEMORY"];
	CURLLink`CURLOption[handle, "CURLOPT_WRITEDATA", "MemoryPointer"];
	CURLLink`CURLOption[handle, "CURLOPT_DEFAULT_PROTOCOL","imaps"];
	CURLLink`CURLOption[handle, "CURLOPT_PORT",port];
	CURLLink`CURLOption[handle, "CURLOPT_URL",Lookup[data,"URL"]];
	CURLLink`CURLOption[handle, "CURLOPT_TCP_KEEPALIVE",1];
	CURLLink`CURLOption[handle, "CURLOPT_TCP_KEEPIDLE",40];
	CURLLink`CURLOption[handle, "CURLOPT_TCP_KEEPINTVL",20];
	CURLLink`CURLOption[handle, "CURLOPT_CONNECTTIMEOUT_MS", timeout];
	CURLLink`CURLOption[handle, "CURLOPT_USERNAME",Lookup[data,"Username"]];
	CURLLink`CURLOption[handle, "CURLOPT_PASSWORD",Lookup[data,"Password"]];
	
	handle
]

findPort[url_String,Automatic]:=Which[
	StringMatchQ[url,"imaps://"~~___],
	993
	,
	StringMatchQ[url,"imap://"~~___],
	143
	,
	True,
	993
	]
findPort[url_,port_Integer]:=port

sendCommand[data_?AssociationQ]:=
Module[
	{handle,headers,res},
	(*this may be evaluated many times*)
	handle=lookupCURLHandle[data];
	TaskSuspend[lookupTask[data]];
	CURLLink`CURLOption[handle, "CURLOPT_CONNECTTIMEOUT_MS", Round[1000*N[30]]];
	CURLLink`CURLOption[handle, "CURLOPT_HEADERFUNCTION","WRITE_HEADER"];
	CURLLink`CURLOption[handle, "CURLOPT_WRITEHEADER", "MemoryPointer"];
	CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "WRITE_MEMORY"];
	CURLLink`CURLOption[handle, "CURLOPT_WRITEDATA", "MemoryPointer"];
	CURLLink`CURLOption[handle, "CURLOPT_URL",Lookup[data,"URL"]];
	CURLLink`CURLOption[handle, "CURLOPT_BUFFERSIZE",1024*16];
	CURLLink`CURLOption[handle, "CURLOPT_CUSTOMREQUEST",Lookup[data,"Command"]];
	CURLLink`CURLPerform[handle];
	headers = CURLLink`CURLHeaderData[handle];
	res = ByteArrayToString[CURLLink`CURLRawContentDataRawArray[handle]];
	CURLLink`CURLReset[handle];
	TaskResume[lookupTask[data]];
	If[Lookup[data,"Search",False],res,headers]
	
]

establishConnection[handle_CURLLink`CURLHandle]:=
Module[
	{headers,data,res,str,task,creds,retCode,error1,error2},
	CURLLink`CURLCreateRunThread[handle];
	retCode = CURLLink`CURLWait[handle];
	
	If[retCode =!=0, 
		error1=CURLLink`CURLErrorData[handle];
		error2=CURLLink`CURLError[retCode];
		Throw[If[StringLength[error1]>StringLength[error2],error1,error2],CURLLink`Utilities`Exception]];
	
	headers = CURLLink`CURLHeaderData[handle];
	data = ByteArrayToString[CURLLink`CURLRawContentDataRawArray[handle]];
	CURLLink`CURLReset[handle];
	str = StringJoin[{headers,data}];
	While[StringContainsQ[str,"Authentication Failure",IgnoreCase->True] ,
		CURLLink`CURLReset[handle];
		creds = fetchCreds[];
		If[creds =!=$Failed,
			CURLLink`CURLOption[handle, "CURLOPT_USERNAME",creds[[1]]];
			CURLLink`CURLOption[handle, "CURLOPT_PASSWORD",creds[[2]]];
			CURLLink`CURLPerform[handle];
			headers = CURLLink`CURLHeaderData[handle];
			data = ByteArrayToString[CURLLink`CURLRawContentDataRawArray[handle]];
			str = StringJoin[{headers,data}];
			MailLink`SetIncomingMailSettings[<|"Username"->creds[[1]],"Password"->creds[[2]]|>];
			,
			Throw[str,CURLLink`Utilities`Exception];
			
		];
		
		
	];
	
	$internalID++;
	task = SessionSubmit[ScheduledTask[IMAPExecute[Join[res,<|"Command"->"NOOP"|>]],25]];
	AssociateTo[IMAPLink`Handles,$internalID-><|"CURLHandle"->handle,"Task"->task|>];
	
	res = <|"Folders"->IMAPLink`Utilities`import[str,"Folders"],"InternalID"->$internalID|>;
	res
]
fetchCreds[]/;frontEndQ:=(AuthenticationDialog["UsernamePassword", ($pwdDlgResult = 
    If[AssociationQ[#], {#Username, #Password}, $Canceled]) &, 
 AppearanceRules -> 
  Association[
   "Description" -> 
    Style[Column[{""}, Spacings -> 1], "FormDescription"]], 
 WindowTitle -> "Authentication Required", Evaluator -> CurrentValue["Evaluator"]])
 
fetchCreds[]/;Not[frontEndQ] :=
	AuthenticationDialog[
		"UsernamePassword",
		($pwdDlgResult = If[AssociationQ[#], {#Username, #Password}, $Canceled])&,
		AppearanceRules -> Association[
			"Description" -> Column[{prompt1, prompt2, prompt3}]
		]
	]
frontEndQ:= ToString[Head[$FrontEnd]] === "FrontEndObject"
makeCommand["MessageID",argument_String]:=StringJoin[{"HEADER"," ","Message-Id:"," ",argument}] 
makeCommand["From",argument_String]:=StringJoin[{"FROM"," ",quote[argument]}]
makeCommand["From",arguments_List]:="OR "<>Riffle[Map[makeCommand["From",#]&,arguments]," "]
makeCommand["To",argument_String]:=StringJoin[{"TO"," ",quote[argument]}]
makeCommand["Cc",argument_String]:=StringJoin[{"CC"," ",quote[argument]}]
makeCommand["Bcc",argument_String]:=StringJoin[{"BCC"," ",quote[argument]}]

makeCommand["On",date_DateObject]:=StringJoin[{"ON"," ",todate[date]}]
makeCommand["SentOnDate",date_DateObject]:=StringJoin[{"SENTON"," ",todate[date]}]
makeCommand["SentOn",date_DateObject]:=StringJoin[{"SENTON"," ",todate[date]}]
makeCommand["SentAfterDate",date_DateObject]:=StringJoin[{"SENTSINCE"," ",todate[date]}]
makeCommand["SentAfter",date_DateObject]:=StringJoin[{"SENTSINCE"," ",todate[date]}]
makeCommand["Since",date_DateObject]:=StringJoin[{"SINCE"," ",todate[date]}]
makeCommand["SentBeforeDate",date_DateObject]:=StringJoin[{"BEFORE"," ",todate[date]}]
makeCommand["SentBefore",date_DateObject]:=StringJoin[{"BEFORE"," ",todate[date]}]

makeCommand["SizeLessThan",bytes_Integer]:=StringJoin[{"SMALLER"," ",ToString[bytes]}]
makeCommand["SizeGreaterThan",bytes_Integer]:=StringJoin[{"LARGER"," ",ToString[bytes]}]

makeCommand["Subject",argument_String]:=StringJoin[{"SUBJECT"," ",quote[argument]}]
makeCommand["Text",argument_String]:=StringJoin[{"TEXT"," ",quote[argument]}]
makeCommand["Body",argument_String]:=StringJoin[{"BODY"," ",quote[argument]}]

makeCommand["Flags",flags_List]:=StringJoin[Riffle[Map[ToUpperCase,flags]," "]]
makeCommand["Tags",flags_List]:=StringJoin[Riffle[Map[ToUpperCase,flags]," "]]
makeCommand[flag_String/;MemberQ[IMAPLink`Utilities`$permFlags,flag],True]:=ToUpperCase[flag]
makeCommand[flag_String/;MemberQ[IMAPLink`Utilities`$permFlags,flag],False]:="UN"<>ToUpperCase[flag]
makeCommand[flag_String,True]:="KEYWORD "<>flag
makeCommand[flag_String,False]:="UNKEYWORD "<>flag
makeCommand["TagsSet",tags_List]:=If[
	Length[tags] > 1,
	 "OR "<>StringJoin[Riffle[Map[makeCommand[#,True]&,tags]," "]],
	  StringJoin[Riffle[Map[makeCommand[#,True]&,tags]," "]]
]
makeCommand["TagsUnset",tags_List]:=
If[
	Length[tags] > 1,
	 "OR "<>StringJoin[Riffle[Map[makeCommand[#,False]&,tags]," "]],
	  StringJoin[Riffle[Map[makeCommand[#,False]&,tags]," "]]
] 
makeCommand[___]:=Nothing

makeFullCommand[query_Association] := "SEARCH "<>StringJoin[Riffle[Map[makeCommand[#, Lookup[query, #]] &, Keys[query]]," "]]

quote[str_String]:=StringJoin[{"\"",str,"\""}]
todate[date_DateObject] :=DateString[date, {"Day", "-", "MonthNameShort", "-", "Year"}]
makeprettyMessage[istring_]:=
Module[
	{string=istring},
	string=StringJoin[
	       			Riffle[StringTrim /@ StringSplit@StringReplace[StringReplace[string, ("\n" | "\r"|"\t") -> " "],Repeated[" "]->" "], " "]
	       		];
  Style[
  	If[StringLength[string] >= 100,     
	    	InsertLinebreaks[StringTake[string,97],52]<>"..."
	    ,
	    InsertLinebreaks[
	    	StringPadRight[
	     		StringJoin[
	       			Riffle[StringTrim /@ StringSplit@StringReplace[string, ("\n" | "\r") -> " "], " "]
	       		],
	       		100],51
	    ]
      ]
     ,
     Gray]
        
]     
makePrettySubject=Function[{subject},StringPadRight[If[StringLength[StringTrim[subject]]>50,StringTake[StringTrim[subject],UpTo[47]]<>"...",StringTrim[subject]],50]]        
makePrettyName=Function[{name},
	
		Style[
			If[name[[1]]=!=Missing["NotAvailable"],
				StringPadRight[
					If[StringLength[StringTrim[name[[1]]]]>=50,
						StringTake[StringTrim[name[[1]]],UpTo[47]]<>"..."
						,
						StringTrim[name[[1]]]
						]
						,50
						]
						,
						StringPadRight[
					If[StringLength[StringTrim[name[[2]]]]>50,
						StringTake[StringTrim[name[[2]]],UpTo[47]]<>"..."
						,
						StringTrim[name[[2]]]
						]
						,50
						]
						
				]
			,Bold]
	
	
]
makePrettyDate=Function[{date},Item[Style[
	Which[
		DateValue[date, {"Year", "Month", "Day"}]===DateValue[Today, {"Year", "Month", "Day"}],
		DateString[TimeObject[date], {"Hour12Short", ":", "Minute"," " ,"AMPM"}],
		DateValue[date, {"Year", "Month", "Day"}]===DateValue[Yesterday, {"Year", "Month", "Day"}],
		"Yesterday"
		,
		True,
		DateString[date,{"MonthShort","/","DayShort","/","YearShort"}]]
	,Gray],Alignment->Right]]
blueDot[unreadQ_?BooleanQ]=If[unreadQ,Style["\[FilledSmallCircle]", RGBColor[.1, .7, 1], Larger](*Graphics[{RGBColor[.1, .7, 1], Disk[{0, 0}, Scaled[{.4,.4}]]},ImageSize -> {12, 12}, Frame -> False]*),""]	

makePrettySummary[summaryData_]:=
Module[
	{seenQ,fromName,fromAddress,date,subject,message},
	seenQ = Lookup[summaryData,"Flags"]["Seen"];
	fromName = Lookup[summaryData,"FromName"];
	fromAddress = Lookup[summaryData,"FromAddress"];
	date = Lookup[summaryData,"OriginatingDate"];
	subject = Lookup[summaryData,"Subject"];
	message = Lookup[summaryData,"Body",""];
	Grid[
  			{
				{blueDot[Not@seenQ],makePrettyName[{fromName,fromAddress}],makePrettyDate[date]},
  				{SpanFromAbove,makePrettySubject[subject],SpanFromAbove},
  				{SpanFromAbove,makeprettyMessage[message],SpanFromAbove}
  			},
  			Alignment->Left,
  			ItemSize -> {{1,30, 6}}
	  	]
]
keys={"MessagePosition","MailItem"(*,"Summary"*), "Answered", "Seen", "Draft", "Flagged", "Deleted", \
"Tags", "OriginatingDate", "Subject", "Sender", "From","FromName","FromAddress", "ToList", \
"Body", "Attachments", "CcList","Path" ,"MessageID", "InReplyTo"
};
makePrettyDataset[searchResult_,property_List:{}]:=
(Dataset[searchResult][
  All,KeyTake[ <|
  		"MessagePosition" -> #MessagePosition,
		"Answered"->#Flags["Answered"],
		"Seen"->#Flags["Seen"],
		"Draft"->#Flags["Draft"],
		"Flagged"->#Flags["Flagged"],
		"Deleted"->#Flags["Deleted"],
		"Tags"->#Tags,		  	
  		"OriginatingDate" -> #OriginatingDate,
  		"Subject"->#Subject,
  		"Sender"->#Sender, 
  		"From" -> #From, 
    	"ToList" -> #ToList,
    	"MailItem"->#MailItem,  
    	"Body" -> #Body, 
    	"Attachments" -> KeyTake[#Attachments, {"FileName", "FileSize", "FileType"}],
    	"Path"->#Path, 
    	"CcList" -> #CcList, "MessageID" -> #MessageID, "InReplyTo" -> #InReplyTo
     |>,If[property==={},keys,property]] &])
End[] (* End Private Context *)

EndPackage[](* Wolfram Language package *)