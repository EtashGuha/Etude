(* Wolfram Language Package *)

BeginPackage["IMAPLink`Utilities`"]
(* Exported symbols added here with SymbolName::usage *)  
import
sequesceIDQ
$permFlags
Begin["`Private`"] (* Begin Private Context *) 
sequenceIDQ = Function[{input},MatchQ[input, _Integer] || StringMatchQ[input, NumberString]]
$permFlags={"Seen","Answered","Draft","Deleted","Flagged"};
import[string_String,"Folders"]:=Module[{res, pick},
 res = StringCases[string, Shortest[StartOfLine ~~ "*" ~~ __ ~~ ("\"/\"" |"\".\"") ~~ (((" \"") ~~ name1__ ~~ "\"") | (" " ~~name2__)) ~~ ("\r\n" | "\n" | "\r")] :> If[name1 === "", name2, name1]];
 pick = StringCases[string, Shortest[StartOfLine ~~ "*" ~~ __ ~~ "(" ~~ flags__ ~~")" ~~ __ ~~ ("\"/\"" |"\".\"") ~~ (((" \"") ~~ name1__ ~~ "\"") | (" " ~~name2__)) ~~ ("\r\n" | "\n" | "\r")] :> If[StringContainsQ[flags, "noselect", IgnoreCase -> True], False,True]];
 res = Pick[res, pick];
 DeleteDuplicates[res]
]

import[str_String,"EML"]:=
Module[
	{message},
	message=getMessage[str];
	Quiet[
		Check[
			Join[
				Quiet[
					ImportString[message,{"EML","FullMessageElements"}]
					,
					{Import::noattfmt}
				]
				,
				<|"OriginalMessage"->message|>
			]
			,
			<|"OriginalMessage"->message,"Body"->message|>
		]
	]
]

import[str_String,"SearchResult"]:=
	Select[DeleteDuplicates[StringSplit[str]], StringMatchQ[#, NumberString] &] 

import[str_String,"SelectFolder"] := 
 Module[{f, d,unread},
 	unread = First@StringCases[str,Shortest["COUNT " ~~ count : NumberString .. ~~ ___] :> count];	 
  d = StringCases[
    str, {Shortest[
       StartOfLine ~~ "*" ~~ __ ~~ "[" ~~ info__ ~~ 
        "]" ~~ __ ~~ ("\r\n" | "\n" | "\r")] -> info,
     Shortest[
       StartOfLine ~~ "* " ~~ num : NumberString ~~ " " ~~ 
        info__ ~~ ("\r\n" | "\n" | "\r")] :> info <> " " <> num}, 
    Overlaps -> True];
  
  f = Function[string, 
    Module[{res}, 
     If[(res = 
         StringCases[string, 
          key__ ~~ (Shortest[" " ~~ "(" ~~ list__ ~~ ")"]) :> (key -> 
             StringSplit@list)]) === {}, 
      StringCases[string, 
       key__ ~~ " " ~~ val__ -> (key -> ToExpression@val)], 
      res[[1]]]]];
  
  KeyMap[toWLName,Join[Association[Map[f, d]],<|"UnreadMessageCount"->unread|>]]
  ]

import[folderPath_String,"PrettyFolderName"]:=  makePrettyFolderName[folderPath]

toWLName["EXISTS"]:= "TotalMessageCount"
toWLName[key_String]:= key

makePrettyFolderName[string_]:=Last[StringSplit[string, "/"]]

import[str_String,"ExamineFolder"] := 
Module[
	{res},
	res = 
	StringCases[
		str, 
		Shortest[StartOfLine~~"* STATUS "~~(("\""~~path1__~~"\" ")|(path2__~~" "))~~"(MESSAGES " ~~ messages : NumberString .. ~~ " UNSEEN " ~~ unseen : NumberString .. ~~ ")"] 
		:> 
		<|
		"Path"-> If[path1==="",path2,path1],
    	"Name"-> makePrettyFolderName[If[path1==="",path2,path1]],
   		"TotalMessageCount" -> ToExpression[messages], 
   		"UnreadMessageCount" -> ToExpression[unseen]
   		|>
   		, 
   		Overlaps -> True
   	];
   	
   	If[ListQ[res] && Length[res] >= 1, 
   		res[[1]]
   		,
   		<||> 
   	]
]
   
import[string_String,"FetchHeaders"]:=
Module[
	{stack,pairs,l,res},
	stack = {};
	pairs = {};
	l = DeleteDuplicates[Flatten[StringPosition[string, "(" | ")"]]];
	Table[
		If[StringTake[string, {l[[i]]}] === "(",
			(*then*)
   			AppendTo[stack, l[[i]]],
   			(*else*)
   			If[StringTake[string, {l[[i]]}] === ")",
   				(*then*) 
   				AppendTo[pairs, {Last[stack], l[[i]]}]; 
   				stack = Most[stack]]
   				]
  		,{i, 1, Length[l]}];
	res = First@StringTake[string, MaximalBy[pairs, Differences]];
	res
]

import[string_, "BodyStructure"] :=
 Module[{results},
  results = 
   DeleteDuplicates[
    StringCases[string, 
     Shortest[ "* " ~~ id : NumberString .. ~~ " FETCH" ~~ response__ ~~"\r\n"] :> id->bodyStructureToExpression@response]];
  
  results=Association[results]

  ]
(*gives string enclosed in paranthesis ()*)  
getEnclosedString[string_]:=Module[{stack, pairs,l,res},
  stack = {};
  pairs = {};
  l = DeleteDuplicates[Flatten[StringPosition[string, "(" | ")"]]];
  Table[
  	If[StringTake[string, {l[[i]]}] === "(",
  		(*then*)
    	AppendTo[stack, l[[i]]],
    	(*else*)
    	If[StringTake[string, {l[[i]]}] === ")",
    		(*then*)
     		If[stack =!={},
     			AppendTo[pairs, {Last[stack], l[[i]]}];
     			If[stack === Most[stack],stack={},stack=Most[stack]]
     		]
     	]
     	], {i, 1, Length[l]}];
     res = Map[StringTake[string, #] &, pairs];
     res	
]
getEnclosedString2[string_String] := 
 Module[{stack, pairs, l, res}, stack = {};
  pairs = {};
  l = DeleteDuplicates[Flatten[StringPosition[string, "(" | ")"]]] // MinMax;
  Table[
  	If[StringTake[string, {l[[i]]}] === "(",
  		(*then*)
    	AppendTo[stack, l[[i]]],
    	(*else*)
    	If[StringTake[string, {l[[i]]}] === ")",
    		(*then*)
     		AppendTo[pairs, {Last[stack], l[[i]]}];
     		If[stack === Most[stack], 
     			stack = {}, 
     			stack = Most[stack]]]
     			]
     			, 
     			{i,1, Length[l]}];
  res = Map[StringTake[string, #] &, pairs];
  res
  ]

getBodyStructures[istring_String]:=
Module[
	{wlBodyStructures,ids,strings,string=istring,result,pos},
	ids = getIDs2[string];
	strings = Map[Most[StringSplit[string,#]]&,{"\n","\n\n","\r\n"}];
	pos=Position[Length/@strings,Length[ids]];
	If[Length[pos]>0,pos=pos[[1,1]],Throw[$Failed]];
	strings=strings[[pos]];
	If[strings==={},strings={istring}];
	wlBodyStructures = Map[getWLBodyStructure, strings];
	result=Thread[Rule[ids,wlBodyStructures]];
	result
	
	]	
bodyStructuresToRules[string_]:=
Module[
	{bodystructures},
	bodystructures=getBodyStructures[string];
	Map[bodyStructureToRules,bodystructures]
]
(*envelopeToExpression[s_String] := Module[{tt, quote, env, exp},
  quote = Function[str, If[Not@StringMatchQ[str, {",", "}", "{", "\",\"", " " }], "\"" <> str <> "\"", str]];
 (* tt = StringReplace[s, Shortest["\"" ~~ stuff___ ~~ "(" ~~ some___ ~~ ")" ~~ more___ ~~ "\""] :> "\"" <> stuff <> " " <> some <> " " <> more <> "\""];*)
  tt = StringReplace[s, {"(" -> " { ", ")" -> " } ", "}" -> " } ", "{" -> " { ","\t" -> " ", "\r" -> "", "\n" -> "", "\"" -> " \" ","\\\"" -> ""}];
  tt = StringJoin[Riffle[Map[quote, StringSplit[tt]], ","]];
  tt = StringReplace[
    "{" <> tt <> "}", {"{," -> "{", ", }, }" -> "}}", ", }," -> "},", 
     "\"\"\"" -> "\"Quote\""}];
  env = StringReplace[
    tt, {" { " -> "{", ",}" -> "}", "}<" -> "},<","\\"->"Nothing"}];
  exp = ReplaceAll[ToExpression@env, {"Quote" -> "\"", "NIL" -> None,"Nothing"->Nothing}];
  exp[[4, 2]]
  ]*)
envelopeToExpression[s_String] := 
 Module[{tt, quote, env, exp, pad, list, is = s, table},
  quote = 
   Function[str, 
    If[Not@StringMatchQ[str, {",", "}", "{", "\",\"", " " }], 
     "\"" <> str <> "\"", str]];
  is = StringReplace[is, "\\\"" -> ""];
  list = StringCases[is, Shortest["\"" ~~ stuff__ ~~ "\""]:>stuff];
  
  is = StringReplace[is, 
    Shortest["\"" ~~ stuff__ ~~ "\""] :> "\"XXX\""];
  tt = StringReplace[
    is, {"(" -> " { ", ")" -> " } ", "}" -> " } ", "{" -> " { ", 
     "\t" -> " ", "\r" -> "", "\n" -> "", "\"" -> " \" ", 
     "\\\"" -> ""}];
  table = 
   Table[tt = 
     StringReplace[tt, "XXX" -> list[[i]], 1], {i, 
     1, Length[list]}];
  tt = table[[-1]];
  tt = StringJoin[Riffle[Map[quote, StringSplit[tt]], ","]];
  tt = StringReplace[
    tt, {"{," -> "{", ", }, }" -> "}}", ", }," -> "},", 
     "\"\"\"" -> "\"Quote\""}];
  tt = StringReplace[
    "{" <> tt <> "}", {"{," -> "{", ",},}" -> "}}", ", }," -> "},"}];
  env = StringReplace[
    tt, {" { " -> "{", ",}" -> "}", "}<" -> "},<", "\\" -> "Nothing"}];
  exp = ReplaceAll[
    ToExpression@env, {"Quote" -> "\"", "NIL" -> None, 
     "Nothing" -> Nothing}];
  exp[[4, 2]]
  ]  
bodyStructureToExpression[s_String] := Module[{tt, quote, env, exp},
  quote = 
   Function[str, 
    If[Not@StringMatchQ[str, {",", "}", "{", "\",\"", " " }], 
     "\"" <> str <> "\"", str]];
  tt = StringReplace[
    s, {"(" -> " { ", ")" -> " } ", "}" -> " } ", "{" -> " { ", 
     "\t" -> " ", "\r" -> "", "\n" -> "", "\"" -> " \" "}];
  tt = StringJoin[Riffle[Map[quote, StringSplit[tt]], ","]];
  tt = StringReplace[
    "{" <> tt <> "}", {"{," -> "{", ", }, }" -> "}}", ", }," -> "},", 
     "\"\"\"" -> "\"Quote\""}];
  env = StringReplace[
    tt, {" { " -> "{", ",}" -> "}", "}<" -> "},<"}];
  exp = ReplaceAll[ToExpression@env, {"Quote" -> "\"", "NIL" -> None}];
  exp[[4,2]]
  ]  
getWLBodyStructure[istring_] := Module[
  {d, makeRange, makeSpan, pos, range, span, string},
  makeSpan = Function[l,If[Length[l[[1]]] === 1 && Length[l[[2]]] === 1, {Span[l[[1, 1]], l[[2, 1]]]}, {Sequence @@l[[1]][[1 ;; -2]], Span[l[[1, -1]], l[[2, -1]]]}]];
  makeRange = Function[l,If[Length[l[[1]]] === 1 && Length[l[[2]]] === 1, Map[{#} &, Range[l[[1, 1]], l[[2, 1]]]],Map[Join[l[[1]][[1 ;; -2]], {#}] &, Range[l[[1, -1]], l[[2, -1]]]]]]; 
  d = bodyStructureToExpression[istring]; 
  While[Position[d, "\""] =!= {}, pos = Position[d, "\""];
   pos = Partition[pos, 2];
   span = makeSpan /@ pos;
   range = makeRange /@ pos;
   string = 
    StringReplace[
     StringJoin[
      Riffle[d[[Sequence @@ span[[1]]]], " "]], {"\" " -> "\"", 
      " \"" -> "\""}];
   d = Delete[d, range[[1]]];
   d = Insert[d, string, range[[1, 1]]]];
  Map[stripQuotesBS,d,All]
  ] 
stripQuotesBS[s_String] := StringTrim[s, "\""]

stripQuotesBS[x___] := x   

getWLEnvelope[istring_] := Module[
  {d, makeRange, makeSpan, pos, range, span, string},
  makeSpan = Function[l,If[Length[l[[1]]] === 1 && Length[l[[2]]] === 1, {Span[l[[1, 1]], l[[2, 1]]]}, {Sequence @@l[[1]][[1 ;; -2]], Span[l[[1, -1]], l[[2, -1]]]}]];
  makeRange = Function[l,If[Length[l[[1]]] === 1 && Length[l[[2]]] === 1, Map[{#} &, Range[l[[1, 1]], l[[2, 1]]]],Map[Join[l[[1]][[1 ;; -2]], {#}] &, Range[l[[1, -1]], l[[2, -1]]]]]]; 
  d = envelopeToExpression[istring]; 
  While[Position[d, "\""] =!= {}, pos = Position[d, "\""];
   pos = Partition[pos, 2];
   span = makeSpan /@ pos;
   range = makeRange /@ pos;
   string = 
    StringReplace[
     StringJoin[
      Riffle[d[[Sequence @@ span[[1]]]], " "]], {"\" " -> "\"", 
      " \"" -> "\""}];
   d = Delete[d, range[[1]]];
   d = Insert[d, string, range[[1, 1]]]];
  d
  ]
getEnvelopes[istring_String]:=
Module[
	{strings,string=istring,wlEnvelopes},
	
	strings = Most[StringSplit[string,"\r\n"]];
	If[strings==={},strings={istring}];
	wlEnvelopes = Map[getWLEnvelope, strings];
	
	
	wlEnvelopes	
	]
(**)
envelopesToRules[string_]:=Module[
	{envelopes},
	envelopes = getEnvelopes[string];
	Map[envelopeToRules,envelopes]
]
	
	
import[string_String,"Expunge"]:=
Module[
	{res,successQ},
	successQ=StringContainsQ[string,"OK EXPUNGE completed",IgnoreCase->True];
	res=Merge[
		StringCases[
			string, 
			Shortest["*" ~~ " " ~~ pos : NumberString .. ~~ " " ~~ info__ ~~ "\r\n"] :> info -> ToExpression[pos]], Identity];
	Join[res,<|"DeletedMessageCount"->If[KeyExistsQ[res,"EXPUNGE"],Length@res["EXPUNGE"],0],"SuccessQ"->successQ|>]		
]
import[string_String,imaxItems_,"SearchIds"]:=
	Module[{ids,maxitems=imaxItems},
		
		ids=StringCases[string, " " ~~ id : NumberString .. :> id];
		If[maxitems=!=Automatic,
			ids=Take[ids,UpTo[maxitems]];
		];
		ids=Partition[ids,UpTo[20]];
		StringJoin[Riffle[#, ","]]&/@ids
	] 
import[string_String,"SearchResults",type_] := 
 Module[{res,ids},
  ids = getIDs[string];  
  res=Which[
  	type==="BODYSTRUCTURE",
   	res=bodyStructuresToRules[string];
   	Map[Merge[#,First]&,Thread[List[ids,res]]]
   	,
   	type==="ENVELOPE",
   	res=envelopesToRules[string];
   	Map[Association,Thread[List[ids,res]]]
   	,
   	type==="FLAGS",
   	res=flagsToAssoc[string];
   	Map[Association,Thread[List[ids,res]]]
  ];
 res
  ]
flagsToAssoc[string_]:= Module[
{res},
res=StringCases[string, 
 Shortest[__ ~~ seqID : NumberString .. ~~ " FETCH (FLAGS (" ~~ flags___ ~~ "))\r\n"] :> {seqID,StringSplit/@ StringTrim /@ StringSplit[flags, {"\\"}]}];
 Map[makeFlagAssoc,res]
] 
makeFlagAssoc = 
Function[{data},
  Block[
  	{assoc = <|"Answered" -> False, "Deleted" -> False, "Draft" -> False, "Flagged" -> False, "Seen" -> False|>},
  	assoc=Append[assoc,Association[Map[Rule[#,True]&,Flatten@data[[2]]]]]; 
   <|"MessagePosition"->data[[1]],"Flags" -> assoc,"Tags"->Complement[Keys[assoc],$permFlags]|>
   ]
]
getMessage[string_]:=
Module[
	{s,s1,seqID,message},
	seqID=StringCases[string,StartOfString ~~ "* " ~~ id : NumberString .. ~~ " FETCH" ~~ ___ :>id];
	If[Length[seqID]>0,seqID=seqID[[1]],Throw[$Failed,CURLLink`Utilities`Exception]];
	s = Last[SortBy[getEnclosedString2[string],StringLength]];
	s1 = StringCases[s,"(" ~~ "BODY[" ~~ (NumberString ..|"") ~~ "] " ~~ "{" ~~ x : NumberString .. ~~ "}" ~~ data__ :>StringTake[data, {3, UpTo[ ToExpression[x] + 2]}]];
	message = If[Length[s1]>0,StringJoin[Riffle[StringSplit[s1[[1]], "\r\n"], "\n"]],None];
	message

]

import[string_String,encoding_,charset_,contentType_,"Body"]:=
Module[
	{s, s1, seqID, message},
	
	seqID=StringCases[string,StartOfString ~~ "* " ~~ id : NumberString .. ~~ " FETCH" ~~ ___ :>id];
	
	If[Length[seqID] > 0, seqID=seqID[[1]], Throw[$Failed,CURLLink`Utilities`Exception]];
	
	s = Last[SortBy[getEnclosedString2[string],StringLength]];
	
	Which[
		contentType === "HTML",
		
		s1 = StringCases[s,"(" ~~ "BODY[" ~~ (NumberString .. | "") ~~ "] " ~~ "{" ~~x : NumberString .. ~~ "}" ~~ data__ :> StringTake[data, {3, ToExpression[x] + 2}]];
		
		message = If[Length[s1]>0,s1[[1]],None];
		,
		True,

		s1 = StringCases[s,"(" ~~ "BODY[" ~~ (NumberString ..|"") ~~ "]"~~("<0> "|" ") ~~ "{" ~~ x : NumberString .. ~~ "}" ~~ data__ :>StringTake[data, {3, ToExpression[x] + 2}]];

		message = If[Length[s1]>0,StringJoin[Riffle[StringSplit[s1[[1]], "\r\n"], "\n"]],None];
	];

	message = If[message =!= None, 
				Which[
					encoding==="QUOTED-PRINTABLE",
					
					message = decodeQuotedPrintable[message,charset];
					
					If[contentType === "HTML", 
						message = Quiet@ImportString[message,"HTML"]
					];
					,
					encoding==="BASE64",
					If[contentType==="HTML",
						message = ByteArrayToString[System`BaseDecode[message]];
						message = Quiet@ImportString[message,"HTML"],
						message = ImportString[message, {"Base64", "Text"}, CharacterEncoding -> toWLCharset[charset]];
						If[message === $Failed,
							message = ByteArrayToString[System`BaseDecode[message]]
						]
					];
					,
					True,
					If[contentType === "HTML", message = Quiet@ImportString[message,"HTML"]];
					
				];
				message
				,
				""
			];
	<|"MessagePosition"->seqID,"Body"->message|>			
] 
getIDs[string_]:=
Module[
	{ids},  
	ids = StringCases[string,Shortest["* " ~~ id : NumberString .. ~~ (" FETCH (BODYSTRUCTURE"|" FETCH (ENVELOPE"|" FETCH (FLAGS")] :>"MessagePosition"->id];
  	ids = DeleteDuplicates[Flatten@ids];
  	ids
]
getIDs2[string_]:=
Module[
	{ids},  
	ids = StringCases[string,Shortest["* " ~~ id : NumberString .. ~~ (" FETCH (BODYSTRUCTURE"|" FETCH (ENVELOPE")|" FETCH (FLAGS"] :>id];
  	ids = DeleteDuplicates[Flatten@ids];
  	ids
]
toExpression=Function[{string},ToExpression[
     StringReplace[
      StringReplace[
       string, {" " -> ",", "(" -> "{", ")" -> "}", 
        "NIL" -> "None"}], {"}{" -> "},{"}]]]


dequote=Function[string,StringReplace[string,{"\""->"","FFF"->"\\\""}]]
stripQuotes=Function[string,StringTrim[string,"\""]]
(*
The fields of the envelope structure are in the following
order: date, subject, from, sender, reply-to, to, cc, bcc,
in-reply-to, and message-id.  The date, subject, in-reply-to,
and message-id fields are strings.  The from, sender, reply-to,
to, cc, and bcc fields are parenthesized lists of address
structures.
*)
toEmailAssoc = 
 Function[{address,type}, 
  Association[
  	type<>"Name" -> If[ListQ[address]&&Length[address]>1,If[StringQ@address[[1]],decode@stripQuotes@address[[1]],Missing["NotAvailable"]],Missing["NotAvailable"],Missing["NotAvailable"]],
  	type<>"Address" -> If[Length[address]>=4,If[StringQ[address[[3]]] && StringQ[address[[4]]] ,stripQuotes@address[[3]]<>"@"<>stripQuotes@address[[4]],Missing["NotAvailable"],Missing["NotAvailable"]],Missing["NotAvailable"]],
  	type<>"Username" -> If[Length[address]>=4,If[StringQ[address[[3]]],stripQuotes@address[[3]],Missing["NotAvailable"],Missing["NotAvailable"]],Missing["NotAvailable"]],
  	type<>"Domain" -> If[Length[address]>=4,If[StringQ[address[[4]]],stripQuotes@address[[4]],Missing["NotAvailable"],Missing["NotAvailable"]],Missing["NotAvailable"]]
   ]
   ]
decodeQuotedPrintable[s_,charset_]:=Module[{},
	Quiet@Check[FromCharacterCode[
 		ToCharacterCode[ 
 			StringReplace[StringReplace[s, "=\n" :> ""],"=" ~~ (x_ /; StringMatchQ[x, RegularExpression["(\\d|[a-f])"],IgnoreCase->True]) ~~ y_ /; StringMatchQ[y, RegularExpression["(\\d|[a-f])"],IgnoreCase->True] :>FromCharacterCode[FromDigits[x <> y, 16], "ISO8859-1"]], 
  			"ISO8859-1"]
  			, toWLCharset[charset]],FromCharacterCode[
 		ToCharacterCode[ 
 			StringReplace[StringReplace[s, "=\n" :> ""],"=" ~~ (x_ /; StringMatchQ[x, RegularExpression["(\\d|[a-f])"],IgnoreCase->True]) ~~ y_ /; StringMatchQ[y, RegularExpression["(\\d|[a-f])"],IgnoreCase->True] :>FromCharacterCode[FromDigits[x <> y, 16], "ISO8859-1"]], 
  			"ISO8859-1"]
  			, toWLCharset["ISO8859-1"]]]
  		]
toWLCharset["windows-1252"]="WindowsANSI"
toWLCharset["UTF-8"]:="UTF-8"
toWLCharset["utf-8"]:="UTF-8" 
toWLCharset["us-ascii"]:="ASCII"
toWLCharset["iso-8859-1"]:="ISO8859-1"
toWLCharset[charset_]:="ISO8859-1"
decodeBase64=Function[{encodedString,s},If[StringLength[encodedString] > 0,ImportString[encodedString, {"Base64","Text"}, CharacterEncoding -> "UTF-8"], s]]
decode[subject_]:=Module[{},
			If[StringQ[subject],
				Which[
					StringMatchQ[subject,"=?UTF-8?B?" ~~ __],
			   		StringCases[subject, ("=?UTF-8?B?" ~~ encodedString__ ~~ "?=") | s___ :>decodeBase64[encodedString,s]][[1]]
			   		,
			   		StringMatchQ[subject,__~~"=?UTF-8?Q?" ~~ __],
			   		StringCases[subject, "=?UTF-8?Q?" ~~ encodedString__ ~~ "?=" :>decodeQuotedPrintable[encodedString,"UTF-8"]][[1]]
			   		,
			   		True,
			   		subject
				]
			   ,
			   Missing["NotAvailable"]
			]
		]
		
toEmailAssocs = Function[{adresses,type}, Map[toEmailAssoc[#,type]&, adresses]]
envelopeToRules=Function[envelope,
If[Length[envelope]>=10,
  {
   "OriginatingDate" -> toDateList@stripQuotes@envelope[[1]]
   ,
   "Subject" -> decode[stripQuotes@envelope[[2]]]
   ,
   "From" -> import[toEmailAssocs[envelope[[3]],"From"],"From"]
   ,
   "FromName" -> import[toEmailAssocs[envelope[[3]],"From"],"FromName"]
   ,
   "FromAddress" -> import[toEmailAssocs[envelope[[3]],"From"],"FromAddress"]
   ,
   "Sender" -> import[toEmailAssocs[envelope[[4]],"Sender"],"Sender"]
   ,
   "ReplyTo" -> toEmailAssocs[envelope[[5]],"ReplyTo"]
   ,
   "ToList" -> import[toEmailAssocs[envelope[[6]],"To"],"ToList"]
   ,
   "ToNameList" -> import[toEmailAssocs[envelope[[6]],"To"],"ToNameList"]
   ,
   "ToAddressList" -> import[toEmailAssocs[envelope[[6]],"To"],"ToAddressList"]
   ,
   "CcList" -> import[toEmailAssocs[envelope[[7]],"Cc"],"CcList"]
   ,
   "CcNameList" -> import[toEmailAssocs[envelope[[7]],"Cc"],"CcNameList"]
   ,
   "CcAddressList" -> import[toEmailAssocs[envelope[[7]],"Cc"],"CcAddressList"]
   ,
   "BccList" -> toEmailAssocs[envelope[[8]],"Bcc"]
   ,
   "InReplyTo" -> Lookup[getMessageID[envelope],"InReplyTo"]
   ,
   "MessageID" -> Lookup[getMessageID[envelope],"MessageID"]
   },
  (*Else*) 
  {}
 ]
  ]
  
import[List[assoc_?AssociationQ,rest___],"From"]:=
Which[
	Head[assoc["FromName"]]=!=Missing && Head[assoc["FromAddress"]]=!=Missing,
	assoc["FromName"]<>" <"<>assoc["FromAddress"]<>">"
	,
	Head[assoc["FromUsername"]]=!=Missing && Head[assoc["FromAddress"]]===Missing,
	assoc["FromUsername"]
	,
	True,
	assoc["FromUsername"]<>" <"<>assoc["FromAddress"]<>">"
]  
import[List[assoc_?AssociationQ,rest___],"Sender"]:=
Module[
	{},
	Which[
		Head[assoc["SenderName"]]=!=Missing && Head[assoc["SenderAddress"]]=!=Missing,
		assoc["SenderName"]<>" <"<>assoc["SenderAddress"]<>">"
		,
		Head[assoc["SenderUsername"]]=!=Missing && Head[assoc["SenderAddress"]]=!=Missing,
		assoc["SenderUsername"]<>" <"<>assoc["SenderAddress"]<>">"
		,
		Head[assoc["SenderAddress"]]===Missing && Head[assoc["SenderUsername"]]=!=Missing,
		If[StringQ[assoc["SenderUsername"]],assoc["SenderUsername"],Missing["NotAvailable"]]
		
]
		]

import[List[assoc_],"FromName"]:=decode@assoc["FromName"]
import[List[assoc_],"FromAddress"]:=assoc["FromAddress"]
import[None,"ToList"]:={}
import[None,"ToNameList"]:={}
import[None,"ToAddressList"]:={}
import[list_,"ToList"]:=Map[makeFullAddress[#["ToName"],#["ToUsername"],#["ToAddress"]]&,list]
import[list_,"ToNameList"]:=Map[#["ToName"]&,list]
import[list_,"ToAddressList"]:=Map[#["ToAddress"]&,list]
import[None,"CcList"]:={}
import[None,"CcNameList"]:={}
import[None,"CcAddressList"]:={}
import[list_,"CcList"]:=(Map[makeFullAddress[#["CcName"],#["CcUsername"],#["CcAddress"]]&,list])
import[list_,"CcNameList"]:=Map[#["CcName"]&,list]
import[list_,"CcAddressList"]:=Map[#["CcAddress"]&,list]

makeFullAddress[name_,user_,email_]:=If[Head[name]=!=Missing,name<>" <"<>email<>">",email]

bodyStructureToRules[id_ -> bodystructure_] := 
 Module[{info, attachments}, 
  info = Cases[{bodystructure}, {"TEXT", texttype_, 
      charsetInfo : {"CHARSET", charset_, "FORMAT" | ___, 
         format : ___} | {"FORMAT", format : ___, "CHARSET" | ___, charset_}|None,boundary___| None, None, stringEncoding_ | None, 
      size_, lines_, None, {___} | None, language_ | None, 
      None | ___} :> 
     texttype -> <|"Size" -> size, 
       "Charset" -> If[MatchQ[charsetInfo, None], None, charset], 
       "Encoding" -> stringEncoding, 
       "Command" -> toCommand[id, #,texttype] & /@ 
        Position[bodystructure, size]|>, Infinity];
  attachments = 
   Cases[{bodystructure}, {app_, 
      type_, {___, "NAME", name_, ___} | None, messageID_, 
      mightbeName_, encoding_, filesize_, 
      None, {attach_, {"FILENAME", filename_, ___}} | None, None, 
      None | ___} :> 
     Association@{"FileName" -> 
        If[DeleteDuplicates[{name, filename}] === {}, None, 
         DeleteDuplicates[{name, filename}][[1]]], 
       "FileSize" -> filesize, "FileType" -> type, 
       "Command" -> toCommand[id, #,None] & /@ 
        Position[bodystructure, filesize]}, Infinity];
  Flatten@Join[info, {"Attachments" -> Flatten@attachments}]]
toString = Function[list, StringJoin[ToString /@ Riffle[If[Length[list] > 1, Most[list], {1}], "."]]]
toCommand=Function[{id,position,texttype},
	Which[
		ToUpperCase[texttype]==="PLAIN",
		"FETCH "<>id<>" BODY.PEEK["<>toString[position]<>"]"
		,
		ToUpperCase[texttype]==="HTML",
		"FETCH "<>id<>" BODY.PEEK["<>toString[position]<>"]"
		,
		True,
		"FETCH "<>id<>" BODY.PEEK["<>toString[position]<>"]"
		]
	]
totimezoneNumber = 
 Function[{sign, hh, mm}, (ToExpression[hh] + ToExpression[mm]/60.)*
   If[sign === "-", -1, +1]]

totimezone[timeZone_String]:=
Module[{timezone},
timezone=
StringCases[timeZone, 
 sign : ("+" | "-") 
 ~~ 
 hh : Repeated[NumberString, {2}]
 ~~ 
 mm : Repeated[NumberString, {2}] :> totimezoneNumber[sign, hh, mm]
 ];
If[Length[timezone]>0,timezone[[1]],timezones[StringTrim@timeZone]] 
]    

toDateList[dateString_String]:=Module[{string,timezone},
	timezone=StringCases[dateString,__~~Repeated[NumberString,{1,2}]~~":"~~Repeated[NumberString,{1,2}]~~":"~~Repeated[NumberString,{1,2}]~~residue__:>residue];
	string=StringReplace[StringDelete[dateString,timezone],Repeated[" "]->" "];
	Which[
			StringMatchQ[string,DatePattern[{"DayName", ", ", "Day", " ", "MonthName", " ", "Year", " ", "Hour",":", "Minute", ":", "Second"}]],
			DateObject[DateList[{string, {"DayName", ", ", "Day", " ", "MonthName", " ", "Year", " ", "Hour",":", "Minute", ":", "Second"}}]]
			,
			StringMatchQ[string,DatePattern[{"Day", " ", "MonthName", " ", "Year", " ", "Hour",":", "Minute", ":", "Second"}]],
			DateObject[DateList[{string, {"Day", " ", "MonthName", " ", "Year", " ", "Hour",":", "Minute", ":", "Second"}}]]
	]
   
]
timezones=<|
 "EDT"->"GMT-4",
 "CDT"->"GMT-5", 
 "CST"->"GMT-6",
 "MDT"-> "GMT-6",
 "PDT"->"GMT-7", 
 "PST"->"GMT-8"

|>
getMessageID[env_]:=Module[
	{mids},
	mids=Flatten@Map[checkMessageID,env];
	Which[
		Length[mids]===2,
		<|"InReplyTo"->stripQuotes@mids[[1]],"MessageID"->stripQuotes@mids[[2]]|>
		,
		Length[mids]===1,
		<|"InReplyTo"->Missing["NotAvailable"],"MessageID"->stripQuotes@mids[[1]]|>
		,
		True,
		<|"InReplyTo"->Missing["NotAvailable"],"MessageID"->Missing["NotAvailable"]|>
		]
]
checkMessageID = Function[input, 
  If[Head[input] === String, 
   StringCases[input, 
    RegularExpression[
     "((([a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+(\\.[a-zA-Z0-9!#$%&'*+/=?^_`{|\
}~-]+)*)|(\"(([\\x01-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]|[\\x21\\x23-\\\
x5B\\x5D-\\x7E])|(\\\\[\\x01-\\x09\\x0B\\x0C\\x0E-\\x7F]))*\"))@(([a-\
zA-Z0-9!#$%&'*+/=?^_`{|}~-]+(\\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*)|(\\\
[(([\\x01-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]|[\\x21-\\x5A\\x5E-\\x7E])|(\
\\\\[\\x01-\\x09\\x0B\\x0C\\x0E-\\x7F]))*\\])))"]], Nothing]]
End[] (* End Private Context *)

EndPackage[]