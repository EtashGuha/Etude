(* Wolfram Language Package *)

BeginPackage["MailLink`"]
(* Exported symbols added here with SymbolName::usage *)  

System`MailServerConnect
System`MailServerConnection
System`MailFolder
System`MailSearch
System`MailItem
System`MailExecute

System`$DefaultMailbox
System`$MailSettings
System`$IncomingMailSettings
Begin["`Private`"] (* Begin Private Context *) 
Needs["IMAPLink`"]
Needs["MailLink`Utilities`"]
Needs["MailLink`icons`"]
System`MailServerConnect::auth = "`1`";
Unprotect[System`$IncomingMailSettings]
Unprotect[System`$DefaultMailbox]

$defaultIncomingMailSettings = <|"Server"->None,"PortNumber"->Automatic,"Username"->None,"Password"->None,"EncryptionProtocol"->Automatic,"MailFolder"->"INBOX"|>;

localIncomingMailSettings=LocalObject["IncomingMailSettings"];


If[
	Quiet@AssociationQ[Check[Get[localIncomingMailSettings],False]],
	System`$IncomingMailSettings=<||>;
	AssociateTo[System`$IncomingMailSettings,Get[localIncomingMailSettings]];
	
	
	,
	
	System`$IncomingMailSettings = $defaultIncomingMailSettings;
	
	Put[System`$IncomingMailSettings,localIncomingMailSettings];
	
]

f[assoc_]:=If[Lookup[assoc,"Password",None]===None,None,Decrypt[MailLink`Utilities`$fetchmailid,Lookup[assoc,"Password",""]]]

(*Setters and getters to modify $IncomingMailSettings*)

MailLink`GetIncomingMailSettings[]:=
Module[
	{},
	If[
		Quiet@AssociationQ[Check[Get[localIncomingMailSettings],False]],
		Get[localIncomingMailSettings]
		,
		
		System`$IncomingMailSettings=$defaultIncomingMailSettings;
		
		Put[System`$IncomingMailSettings,localIncomingMailSettings];
		System`$IncomingMailSettings
	]
]

MailLink`SetIncomingMailSettings[assoc_?AssociationQ]:=
Module[
	{},
	
	Which[
		KeyExistsQ[assoc,"Remember me"] && Not[assoc["Remember me"]],
		System`$IncomingMailSettings=KeyDrop[assoc,"Remember me"]
		,
		KeyExistsQ[assoc,"Remember me"] && assoc["Remember me"],
		Put[
			Join[
				System`$IncomingMailSettings,
				KeyDrop[assoc,{"Password","Remember me"}],
				encryptPassword[assoc]
				
				]
			,
			localIncomingMailSettings
			];
		,
		True,
		Put[
			Join[
				System`$IncomingMailSettings,
				KeyDrop[assoc,{"Password"}],
				encryptPassword[assoc]
				
				]
			,
			localIncomingMailSettings
			];
	];
	System`$IncomingMailSettings = Get[localIncomingMailSettings];
	
]

MailLink`ResetIncomingMailSettings[]:=
Module[
	{},
	
	System`$IncomingMailSettings=<||>;
	AssociateTo[System`$IncomingMailSettings,$defaultIncomingMailSettings];
	Put[System`$IncomingMailSettings,localIncomingMailSettings];
	
	System`$IncomingMailSettings
]

encryptPassword[assoc_?AssociationQ]:=
If[ KeyExistsQ[assoc,"Password"] && StringQ[assoc["Password"]] && StringLength[assoc["Password"]] > 0
	,
	<|"Password"->MailLink`Utilities`encrypt[Lookup[assoc,"Password"],MailLink`Utilities`$fetchmailid]|>
	,
	<||>
	]

(*MailServerConnect*)
Options[System`MailServerConnect]=
	{
		MailSettings->$defaultIncomingMailSettings,
		Authentication->
		<|
			"Username"->$defaultIncomingMailSettings["Username"],
			"Password"->$defaultIncomingMailSettings["Password"]
		|>
	}

System`MailServerConnect[]:=With[
	{res = Catch[
		implMailServerConnect[
			Join[System`$IncomingMailSettings,
				decryptPassword[System`$IncomingMailSettings]]],CURLLink`Utilities`Exception,foo[#,CURLLink`Utilities`Exception]&]},
	
	res/;res=!=False
]


System`MailServerConnect[server_String,opts:OptionsPattern[]]:=
With[{res=Catch[
		System`MailServerConnect[MailSettings->Join[MailLink`Utilities`overWriteDefaults[OptionValue[Authentication],MailLink`Utilities`getMSettings@OptionValue[MailSettings]],<|"Server"->server|>]],CURLLink`Utilities`Exception,foo[#,CURLLink`Utilities`Exception]&]},
				res/;res=!=False
]


System`MailServerConnect[server_String,userName_String,opts:OptionsPattern[]]:=
With[{res=Catch[
	System`MailServerConnect[
		MailSettings->
		Join[
			MailLink`Utilities`overWriteDefaults[OptionValue[Authentication],MailLink`Utilities`getMSettings@OptionValue[MailSettings]],
			<|"Server"->server,"Username"->userName|>]],
			CURLLink`Utilities`Exception,foo[#,CURLLink`Utilities`Exception]&]},
				res/;res=!=False
]

System`MailServerConnect[server_String,userName_String,pass_String,opts:OptionsPattern[]]:=
With[{res=Catch[
		implMailServerConnect[
			Join[$defaultIncomingMailSettings,MailLink`Utilities`overWriteDefaults[OptionValue[Authentication],MailLink`Utilities`getMSettings@OptionValue[MailSettings]],<|"Server"->server,"Username"->userName,"Password"->pass|>
				]],CURLLink`Utilities`Exception,foo[#,CURLLink`Utilities`Exception]&]},
				res/;res=!=False
]
System`MailServerConnect[opts:OptionsPattern[]]:=(With[
	{res = Catch[
		implMailServerConnect[
			Join[$defaultIncomingMailSettings,MailLink`Utilities`getMSettings@OptionValue[MailSettings],
				decryptPassword[Join[$defaultIncomingMailSettings,MailLink`Utilities`getMSettings@OptionValue[MailSettings]]]]],CURLLink`Utilities`Exception,foo[#,CURLLink`Utilities`Exception]&]},
	
	res/;res=!=False
])

decryptPassword[assoc_?AssociationQ]:=Module[{res},
 Which[
 	assoc === $defaultIncomingMailSettings,
	res=AuthenticationDialog[
 		FormObject[
 			{
 			"Server"-><|"Required"->True,"Hint"->"imap.example.com"|>,
 			"Username" -> <|"Required"->True,"Hint"->"userid"|>,
 			"Password" -> <|"Masked" -> True|>,
 			"Folder"-><|"Required"->False,"Hint"->"INBOX"|>,
    		"Remember me" -> "Boolean"
    		}
    		]
    		];
    MailLink`SetIncomingMailSettings[res];
    If[res===$Failed ||res===$Canceled,<||>,res]			
	,
 	Head[assoc["Password"]]===List,
	<|"Password"->MailLink`Utilities`lookupid[Lookup[assoc,"Password"],MailLink`Utilities`$fetchmailid]|>
	,
	
	KeyDrop[assoc,{"Server","Username"}] === KeyDrop[$defaultIncomingMailSettings,{"Server","Username"}],
	res=AuthenticationDialog[
 		FormObject[
 			{
 			"Server"-><|"Required"->True,"Hint"->"imap.example.com"|>,
 			"Username" -> <|"Required"->True,"Hint"->"userid"|>,
 			"Password" -> <|"Masked" -> True|>,
 			"Folder"-><|"Required"->False,"Hint"->"INBOX"|>
    		
    		}
    		][KeyTake[assoc,{"Server","Username"}]]
    		];
    If[res===$Failed ||res===$Canceled,Throw["Canceled",CURLLink`Utilities`Exception],res]		
    		,
	
	KeyDrop[assoc,"Server"] ===KeyDrop[$defaultIncomingMailSettings,"Server"],
	res=AuthenticationDialog[
 		FormObject[
 			{
 			"Server"-><|"Required"->True,"Hint"->"imap.example.com"|>,
 			"Username" -> <|"Required"->True,"Hint"->"userid"|>,
 			"Password" -> <|"Masked" -> True|>,
 			"MailFolder"-><|"Required"->False,"Hint"->"INBOX"|>
    		
    		}
    		][KeyTake[assoc,"Server"]]
    		];
    If[res===$Failed ||res===$Canceled,Throw["Canceled",CURLLink`Utilities`Exception],res]		
	,
	assoc["Password"]===None,
	res=AuthenticationDialog[FormObject[
 			{
 			"Server"-><|"Required"->True,"Hint"->"imap.example.com"|>,
 			"Username" -> <|"Required"->True,"Hint"->"userid"|>,
 			"Password" -> <|"Masked" -> True|>,
 			"MailFolder"-><|"Required"->False,"Hint"->"INBOX"|>
    		
    		}
    		]][KeyTake[assoc,{"Server","Username"}]];
    		If[res===$Failed ||res===$Canceled,Throw["Canceled",CURLLink`Utilities`Exception],res]
	,
	True,
	<||>
	]
]
foo["Canceled",tag_]:=($Canceled)
foo[message_,tag_]:=(Message[System`MailServerConnect::auth, message];$Failed)	

implMailServerConnect[assoc_Association]:=
Module[
	{o,data,assocs,internalID,url,serverConfig = assoc,authentication,port = 993},
	
	url = Lookup[serverConfig,"Server"];
	
	authentication = KeyTake[serverConfig,{"Username","Password"}];
	
	data = IMAPLink`IMAPConnect[Join[assoc,authentication,<|"URL"->url|>]];
	
	internalID = KeyTake[data,"InternalID"];
	
	assocs = Map[#->System`MailFolder[Join[IMAPLink`IMAPExecute[data,"Examine Folder",#],internalID]]&,Lookup[data,"Folders",{}]];
	
	data = Join[data,<|"MailFolderAssociation"->Association[assocs]|>];
	
	o = System`MailServerConnection[data];
	If[MailLink`Utilities`incomingMailSettingsQ[assoc],
		System`$DefaultMailbox=o[Lookup[System`$IncomingMailSettings,"MailFolder","INBOX"]]
		,
		If[Head[System`$DefaultMailbox]=!=System`MailFolder,System`$DefaultMailbox=o[Lookup[assoc,"MailFolder","INBOX"]]]
		];
	o
]

System`MailServerConnection[data_?AssociationQ]["MailFolderAssociation"]:=Lookup[data,"MailFolderAssociation"]
System`MailServerConnection[data_?AssociationQ]["MailFolderList"]:=Values[Lookup[data,"MailFolderAssociation"]]
System`MailServerConnection[data_?AssociationQ][folder_]/;KeyExistsQ[Lookup[data,"MailFolderAssociation"],folder]:=Lookup[data,"MailFolderAssociation"][folder]
System`MailServerConnection[data_?AssociationQ][folder_String]/;Not[KeyExistsQ[Lookup[data,"MailFolderAssociation"],folder]]:=
	System`Failure[
		"MissingMailFolder", 
		<|
  		"MessageTemplate" :> "Mail folder \"`folder`\" could not be found.",
   		"MessageParameters" -> <|"folder" :> folder|>
   		|>
 ]

System`MailFolder[data_Association][messagePosition_Integer/;Positive[messagePosition]]:=
Module[{assoc,totalMessages},
 assoc = IMAPLink`IMAPExecute[data,"Select Folder",data["Path"]];
 totalMessages = ToExpression[assoc["TotalMessageCount"]];
 If[TrueQ[messagePosition <= totalMessages],
 	assoc = assoc~Join~IMAPLink`Utilities`import[IMAPLink`IMAPExecute[data,"FETCH "<>ToString[messagePosition]<>" "<>"(FLAGS)"],"SearchResults","FLAGS"][[1]];
 	System`MailItem[assoc~Join~IMAPLink`IMAPExecute[data,"FetchMail",messagePosition]]
 	,
 	$Failed
 ]
]

System`MailFolder[data_Association][messagePosition_Integer/;Negative[messagePosition]]:=
Module[{assoc,MessagePosition, totalMessages},
 assoc = IMAPLink`IMAPExecute[data,"Select Folder",data["Path"]];
 totalMessages = ToExpression[assoc["TotalMessageCount"]];
 MessagePosition = ToExpression[assoc["TotalMessageCount"]] + messagePosition + 1 ;
 If[TrueQ[MessagePosition > 0],
 	assoc = assoc~Join~IMAPLink`Utilities`import[IMAPLink`IMAPExecute[data,"FETCH "<>ToString[MessagePosition]<>" "<>"(FLAGS)"],"SearchResults","FLAGS"][[1]];
 	System`MailItem[assoc~Join~IMAPLink`IMAPExecute[data,"FetchMail",MessagePosition]]
 	,
 	$Failed
 ]
]

System`MailFolder[data_Association][messagePositions_List]:=
 Map[System`MailFolder[data][#]&,messagePositions]

System`MailFolder[data_Association][All]:=System`MailFolder[data][1;;-1]

System`MailFolder[data_Association][span_Span]:=Module[
	{assoc,list},
	assoc = IMAPLink`IMAPExecute[data,"Select Folder",data["Path"]];
	Check[list = Range[ToExpression@assoc["TotalMessageCount"]][[span]],Return[$Failed]];	
 	Map[System`MailFolder[data][#]&,list]
]
System`MailFolder[data_Association][key_String]/;KeyExistsQ[data,key]:= data[key]

System`MailFolder[data_Association][command_String]:=
Module[
	{assoc},
	assoc = IMAPLink`IMAPExecute[data,"Select Folder",data["Path"]];
	IMAPLink`IMAPExecute[assoc,command]
]
System`MailFolder[data_Association]["Properties"]:=Keys[data]

Options[System`MailSearch]={TimeConstraint->Automatic,MaxItems->Automatic}
System`MailSearch[opts:OptionsPattern[]]:=System`MailSearch[System`$DefaultMailbox,<|"Property"->{},"Deleted"->False,"Seen"->False|>,opts]

System`MailSearch[query_String,opts:OptionsPattern[]]:=System`MailSearch[System`$DefaultMailbox,<|"Text"->query,"Property"->{}|>,opts]
System`MailSearch[query_String,prop_List,opts:OptionsPattern[]]:=System`MailSearch[System`$DefaultMailbox,<|"Text"->query,"Property"->{}|>,opts]

System`MailSearch[query_Association,opts:OptionsPattern[]]:=System`MailSearch[System`$DefaultMailbox,Join[query,<|"Property"->Lookup[query,"Property",{}]|>],opts]
System`MailSearch[query_Association,prop_List,opts:OptionsPattern[]]:=System`MailSearch[System`$DefaultMailbox,Join[query,<|"Property"->prop|>],opts]

System`MailSearch[query_Rule,opts:OptionsPattern[]]:=System`MailSearch[System`$DefaultMailbox,Join[Association[query],<|"Property"->{}|>],opts]
System`MailSearch[query_Rule,prop_List,opts:OptionsPattern[]]:=System`MailSearch[System`$DefaultMailbox,Join[Association[query],<|"Property"->prop|>],opts]

System`MailSearch[folder_System`MailFolder,query_String,prop_List,opts:OptionsPattern[]]/;MatchQ[Lookup[Normal[folder],"ObjectType",None],"MailFolder"]:=
System`MailSearch[folder,<|"Text"->query,"Property"->prop|>,opts]

System`MailSearch[ms_System`MailServerConnection,query_Association,prop_List,opts:OptionsPattern[]]:=
	Join[Sequence@@Map[System`MailSearch[#,Join[query,<|"Property"->prop|>],opts]&,ms["MailFolderList"]]]

System`MailSearch[ms_System`MailServerConnection,query_Association,opts:OptionsPattern[]]:=
	Join[Sequence@@Map[System`MailSearch[#,Join[query,<|"Property"->{}|>],opts]&,ms["MailFolderList"]]]


System`MailSearch[list:{System`MailFolder[_Association]..},query_Association,opts:OptionsPattern[]]:=
	Join[Sequence@@Map[System`MailSearch[#,Join[query,<|"Property"->{}|>],opts]&,list]]

System`MailSearch[list:{System`MailFolder[_Association]..},query_Association,prop_List,opts:OptionsPattern[]]:=
	Join[Sequence@@Map[System`MailSearch[#,Join[query,<|"Property"->prop|>],opts]&,list]]
		
System`MailSearch[folder_System`MailFolder,query_Association,prop_List,opts:OptionsPattern[]]/;MatchQ[Lookup[Normal[folder],"ObjectType",None],"MailFolder"]:=
	System`MailSearch[folder,Join[query,<|"Property"->prop|>],opts]

System`MailSearch[System`MailFolder[data_Association],query_Association,opts:OptionsPattern[]]:=
	Module[
	{res,assoc},
	assoc = IMAPLink`IMAPExecute[data,"Select Folder",data["Path"]];
	res=IMAPLink`IMAPExecute[data,"Search",Join[query,<|MaxItems->OptionValue[MaxItems],TimeConstraint->OptionValue[TimeConstraint]|>]]
	
]
System`MailExecute["Delete",mi:System`MailItem[_Association]]:=
Module[
	{mailitem},
	mailitem=System`MailExecute[Rule["SetFlags",{"Deleted"}],{mi}];
	mailitem=If[ListQ[mailitem],mailitem[[1]]];
	If[mailitem=!=Null,
		If[mailitem["Flags"]["Deleted"],
			System`Success[
			"Delete", 
			<|"MessageTemplate" :>"Message at position `position` has been marked for deletion.", 
  		  	  "MessageParameters" -> <|"position" :>ToString[mailitem["MessagePosition"]]|>
   			|>]
   			,
   			System`Failure[
			"Delete flag could not be set", 
			<|"MessageTemplate" :>"Message at position `position`could not be marked for deletion.", 
  		  	  "MessageParameters" -> <|"position" :>ToString[mailitem["MessagePosition"]]|>
   			|>]
		]
		]
]
System`MailExecute["Delete",list:{System`MailItem[_Association]..}]:=
	If[AllTrue[Map[System`MailExecute["Delete",#]&,list],(Head[#]===System`Success&)],
		System`Success[
			"Delete", 
			<|"MessageTemplate" :>"All given messages were successfully marked for deletion."
   			|>]
   			,
   		System`Failure[
			"Delete flag could not be set", 
			<|"MessageTemplate" :>"Some messages could not be marked for deletion."
   			|>
   			]
	]

System`MailExecute["SetTag"->tag_String,mi:System`MailItem[_Association]]:=
	System`MailExecute[Rule["SetFlags",{tag}],mi]

System`MailExecute["SetTag"->tag_String,list:{System`MailItem[_Association]..}]:=
	Map[System`MailExecute[Rule["SetFlags",{tag}],#]&,list]

System`MailExecute["SetFlags"->flag_String,list:{System`MailItem[_Association]..}]:=
	Map[System`MailExecute[Rule["SetFlags",{flag}],#]&,list]

System`MailExecute["SetFlags"->flag_String,mi:System`MailItem[_Association]]:=
	System`MailExecute[Rule["SetFlags",{flag}],mi]


System`MailExecute["SetFlags"->flags_List,list:{System`MailItem[_Association]..}]:=
	Map[System`MailExecute[Rule["SetFlags",flags],#]&,list]

System`MailExecute[Rule["SetFlags",flags_List],System`MailItem[miAssoc_Association]]:=
Module[
	{flagsAssoc,resp},
	flagsAssoc = Association[Map[Rule[#,True]&,flags]];
	
	resp = IMAPLink`IMAPExecute[miAssoc,"Select Folder",miAssoc["Path"]];
	resp = IMAPLink`IMAPExecute[miAssoc,"SetFlags",flagsAssoc];
	System`MailItem[resp]
	
]
System`MailExecute["ClearFlags",mi:System`MailItem[_Association]]:=
	System`MailExecute[Rule["ClearFlags",System`MailExecute["Flags",mi]],mi]
	
System`MailExecute["ClearFlags",list:{System`MailItem[_Association]..}]:=
	Map[System`MailExecute[Rule["ClearFlags",System`MailExecute["Flags",#]],#]&,list]

System`MailExecute["ClearFlags"->flags_List,list:{System`MailItem[_Association]..}]:=
	Map[System`MailExecute[Rule["ClearFlags",flags],#]&,list]

System`MailExecute["ClearFlags"->flag_String,mi:System`MailItem[_Association]]:=
	System`MailExecute[Rule["ClearFlags",{flag}],mi]
	
System`MailExecute["ClearFlags"->flag_String,list:{System`MailItem[_Association]..}]:=
	Map[System`MailExecute[Rule["ClearFlags",{flag}],#]&,list]

System`MailExecute[Rule["ClearFlags",flags_List],mi:System`MailItem[miAssoc_Association]]:=
Module[
	{resp,flagsAssoc},
	flagsAssoc = Association[Map[Rule[#,False]&,flags]];
	resp = IMAPLink`IMAPExecute[miAssoc,"Select Folder",miAssoc["Path"]];
	resp = IMAPLink`IMAPExecute[miAssoc,"SetFlags",flagsAssoc];
	System`MailItem[resp]
]

System`MailExecute["Flags",item_System`MailItem]:= 
	System`MailExecute["Flags",{item}][[1]]

System`MailExecute["Flags",list:{System`MailItem[_Association]..}]:=
Module[
	{itemFlagAssoc,getFlags,res},
	res = implMailExecute["Flags",list];
	getFlags=Function[{mailitem},Keys@Select[Normal[mailitem]["Flags"],#&]];
	itemFlagAssoc = Association[Map[#->getFlags[#]&,res]]
]

System`MailExecute[Rule["Copy",folder_System`MailFolder],list:{System`MailItem[assoc_Association]..}]:=
	Map[System`MailExecute["Copy",#,Normal[folder]["Path"]]&,list]


System`MailExecute[Rule["Copy",folder_System`MailFolder ],System`MailItem[assoc_Association]]:=
	System`MailExecute[Rule["Copy",Normal[folder]["Path"]],System`MailItem[assoc]]

System`MailExecute[Rule["Copy",folder_String],System`MailItem[assoc_Association]]:=
Module[
	{resp,query,res},
	resp = IMAPLink`IMAPExecute[assoc,"Select Folder",assoc["Path"]];
	
	IMAPLink`IMAPExecute[resp,"Copy",folder,assoc["MessagePosition"]];
	
	resp = IMAPLink`IMAPExecute[resp,"Select Folder",folder];
	
	query = <|"MessageID"->assoc["MessageID"],"Return"->"MailItem"|>;
	
	res = IMAPLink`IMAPExecute[resp,"Search",query];
	
	res

]
System`MailExecute["Expunge",folders_List]:=
	If[AllTrue[Map[System`MailExecute["Expunge",#]&,folders],(Head[#]===System`Success&)],
		System`Success[
			"Expunge", 
			<|"MessageTemplate" :>"Messages have been deleted."
   			|>]
   			,
   			Failure[
			"Expunge", 
			<|"MessageTemplate" :> "Some messages could not be deleted" |>
			]
	]
System`MailExecute["Expunge",System`MailFolder[assoc_Association]]:=
Module[
	{resp,res,internalID},
	resp = IMAPLink`IMAPExecute[assoc,"Select Folder",assoc["Path"]];
	
	internalID = KeyTake[assoc,"InternalID"];
	
	res = IMAPLink`IMAPExecute[assoc,"Expunge"];
	
	If[
		res["SuccessQ"],
		System`Success[
			"Expunge", 
			<|"MessageTemplate" :>"`count`", 
  		  	  "MessageParameters" -> <|"count" :>Which[
				res["DeletedMessageCount"] > 1, 
				ToString[res["DeletedMessageCount"]]<>" messages deleted."
				,
				res["DeletedMessageCount"] === 1,
				ToString[res["DeletedMessageCount"]]<>" message deleted."
				,
				True,
				"No messages were deleted"
				] |>
   			|>
   		],
   		Failure[
			"Some messages could not be deleted", 
			<|"MessageTemplate" :> "Message deleted `count`.", 
  		  	  "MessageParameters" -> <|"count" -> res["DeletedMessageCount"]|> 
   			|>
   		]
	]

]
System`MailExecute[Rule["Create",folder_String],System`MailFolder[assoc_Association]]:=
Module[
	{resp,res,internalID,fullFolder},
	resp = IMAPLink`IMAPExecute[assoc,"Select Folder",assoc["Path"]];
	
	internalID = KeyTake[assoc,"InternalID"];
	
	fullFolder = StringJoin[{assoc["Path"],"/",folder}];
	
	res = IMAPLink`IMAPExecute[assoc,StringJoin[{"CREATE"," ",fullFolder}]];
	
	res = System`MailFolder[Join[IMAPLink`IMAPExecute[assoc,"Examine Folder",fullFolder],internalID]];
	
	res

]
System`MailExecute[Rule["Create",folder_String],System`MailServerConnection[assoc_Association]]:=
Module[
	{resp,res,internalID},
	
	resp = IMAPLink`IMAPExecute[assoc,"Select Folder",assoc["INBOX"]];
	
	internalID = KeyTake[assoc,"InternalID"];
	
	res = IMAPLink`IMAPExecute[assoc,StringJoin[{"CREATE"," ",folder}]];
	
	res = System`MailFolder[Join[IMAPLink`IMAPExecute[assoc,"Examine Folder",folder],internalID]];
	
	res

]

implMailExecute["Flags",list:{System`MailItem[_Association]..}]:= Map[implMailExecute["Flags",#]&,list]

implMailExecute["Flags",System`MailItem[data_]]:=
Module[
	{assoc},
	assoc = IMAPLink`IMAPExecute[data,"Select Folder",data["Path"]];
	assoc = IMAPLink`IMAPExecute[data,"FetchFlags",data["MessagePosition"]];
	System`MailItem[assoc]
]


System`MailExecute["Download",list:{System`MailItem[_]..}]:=Map[System`MailExecute[#,"Download"]&,list]

System`MailExecute["Download",System`MailItem[data_]]:=
Module[
	{assoc},
	assoc = IMAPLink`IMAPExecute[data,"FetchMail",data["MessagePosition"]];
	System`MailItem[assoc]
]



System`MailItem[data_?AssociationQ][key_String]:= data[key]
System`MailItem[data_?AssociationQ]["Properties"]:= Keys[data]

Unprotect[Append]
Append[System`MailItem[x_],y_]:=System`MailItem[Append[x,y]]
Protect[Append]

Unprotect[Normal]
Normal[System`MailItem[x_]]:=x
Normal[System`MailServerConnection[data_]]:=data
Normal[System`MailFolder[data_]]:=data
Protect[Normal]




End[] (* End Private Context *)

EndPackage[](* Wolfram Language package *)