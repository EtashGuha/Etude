
Get["PushbulletAPIFunctions.m"]

Begin["PushbulletAPI`"]

ServiceExecute::invtoken="Invalid token";
ServiceExecute::fufld="File upload failed";
ServiceExecute::invimg="Invalid image";
ServiceExecute::noopen="Cannot open file at `1`."

Begin["`Private`"]

(******************************* Pushbullet *************************************)

(* Authentication information *)

pushbulletdata[]:={
		"ServiceName" 		-> "Pushbullet", 
        "URLFetchFun"		:> (With[{params=Lookup[{##2},"Parameters",{}]},
        	(URLFetch[#1,{"Content","StatusCode"},Sequence@@FilterRules[{##2},Except["Parameters"]],  
        		"Username"-> Lookup[params,"apikey",""], "Parameters" -> FilterRules[params,Except["apikey"]]])]&)
        	,
        "ClientInfo"		:> OAuthDialogDump`Private`MultipleKeyDialog["Pushbullet",{"Access Token"->"apikey"},
        								"https://www.pushbullet.com/signin","https://www.pushbullet.com/tos"],
	 	"Gets"				-> {"Devices", "PushHistory","Contacts","UserData"},
	 	"Posts"				-> {"PushNote", "PushHyperlink", "PushFile","PushImage","SendSMS"},
	 	"RawGets"			-> {"RawHistory", "RawDevices","RawContacts","RawUserData"},
	 	"RawPosts"			-> {"RawPush","RawUploadRequest","RawSendSMS"},
 		"Information"		-> "Use Pushbullet with Wolfram Language"
}

pushbulletimport[rawdata_]:=ImportString[ToString[rawdata[[1]],CharacterEncoding->"UTF-8"],"RawJSON"]

(* Raw *)
pushbulletdata["RawPush"] := {
        "URL"				-> "https://api.pushbullet.com/v2/pushes",
        "BodyData"			-> {"ParameterlessBodyData"->"Data"},
        "HTTPSMethod"		-> "POST",
        "Headers"			-> {"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"Data"},
        "ResultsFunction"	-> pushbulletimport
    }
   
pushbulletdata["RawHistory"] := {
        "URL"				-> "https://api.pushbullet.com/v2/pushes",        
        "HTTPSMethod"		-> "GET",
        "Headers"			-> {"Content-Type" -> "application/json"},
        "Parameters"		-> {"modified_after"},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> pushbulletimport
    }
    
pushbulletdata["RawDevices"] := {
        "URL"				-> "https://api.pushbullet.com/v2/devices",        
        "HTTPSMethod"		-> "GET",
        "Headers"			-> {"Content-Type" -> "application/json"},        
        "RequiredParameters"-> {},
        "ResultsFunction"	-> pushbulletimport
    }
    
pushbulletdata["RawContacts"] := {
        "URL"				-> "https://api.pushbullet.com/v2/contacts",        
        "HTTPSMethod"		-> "GET",
        "Headers"			-> {"Content-Type" -> "application/json"},        
        "RequiredParameters"-> {},
        "ResultsFunction"	-> pushbulletimport
    }
    
pushbulletdata["RawUploadRequest"] := {
        "URL"				-> "https://api.pushbullet.com/v2/upload-request",
        "BodyData"			-> {"ParameterlessBodyData"->"Data"},
        "HTTPSMethod"		-> "POST",
        "Headers"			-> {"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"Data"},
        "ResultsFunction"	-> pushbulletimport
    }
    
pushbulletdata["RawUserData"] := {
        "URL"				-> "https://api.pushbullet.com/v2/users/me",
        "HTTPSMethod"		-> "GET",
        "Headers"			-> {"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> pushbulletimport
    }

pushbulletdata["RawSendSMS"] := {
        "URL"				-> "https://api.pushbullet.com/v2/ephemerals",
        "BodyData"			-> {"ParameterlessBodyData"->"Data"},
        "HTTPSMethod"		-> "POST",
        "Headers"			-> {"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"Data"},
        "ResultsFunction"	-> pushbulletimport
    }
       
(* Cooked *)

(*Sends a note containing a title and a body*)
pushbulletcookeddata["PushNote", id_, args_] := Block[{invalidParameters, data="\"type\":\"note\"", title, body, device, email, response},

	invalidParameters = Select[Keys[args],!MemberQ[{"Title","Body","Device","Email"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Pushbullet"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Title"],
	(
		title = Lookup[args,"Title"];
		If[!StringQ[title], title = ToString[title]];
		
		data = StringJoin[{data, ",", "\"title\":\"" <> title <> "\""}];
		
	),
		data = StringJoin[{data, ",", "\"title\":\"" <> "Untitled" <> "\""}];
	];
	
	If[KeyExistsQ[args,"Body"],
	(
		body = Lookup[args,"Body"];
		If[!StringQ[body], body = ToString[body]];
		
		data = StringJoin[{data, ",", "\"body\":\"" <> body <> "\""}];
		
	),
		data = StringJoin[{data, ",", "\"body\":\"" <> "" <> "\""}];
	];
	
	If[KeyExistsQ[args,"Device"],
	(
		device = Lookup[args,"Device"];
		If[!StringQ[device], device = ToString[device]];
		
		data = StringJoin[{data, ",", "\"device_iden\":\"" <> device <> "\""}];
		
	)];
	
	If[KeyExistsQ[args,"Email"],
	(
		email = Lookup[args,"Email"];
		If[!StringQ[email], email = ToString[email]];
		
		data = StringJoin[{data, ",", "\"email\":\"" <> email <> "\""}];
		
	)];
		
	data = StringJoin[{"{", data, "}"}];
	
	response = KeyClient`rawkeydata[id,"RawPush",{"Data" -> data}];
	
	If[MatchQ[response[[2]],200],Association[formatrespone@pushbulletimport[response]],maperrorcode[response]]
	
]

(*Sends a hyperlink containing title, body and url*)
pushbulletcookeddata["PushHyperlink", id_, args_] := Block[{invalidParameters, data="\"type\":\"link\"", title, body, url, device, email, response},

	invalidParameters = Select[Keys[args],!MemberQ[{"Title","Body","URL","Device","Email"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Pushbullet"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Title"],
	(
		title = Lookup[args,"Title"];
		If[!StringQ[title], title = ToString[title]];
		
		data = StringJoin[{data, ",", "\"title\":\"" <> title <> "\""}];
		
	),
		data = StringJoin[{data, ",", "\"title\":\"" <> "Untitled" <> "\""}];
	];
	
	If[KeyExistsQ[args,"Body"],
	(
		body = Lookup[args,"Body"];
		If[!StringQ[body], body = ToString[body]];
		
		data = StringJoin[{data, ",", "\"body\":\"" <> body <> "\""}];
		
	),
		data = StringJoin[{data, ",", "\"body\":\"" <> "" <> "\""}];
	];
	
	If[KeyExistsQ[args,"URL"],
	(
		url = Lookup[args,"URL"];
		If[!MatchQ[url,_String|_Hyperlink],
			Message[ServiceExecute::nval,"URL","Pushbullet"];
			Throw[$Failed]
		];
		If[Head[url]==Hyperlink, url = Last[url]];

		data = StringJoin[{data, ",", "\"url\":\"" <> url <> "\""}];
		
	),
	(
		Message[ServiceExecute::nparam,"URL","Pushbullet"];
		Throw[$Failed]
	)
	];
	
	If[KeyExistsQ[args,"Device"],
	(
		device = Lookup[args,"Device"];
		If[!StringQ[device], device = ToString[device]];
		
		data = StringJoin[{data, ",", "\"device_iden\":\"" <> device <> "\""}];
		
	)];
	
	If[KeyExistsQ[args,"Email"],
	(
		email = Lookup[args,"Email"];
		If[!StringQ[email], email = ToString[email]];
		
		data = StringJoin[{data, ",", "\"email\":\"" <> email <> "\""}];
		
	)];
	
	data = StringJoin[{"{", data, "}"}];

	response = KeyClient`rawkeydata[id,"RawPush",{"Data" -> data}];
	
	If[MatchQ[response[[2]],200],Association[formatrespone@pushbulletimport[response]],maperrorcode[response]]
	
]

(*Sends an address containing name and address*)
(*pushbulletcookeddata["PushAddress", id_, args_] := Block[{data="\"type\":\"address\"", name, address, device, email, response},
	
	Which[Head[args]===List,
		If[KeyExistsQ[args,"Name"],
		(
			name = "Name" /. args;
			If[Head[name] =!= String, name = ToString[name]];
			
			data = StringJoin[{data, ",", "\"name\":\"" <> name <> "\""}];
			
		),
			data = StringJoin[{data, ",", "\"name\":\"" <> "Unnamed" <> "\""}];
		];
		
		If[KeyExistsQ[args,"Address"],
		(
			address = "Address" /. args;
			If[Head[address] =!= String, address = ToString[address]];
			
			data = StringJoin[{data, ",", "\"address\":\"" <> address <> "\""}];
			
		),
			data = StringJoin[{data, ",", "\"address\":\"" <> "" <> "\""}];
		];
		
		If[KeyExistsQ[args,"Device"],
		(
			device = "Device" /. args;
			If[Head[device] =!= String, device = ToString[device]];
			
			data = StringJoin[{data, ",", "\"device_iden\":\"" <> device <> "\""}];
			
		)];
		
		If[KeyExistsQ[args,"Email"],
		(
			email = "Email" /. args;
			If[Head[email] =!= String, email = ToString[email]];
			
			data = StringJoin[{data, ",", "\"email\":\"" <> email <> "\""}];
			
		)]
		,
		
		Head[args]===String,
		
		data = StringJoin[{data, ",", "\"name\":\"Unnamed\", \"address\":\"" <> args <> "\""}];
	];
	
	data = StringJoin[{"{", data, "}"}];
	
	(*ServiceExecute["Pushbullet", "RawPush", {"Data" -> data}]*)
	response = KeyClient`rawkeydata[id,"RawPush",{"Data" -> data}];
	
	If[response[[2]]==200,pushbulletimport[response],maperrorcode[response]]
	
]

(*Sends a checklist containing title and items*)
pushbulletcookeddata["PushChecklist", id_, args_] := Block[{data="\"type\":\"list\"", title, items, device, email, response},
	
	Which[Head[args]===List,
		If[KeyExistsQ[args,"Title"],
		(
			title = "Title" /. args;
			If[Head[title] =!= String, title = ToString[title]];
			
			data = StringJoin[{data, ",", "\"title\":\"" <> title <> "\""}];
			
		),
			data = StringJoin[{data, ",", "\"title\":\"" <> "Untitled" <> "\""}];
		];
		
		If[KeyExistsQ[args,"Items"],
		(
			items = "Items" /. args;
			If[Head[items] === String, 
				items = StringReplacePart[ToString[("\""<>#<>"\""&)/@(StringTrim/@StringSplit[items,","])],{"[","]"},{{1,1},{-1,-1}}];,
				If[Head[items]==List,
					items = StringReplacePart[ToString[("\""<>ToString[#]<>"\""&)/@items],{"[","]"},{{1,1},{-1,-1}}];
				]
			];
			
			data = StringJoin[{data, ",", "\"items\":" <> items}];
			
		),
			items = StringReplacePart[ToString[("\""<>ToString[#]<>"\""&)/@args],{"[","]"},{{1,1},{-1,-1}}];
			data = StringJoin[{data, ",", "\"items\":" <> items}];
		];
		
		If[KeyExistsQ[args,"Device"],
		(
			device = "Device" /. args;
			If[Head[device] =!= String, device = ToString[device]];
			
			data = StringJoin[{data, ",", "\"device_iden\":\"" <> device <> "\""}];
			
		)];
		
		If[KeyExistsQ[args,"Email"],
		(
			email = "Email" /. args;
			If[Head[email] =!= String, email = ToString[email]];
			
			data = StringJoin[{data, ",", "\"email\":\"" <> email <> "\""}];
			
		)]
		,
		
		Head[args]===String,
		
		items = StringReplacePart[ToString[("\""<>#<>"\""&)/@(StringTrim/@StringSplit[args,","])],{"[","]"},{{1,1},{-1,-1}}];
		
		data = StringJoin[{data, ",", "\"title\":\"Untitled\", \"items\":" <> items }];
		
	];
	
	data = StringJoin[{"{", data, "}"}];
	
	(*ServiceExecute["Pushbullet", "RawPush", {"Data" -> data}]*)
	response = KeyClient`rawkeydata[id,"RawPush",{"Data" -> data}];
	
	If[response[[2]]==200,pushbulletimport[response],maperrorcode[response]]
	
]*)

(* Push a file *)
pushbulletcookeddata["PushFile", id_, args_] := Block[{invalidParameters, data="\"type\":\"file\"", uploadRequestData, uploadRequestResponse, filePath, mimeType, fileName, fileType,
														fileURL, uploadHeaders, uploadBody, body, device, boundary, characters, uploadStatus, email, response},

	invalidParameters = Select[Keys[args],!MemberQ[{"FilePath","Body","Device","Email"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Pushbullet"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"FilePath"],
	(
		filePath = Lookup[args,"FilePath"];
		If[!FileExistsQ[filePath],
			Message[ServiceExecute::noopen,filePath];
			Throw[$Failed]
		];
	),
	(
		Message[ServiceExecute::nparam,"FilePath"];
		Throw[$Failed]
	)
	];
	
	If[KeyExistsQ[args,"Body"],
	(
		body = Lookup[args,"Body"];
		If[!StringQ[body], body = ToString[body]];
		
		data = StringJoin[{data, ",", "\"body\":\"" <> body <> "\""}];
		
	),
		data = StringJoin[{data, ",", "\"body\":\"" <> "" <> "\""}];
	];
	
	If[KeyExistsQ[args,"Device"],
	(
		device = Lookup[args,"Device"];
		If[!StringQ[device], device = ToString[device]];
		
		data = StringJoin[{data, ",", "\"device_iden\":\"" <> device <> "\""}];
		
	)];
	
	If[KeyExistsQ[args,"Email"],
	(
		email = Lookup[args,"Email"];
		If[!StringQ[email], email = ToString[email]];
		
		data = StringJoin[{data, ",", "\"email\":\"" <> email <> "\""}];
		
	)];
	
	mimeType = mapFileExtensionToMIMEType[filePath];
	fileName = FileNameTake[filePath];
	
	(* step 1 upload request *)
	uploadRequestData = "{\"file_name\":\"" <> fileName <> "\",\"file_type\":\"" <> mimeType <> "\"}";
	
	uploadRequestResponse = KeyClient`rawkeydata[id,"RawUploadRequest",{"Data" -> uploadRequestData}];
	
	If[MatchQ[uploadRequestResponse[[2]],200],uploadRequestResponse = pushbulletimport[uploadRequestResponse],maperrorcode[uploadRequestResponse]];

	{fileType,fileURL} = Lookup[uploadRequestResponse, {"file_type", "file_url"}];
	
	(* step 2 upload file *)
	characters = Join[CharacterRange["a", "z"], CharacterRange["A", "Z"],ToString /@ Range[0, 9]];
	boundary = RandomChoice[characters, 12] // StringJoin;
	(*Print[buildMultipartDataBody[uploadRequestResponse, filePath, boundary]];*)
	uploadBody = ToCharacterCode[buildMultipartDataBody[uploadRequestResponse, filePath, boundary]];
	uploadHeaders = {"Content-type" -> "multipart/form-data, boundary=" <> boundary, 
					"Content-Length" -> ToString@Length[uploadBody]};
	(*Print[uploadHeaders];*)
	uploadStatus = URLFetch[Lookup[uploadRequestResponse, "upload_url"], "Method" -> "POST", "Headers" -> uploadHeaders, "BodyData" -> uploadBody];
	If[uploadStatus === "",
	(
		data = StringJoin[{data, ",", "\"file_name\":\"" <> fileName <> "\""}];
		data = StringJoin[{data, ",", "\"file_type\":\"" <> fileType <> "\""}];
		data = StringJoin[{data, ",", "\"file_url\":\"" <> fileURL <> "\""}];
		data = StringJoin[{"{", data, "}"}];
		response = KeyClient`rawkeydata[id,"RawPush",{"Data" -> data}];
	
		If[MatchQ[response[[2]],200],Association[formatrespone@pushbulletimport[response]],maperrorcode[response]]
	)
	,
	(
		Message[ServiceExecute::fufld];
		Throw[$Failed](* file upload failed *)
	)]	
]

(* Push a WL Image element *)
pushbulletcookeddata["PushImage", id_, args_] := Block[{invalidParameters, data="\"type\":\"file\"", uploadRequestData, uploadRequestResponse, image, imageName, mimeType, fileType,
														fileURL, uploadHeaders, uploadBody, body, device, boundary, characters, uploadStatus, email, response},

	invalidParameters = Select[Keys[args],!MemberQ[{"Image","ImageName","Body","Device","Email"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Pushbullet"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Image"],
	(
		image = Lookup[args,"Image"];		
	),
	(
		Message[ServiceExecute::nparam,"Image"];
		Throw[$Failed];
	)];
	
	If[KeyExistsQ[args,"ImageName"],
	(
		imageName = Lookup[args,"ImageName"];			
	),
		imageName = "Untitled.jpeg";
	];
	
	If[KeyExistsQ[args,"Body"],
	(
		body = Lookup[args,"Body"];
		If[!StringQ[body], body = ToString[body]];
		
		data = StringJoin[{data, ",", "\"body\":\"" <> body <> "\""}];
		
	),
		data = StringJoin[{data, ",", "\"body\":\"" <> "" <> "\""}];
	];
	
	If[KeyExistsQ[args,"Device"],
	(
		device = Lookup[args,"Device"];
		If[!StringQ[device], device = ToString[device]];
		
		data = StringJoin[{data, ",", "\"device_iden\":\"" <> device <> "\""}];
		
	)];
	
	If[KeyExistsQ[args,"Email"],
	(
		email = Lookup[args,"Email"];
		If[!StringQ[email], email = ToString[email]];
		
		data = StringJoin[{data, ",", "\"email\":\"" <> email <> "\""}];
		
	)];

	image = Quiet[Image[image]];
	If[!ImageQ[image],
	(
		Message[ServiceExecute::invimg];
		Throw[$Failed](* file upload failed *)
	)];
	
	mimeType = "image/jpeg";
	
	(* step 1 upload request *)
	uploadRequestData = "{\"file_name\":\"" <> imageName <> "\",\"file_type\":\"" <> mimeType <> "\"}";
	uploadRequestResponse = ServiceExecute["Pushbullet", "RawUploadRequest", {"Data" -> uploadRequestData}];
	
	{fileType,fileURL} = Lookup[uploadRequestResponse, {"file_type", "file_url"}];
	
	(* step 2 upload file *)
	characters = Join[CharacterRange["a", "z"], CharacterRange["A", "Z"],ToString /@ Range[0, 9]];
	boundary = RandomChoice[characters, 12] // StringJoin;
	uploadBody = ToCharacterCode[imageBuildMultipartDataBody[uploadRequestResponse, image, boundary]];
	uploadHeaders = {"Content-type" -> "multipart/form-data, boundary=" <> boundary, 
					"Content-Length" -> ToString@Length[uploadBody]};
					
	uploadStatus = URLFetch[Lookup[uploadRequestResponse, "upload_url"], "Method" -> "POST", "Headers" -> uploadHeaders, "BodyData" -> uploadBody];
	
	If[uploadStatus === "",
	(
		data = StringJoin[{data, ",", "\"file_name\":\"" <> imageName <> "\""}];
		data = StringJoin[{data, ",", "\"file_type\":\"" <> fileType <> "\""}];
		data = StringJoin[{data, ",", "\"file_url\":\"" <> fileURL <> "\""}];
		data = StringJoin[{"{", data, "}"}];
		response = KeyClient`rawkeydata[id,"RawPush",{"Data" -> data}];
	
		If[MatchQ[response[[2]],200],Association[formatrespone@pushbulletimport[response]],maperrorcode[response]]
	)
	,
	(
		Message[ServiceExecute::fufld];
		Throw[$Failed](* file upload failed *)
	)]
]

pushbulletcookeddata["Devices", id_, args_] := Block[{data, response},
	response = KeyClient`rawkeydata[id,"RawDevices"];

	If[MatchQ[response[[2]],200],data = pushbulletimport[response],maperrorcode[response]];

	data = (formatrespone/@Lookup[data,"devices",{}]);
	Dataset[Association /@ data]
]

pushbulletcookeddata["Contacts", id_, args_] := Block[{data, response},
	response = KeyClient`rawkeydata[id,"RawContacts"];
	
	If[MatchQ[response[[2]],200],data = pushbulletimport[response],maperrorcode[response]];

	data = (formatrespone/@Lookup[data,"contacts",{}]);
	Dataset[Association /@ data]
	
]

pushbulletcookeddata["PushHistory", id_, args_] := Block[{data, modifiedafter, response},
	
	If[KeyExistsQ[args,"ModifiedAfter"],
		(
			modifiedafter = Lookup[args,"ModifiedAfter"];
			response = KeyClient`rawkeydata[id,"RawHistory",{"modified_after" -> ToString[UnixTime[modifiedafter]]}];

			If[MatchQ[response[[2]],200],data = pushbulletimport[response],maperrorcode[response]];

		),
			response = KeyClient`rawkeydata[id,"RawHistory"];

			If[MatchQ[response[[2]],200],data = pushbulletimport[response],maperrorcode[response]];

		];

	data = (formatrespone/@Lookup[data,"pushes",{}]);
	Dataset[Association /@ data]
	
]

pushbulletcookeddata["UserData", id_, args_] := Block[{response},
	response = KeyClient`rawkeydata[id,"RawUserData"];
	
	If[MatchQ[response[[2]],200],Association[formatrespone@pushbulletimport[response]],maperrorcode[response]]
]

pushbulletcookeddata["SendSMS", id_, args_] := Block[{invalidParameters,data,device,response,pnumber,message,userData,userID},

	invalidParameters = Select[Keys[args],!MemberQ[{"Device","PhoneNumber","Message"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Pushbullet"]&/@invalidParameters;
			Throw[$Failed]
		)];	

	If[KeyExistsQ[args,"Device"],
	(
		device = ToString[Lookup[args,"Device"]];
	),
	(
		Message[ServiceExecute::nparam,"Device"];			
		Throw[$Failed]
	)];
		
	If[KeyExistsQ[args,"PhoneNumber"],
	(
		pnumber = ToString[Lookup[args,"PhoneNumber"]];
	),
	(
		Message[ServiceExecute::nparam,"PhoneNumber"];			
		Throw[$Failed]
	)];	
	
	If[KeyExistsQ[args,"Message"],
	(
		message = ToString[Lookup[args,"Message"]];
	),
	(
		Message[ServiceExecute::nparam,"Message"];			
		Throw[$Failed]
	)];	
	
	userData = KeyClient`rawkeydata[id,"RawUserData"];
	userID = Lookup[pushbulletimport[userData],"iden"];
	data = "{\"type\":\"push\",\"push\":{\"type\":\"messaging_extension_reply\",\"package_name\":\"com.pushbullet.android\",\
			\"source_user_iden\":\"" <> userID <> "\",\"target_device_iden\":\"" <> device <> "\",\"conversation_iden\":\"" 
			<> pnumber <> "\",\"message\":\"" <> message <> "\"}}";
	
	response = KeyClient`rawkeydata[id,"RawSendSMS",{"Data" -> data}];
	If[MatchQ[response[[2]],200],message,maperrorcode[response]]
]

pushbulletcookeddata[___]:=$Failed

pushbulletsendmessage[id_,message_String]:= pushbulletcookeddata["PushNote", id, {"Body" -> message}]
pushbulletsendmessage[id_,{"SMS",message_String,device_String,pnumber_}]:= pushbulletcookeddata["SendSMS", id, {"Message" -> message,"Device"->device,"PhoneNumber"->pnumber}]
pushbulletsendmessage[id_,hyperlink_Hyperlink]:= pushbulletcookeddata["PushHyperlink", id, {"URL" -> hyperlink}]
pushbulletsendmessage[id_,{"URL",url_String}]:= pushbulletcookeddata["PushHyperlink", id, {"URL" -> Hyperlink[url]}]
pushbulletsendmessage[id_,{"File",filepath_String}]:= pushbulletcookeddata["PushFile", id, {"FilePath" -> filepath}]
pushbulletsendmessage[id_,image: _?ImageQ | _Graphics]:= pushbulletcookeddata["PushImage", id, {"Image" -> image}]
pushbulletsendmessage[id_,{"Image",image_}]:= pushbulletcookeddata["PushImage", id, {"Image" -> image}]

pushbulletsendmessage[___]:=$Failed


(* Utilities *)
getallparameters[str_]:=DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.pushbulletdata[str]],
	("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]

camelCase[text_] := Module[{split, partial}, (
	split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    partial = StringJoin[partial];
    partial = StringReplace[partial,RegularExpression["[Uu][Rr][Ll]"]->"URL"];
    partial = StringReplace[partial,RegularExpression["^[Ii][Dd]$"]->"ID"];
    partial
    )]

formatrespone = KeyValueMap[
		Which[
			MatchQ[#1, "modified" | "created"],
				camelCase[#1] -> FromUnixTime[#2, TimeZone -> 0],
			True,
				camelCase[#1] -> #2
		]&]

maperrorcode[response_] := Module[{msg},
(
	If[response == {$Failed}, Throw[$Failed]];
	msg = pushbulletimport[response]["error"]["message"];
	Message[ServiceExecute::serrormsg,msg];
	Throw[$Failed]
)]

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{PushbulletAPI`Private`pushbulletdata,PushbulletAPI`Private`pushbulletcookeddata,PushbulletAPI`Private`pushbulletsendmessage}
