Begin["DropboxOAuth`"]

Begin["`Private`"]

(******************************* Dropbox *************************************)

ServiceExecute::ndir="The specified path `1` is not a directory in the connected Dropbox account. Data will be given for the file instead"
ServiceExecute::nfile="The specified path `1` is not a file in the connected Dropbox account. Data will be given for the directory instead"
ServiceExecute::grext="The graphic could not be exported as the file type, `1`, given in the path."
ServiceExecute::svcerror="The service returned `1` error for the connected Dropbox account"
ServiceExecute::npath="Service error: `1`"


(* Authentication information *)

dropboxdata[]:=
If[TrueQ[OAuthClient`Private`$UseChannelFramework],
	{
        "OAuthVersion"      -> "2.0",
        "ServiceName"       -> "Dropbox", 
        "AuthorizeEndpoint" -> "https://www.dropbox.com/oauth2/authorize", 
        "AccessEndpoint"    -> "https://api.dropbox.com/oauth2/token", 
        "RedirectURI"       -> "WolframConnectorChannelListen",
        "Blocking"          -> False,
        "VerifierLabel"     -> "code",
        "ClientInfo"        -> {"Wolfram","Token"},
        "AuthenticationDialog"	:> "WolframConnectorChannel",
        "AuthorizationFunction"	-> "Dropbox",
        "RedirectURLFunction"	->(#1&),
        "Gets"              -> {},
        "Posts"             -> {
        	"UserData","QuotaInfo","DataUpload","GraphicsUpload","StartUploadSession","AppendToUploadSession",
        	"FinishUploadSession","DirectoryTreePlot","FileSearch","FileData","FileNames","FileContents","DirectoryData",
        	"ImportFile","FileRevisions","GetThumbnail","MoveContents","DeleteContents","CopyContents","CreateFolder","FileRestore"
        	},
        "RawGets"           -> {},
        "RawPosts"          -> {
        	"RawUserData","RawQuotaInfo","RawFileDownload","RawPathData","RawListFolder","RawListFolderContinue",
        	"RawFileUpload","RawFileRestore","RawFilePreviewLink","RawFileLink","RawUploadSessionStart","RawUploadSessionAppend",
        	"RawUploadSessionFinish","RawFileCopy","RawCreateFolder","RawFileRevisions","RawThumbnail","RawMoveContents","RawCopyContents",
        	"RawDeleteContents","RawFileSearch","RawCopyFileReference"
        	},
        "RequestFormat"     -> (Block[{params=Cases[{##},("Parameters"->x_):>x,Infinity], 
            url=DeleteCases[{##},"Parameters"->_,Infinity],
            method=Cases[{##},("Method"->x_):>x,Infinity],
            headers=Flatten[Cases[{##},("Headers"->x_):>x,Infinity]],
            bodydata="BodyData" /. Rest[{##}],
            headerargdata},
            
            (*special case: basically params(key->val,..) are added to headers, for more info see:
            https://www.dropbox.com/developers/documentation/http/documentation#files-upload*)
            headerargdata = Lookup[headers,"Dropbox-API-Arg",False];
            If[headerargdata==={},
            	headerargdata = formatheaderpath[DeleteCases[Flatten@params,"access_token"->_]];
            	headers = headers/.{("Dropbox-API-Arg"->{})->("Dropbox-API-Arg"->headerargdata)};
            	(*Remove params from url*)
            	url[[1]] = StringReplace[url[[1]],"?"~~___->""];
            	(*use modified header*)
            	url = url/.{("Headers"->x_)->("Headers"->headers)};
            ];
            
            If[method==={"GET"},
                URLFetch@@({Sequence@@url, "Parameters"->Flatten@params}),
                (*else*)
                (*insert access-token to header's Authorization param*)
                If[headers[[1]] === ("Authorization"->"Bearer"),
                	url = url/.{
                		("Authorization"->"Bearer")->("Authorization"->("Bearer "<>(Association@Flatten@params)["access_token"]))
                		};
                ];
                (*This check handles: mostly body params are string, but in some cases dropbox accepts in boolean format*)
                If[(StringQ[bodydata]) && (StringContainsQ[bodydata,"="]),
                	bodydata = URLQueryDecode[bodydata];
                	(*Convert boolean strings*)
                	bodydata = bodydata/.{"true"->True,"false"->False};
                	bodydata = ExportString[Replace[bodydata, HoldPattern[Rule][a_, b_] :> Rule[tolowerfirstchar[a], b], Infinity],"JSON"];
                	url = url/.{("BodyData"->x_)->("BodyData"->bodydata)};
            	];
                URLFetch@@{Sequence@@url}
            ]
        ]&),
        "LogoutURL"         -> "https://www.dropbox.com/logout",
        "Information"       -> "Connect the Wolfram Language with your dropbox account"
    },
    {
        "OAuthVersion"      -> "2.0",
        "ServiceName"       -> "Dropbox", 
        "AuthorizeEndpoint" -> "https://www.dropbox.com/oauth2/authorize", 
        "AccessEndpoint"    -> "https://api.dropbox.com/oauth2/token", 
        "RedirectURI"       -> "https://www.wolfram.com/oauthlanding?service=Dropbox",
        "ClientInfo"        -> {"Wolfram","Token"},
        "AuthenticationDialog" :> (OAuthClient`tokenOAuthDialog[#, "Dropbox",dbicon]&),
        "Gets"              -> {},
        "Posts"             -> {"UserData","QuotaInfo","DataUpload","GraphicsUpload","StartUploadSession","AppendToUploadSession","FinishUploadSession",
        	"DirectoryTreePlot","FileSearch","FileData","FileNames","FileContents","DirectoryData","ImportFile"},
        "RawGets"           -> {"RawFileRevisions",
            "RawFileSearch","RawCopyFileReference",  "RawThumbnail"},
        "RawPosts"          -> {"RawUserData","RawQuotaInfo","RawFileDownload","RawPathData","RawListFolder","RawListFolderContinue",
        	"RawFileUpload","RawFileRestore","RawFilePreviewLink","RawFileLink","RawUploadSessionStart","RawUploadSessionAppend",
        	"RawUploadSessionFinish","RawFileCopy","RawCreateFolder","RawFileDelete","RawFileMove"},
        "RequestFormat"     -> (Block[{params=Cases[{##},("Parameters"->x_):>x,Infinity], 
            url=DeleteCases[{##},"Parameters"->_,Infinity],
            method=Cases[{##},("Method"->x_):>x,Infinity],
            headers=Flatten[Cases[{##},("Headers"->x_):>x,Infinity]],
            bodydata="BodyData" /. Rest[{##}],
            headerargdata},
            
            (*special case: basically params(key->val,..) are added to headers, for more info see:
            https://www.dropbox.com/developers/documentation/http/documentation#files-upload*)
            headerargdata = Lookup[headers,"Dropbox-API-Arg",False];
            If[headerargdata==={},
            	headerargdata = formatheaderpath[DeleteCases[Flatten@params,"access_token"->_]];
            	headers = headers/.{("Dropbox-API-Arg"->{})->("Dropbox-API-Arg"->headerargdata)};
            	(*Remove params from url*)
            	url[[1]] = StringReplace[url[[1]],"?"~~___->""];
            	(*use modified header*)
            	url = url/.{("Headers"->x_)->("Headers"->headers)};
            ];
            
            If[method==={"GET"},
                URLFetch@@({Sequence@@url, "Parameters"->Flatten@params}),
                (*else*)
                (*insert access-token to header's Authorization param*)
                If[headers[[1]] === ("Authorization"->"Bearer"),
                	url = url/.{
                		("Authorization"->"Bearer")->("Authorization"->("Bearer "<>(Association@Flatten@params)["access_token"]))
                		};
                ];
                (*This check handles: mostly body params are string, but in some cases dropbox accepts in boolean format*)
                If[(StringQ[bodydata]) && (StringContainsQ[bodydata,"="]),
                	bodydata = URLQueryDecode[bodydata];
                	(*Convert boolean strings*)
                	bodydata = bodydata/.{"true"->True,"false"->False};
                	bodydata = ExportString[Replace[bodydata, HoldPattern[Rule][a_, b_] :> Rule[tolowerfirstchar[a], b], Infinity],"JSON"];
                	url = url/.{("BodyData"->x_)->("BodyData"->bodydata)};
            	];
                URLFetch@@{Sequence@@url}
            ]
        ]&),
        "LogoutURL"         -> "https://www.dropbox.com/logout",
        "Information"       -> "Connect the Wolfram Language with your dropbox account"
}
];

(* a function for importing the raw data - usually json or xml - from the service *)
dropboximport[$Failed]:=Throw[$Failed]
dropboximport[raw_String]:=If[StringFreeQ[raw,"error"],raw,Message[ServiceExecute::apierr,raw]
]
dropboximport[raw_]:=raw


dropboximportjson[$Failed]:=Throw[$Failed]
dropboximportjson[json_, forcelistQ_:False]:=With[{res=ImportString[json,"JSON"]},
	If[FreeQ[res,_["errors",_]],
		If[forcelistQ,Association/@res,
			Switch[res,
				_Rule|{_Rule...},Association@res,
				{{_Rule...}...},Association/@res,
				_,res
			]
		],
		Message[ServiceExecute::apierr,("errors"/.res)];
		Throw[$Failed]
	]
]
 
(*** Raw ***)

dropboxdata["RawUserData"] = {
        "URL"				-> "https://api.dropboxapi.com/2/users/get_current_account",
        "Headers" 			-> {"Authorization"->"Bearer","Content-Type" -> "application/json"},
        "HTTPSMethod"		-> "POST",
        "BodyData" 			-> {"ParameterlessBodyData"},
        "ResultsFunction"	-> dropboximportjson
    }
    
dropboxdata["RawQuotaInfo"] = {
        "URL"				-> "https://api.dropboxapi.com/2/users/get_space_usage",
        "Headers" 			-> {"Authorization"->"Bearer","Content-Type" -> "application/json"},
        "HTTPSMethod"		-> "POST",
        "BodyData" 			-> {"ParameterlessBodyData"},
        "ResultsFunction"	-> dropboximportjson
    } 


dropboxdata["RawFileDownload"] = {
        "URL"				-> "https://content.dropboxapi.com/2/files/download",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "",
        						"Dropbox-API-Arg"->{}},
        "PathParameters"	-> {},
        "Parameters"		-> {"Path"},
        "RequiredParameters"-> {"Path"}, 
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> dropboximport
    }
       
    
dropboxdata["RawPathData"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/get_metadata",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "Parameters"		-> {"include_media_info","include_deleted","include_has_explicit_shared_members"},
        "RequiredParameters"-> {"Path"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData" 			-> {"Path"},
        "ResultsFunction"	-> dropboximportjson
    }
    
dropboxdata["RawFileLink"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/get_temporary_link",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "RequiredParameters"-> {"Path"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData" 			-> {"Path"},
        "ResultsFunction"	-> dropboximportjson
    }
    
dropboxdata["RawListFolder"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/list_folder",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "Parameters"		-> {"Recursive","include_media_info","include_deleted","include_has_explicit_shared_members"},
        "RequiredParameters"-> {"Path"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData" 			-> {"Path","Recursive"},
        "ResultsFunction"	-> dropboximportjson
    }
    
dropboxdata["RawListFolderContinue"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/list_folder/continue",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "PathParameters"	-> {"cursor"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"cursor"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData" 			-> {"cursor"},
        "ResultsFunction"	-> dropboximportjson
    }
      
dropboxdata["RawFileRevisions"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/list_revisions",
         "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "PathParameters"	-> {},
        "Parameters"		-> {"Limit"},
        "RequiredParameters"-> {"Path"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData" 			-> {"Path","Limit"},
        "ResultsFunction"	-> dropboximportjson
    }  
    
dropboxdata["RawFileSearch"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/search",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"Path","Query"},
        "HTTPSMethod"		-> "POST",
        "BodyData" 			-> {"Path","Query","start","max_results","mode"},
        "ResultsFunction"	-> dropboximportjson
    }
    
dropboxdata["RawFileUpload"] = {
        "URL"				-> "https://content.dropboxapi.com/2/files/upload",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/octet-stream",
        						"Dropbox-API-Arg"->{}},
       	"PathParameters"	-> {},
        "Parameters"		-> {"Path","Mode","Autorename"},
        "RequiredParameters"-> {"Path"}, 
        "BodyData"			-> {"ParameterlessBodyData"},
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> dropboximportjson
    }

(*TODO commenting new requests that are not yet documented, uncomment it later once it is decided to support in future versions*)
(*    
dropboxdata["RawUploadSessionStart"] = {
        "URL"				-> "https://content.dropboxapi.com/2/files/upload_session/start",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/octet-stream",
        						"Dropbox-API-Arg"->{}},
       	"PathParameters"	-> {},
        "Parameters"		-> {"close"},
        "RequiredParameters"-> {}, 
        "BodyData"			-> {"ParameterlessBodyData"},
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> dropboximportjson
    }

dropboxdata["RawUploadSessionAppend"] = {
        "URL"				-> "https://content.dropboxapi.com/2/files/upload_session/append_v2",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/octet-stream",
        						"Dropbox-API-Arg"->{}},
       	"PathParameters"	-> {},
        "Parameters"		-> {"cursor"},
        "RequiredParameters"-> {"cursor"}, 
        "BodyData"			-> {"ParameterlessBodyData"},
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> dropboximportjson
    }
    
dropboxdata["RawUploadSessionFinish"] = {
        "URL"				-> "https://content.dropboxapi.com/2/files/upload_session/finish",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/octet-stream",
        						"Dropbox-API-Arg"->{}},
       	"PathParameters"	-> {},
        "Parameters"		-> {"cursor","commit"},
        "RequiredParameters"-> {"cursor","commit"}, 
        "BodyData"			-> {},
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> dropboximportjson
    }
            
dropboxdata["RawFileRestore"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/restore",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "PathParameters"	-> {},
        "Parameters"		-> {},
        "RequiredParameters"-> {"Path","Rev"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"Path","Rev"},
        "ResultsFunction"	-> dropboximportjson
    }   
            
dropboxdata["RawFilePreviewLink"] = {
        "URL"				-> (ToString@StringForm["https://api.dropbox.com/1/shares/`1`", formatrootpath[##]]&),
        "PathParameters"	-> {"Root","Path"},
        "Parameters"		-> {"short_url","locale"},
        "RequiredParameters"-> {"Root","Path"}, 
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> dropboximportjson
    }       
    
dropboxdata["RawCopyFileReference"] = {
        "URL"				-> (ToString@StringForm["https://api.dropbox.com/1/search/`1`", formatrootpath[##]]&),
        "PathParameters"	-> {"Root","Path"},
        "RequiredParameters"-> {"Root","Path"}, 
        "HTTPSMethod"		-> "GET",
        "ResultsFunction"	-> dropboximportjson
    } 
    
dropboxdata["RawThumbnail"] = {
        "URL"				-> "https://content.dropboxapi.com/2/files/get_thumbnail",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "",
        						"Dropbox-API-Arg"->{}},
        "PathParameters"	-> {},
        "Parameters"		-> {"Path","format","size"},
        "RequiredParameters"-> {"Path"}, 
        "HTTPSMethod"		-> "POST",
        "ResultsFunction"	-> dropboximportjson
    }   

(** File Operations **)
dropboxdata["RawCreateFolder"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/create_folder",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"Path"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"Path"},
        "ResultsFunction"	-> dropboximportjson
    } 
    
dropboxdata["RawDeleteContents"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/permanently_delete",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"Path"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"Path"},
        "ResultsFunction"	-> dropboximportjson
    } 
     
dropboxdata["RawMoveContents"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/move",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"from_path","to_path"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"from_path","to_path"},
        "ResultsFunction"	-> dropboximportjson
    }
    
dropboxdata["RawCopyContents"] = {
        "URL"				-> "https://api.dropboxapi.com/2/files/copy",
        "Headers" 			-> {"Authorization"->"Bearer",
        						"Content-Type" -> "application/json"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"from_path","to_path"}, 
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"from_path","to_path"},
        "ResultsFunction"	-> dropboximportjson
    }
*)    
    
dropboxdata["icon"]=dbicon
    
dropboxdata[___]:=$Failed
(****** Cooked Properties ******)

dropboxcookeddata["FileContents",id_,args_]:=Block[
	{params,rawdata,data},
	params=filterparameters[args,getallparameters["RawFileDownload"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path"};
	If[isValidPath["Path"/.params]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawFileDownload",params];
	data=dropboximport[rawdata];
	data/;data=!=$Failed
]

dropboxcookeddata["ImportFile",id_,args_]:=Block[
	{params,rawdata,data, ext,res,url,urllocal},
	params=filterparameters[args,getallparameters["RawFileLink"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path"};
	If[isValidPath["Path"/.params]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawFileLink",params];
	data=dropboximportjson[rawdata];
	(*check for valid response*)
	If[(res=Lookup[data,"error_summary",False])=!=False,Message[ServiceExecute::npath,res];Throw[$Failed]];
	(url=Lookup[data,"link",$Failed];
		(	
			urllocal = URLDownload[url,Automatic]//First;
			res = If[StringContainsQ["Path"/.args,__~~".jpg"|".png"],
					Import[urllocal,ImageSize->Large],
					Import[urllocal]
				];
			res/;res=!=$Failed
		)/;url=!=$Failed
	)/;data=!=$Failed
]

dropboxcookeddata["GetThumbnail",id_,args_]:=Block[
	{params,rawdata,data, ext,res,url},
	params=filterparameters[args/.{("Format"->x_)->("Format"->{".tag"->ToLowerCase[x]})},getallparameters["RawThumbnail"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path"};
	If[isValidPath["Path"/.params]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawThumbnail",params];
	If[StringContainsQ[rawdata,"error_summary"],
		data = dropboximportjson[rawdata];
		Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
			Rule[".Tag", x_]->Rule["Tag", x]
		}],
		(*else*)
		ImportString[rawdata]
	]
]

dropboxcookeddata["FileRevisions",id_,args_]:=Block[
	{params,rawdata,data, ext,res,url},
	params=filterparameters[args,getallparameters["RawFileRevisions"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path"};
	If[isValidPath["Path"/.params]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawFileRevisions",params];
	data=dropboximportjson[rawdata];
	data = Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
			Rule[".Tag", x_]->Rule["Tag", x],
			fval["ClientModified"->(readDate[#]&)],
			fval["ServerModified"->(readDate[#]&)]
	}];
	If[MemberQ[Keys[data],"Error"],
		data,
		(*else no error*)
		data = Normal[data] /. HoldPattern[Rule["Entries", val_]] :> Rule["Entries", Apply[Association, val, {1}]];
		data//Association//Dataset
	]
]

dropboxcookeddata["DeleteContents",id_,args_]:=Block[
	{params,rawdata,data, ext,res,url},
	params=filterparameters[args,getallparameters["RawDeleteContents"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path"};
	If[isValidPath["Path"/.params]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawDeleteContents",params];
	rawdata
	(*Currently wolfram app is not allowed to use this api
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
		Rule[".Tag", x_]->Rule["Tag", x]
	}]
	*)
]

dropboxcookeddata["MoveContents",id_,args_]:=Block[
	{params,rawdata,data, ext,res,url},
	params=filterparameters[args/.{("FromPath"->p_)->("from_path"->p),("ToPath"->p_)->("to_path"->p)}
		,getallparameters["RawMoveContents"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"from_path","to_path"};
	If[isValidPath["from_path"/.params]===False,Message[ServiceExecute::npath,"Invalid FromPath format"];Throw[$Failed]];
	If[isValidPath["to_path"/.params]===False,Message[ServiceExecute::npath,"Invalid ToPath format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawMoveContents",params];
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
		Rule[".Tag", x_]->Rule["Tag", x],
		fval["ClientModified"->(readDate[#]&)],
		fval["ServerModified"->(readDate[#]&)]
	}]
]

dropboxcookeddata["CopyContents",id_,args_]:=Block[
	{params,rawdata,data, ext,res,url},
	params=filterparameters[args/.{("FromPath"->p_)->("from_path"->p),("ToPath"->p_)->("to_path"->p)}
		,getallparameters["RawCopyContents"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"from_path","to_path"};
	If[isValidPath["from_path"/.params]===False,Message[ServiceExecute::npath,"Invalid FromPath format"];Throw[$Failed]];
	If[isValidPath["to_path"/.params]===False,Message[ServiceExecute::npath,"Invalid ToPath format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawCopyContents",params];
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
		Rule[".Tag", x_]->Rule["Tag", x],
		fval["ClientModified"->(readDate[#]&)],
		fval["ServerModified"->(readDate[#]&)]
	}]
]

dropboxcookeddata["FileRestore",id_,args_]:=Block[
	{params,rawdata,data, ext,res,url},
	params=filterparameters[args,getallparameters["RawFileRestore"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path","Rev"};
	If[isValidPath["Path"/.params]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawFileRestore",params];
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
		Rule[".Tag", x_]->Rule["Tag", x],
		fval["ClientModified"->(readDate[#]&)],
		fval["ServerModified"->(readDate[#]&)]
	}]
]

dropboxcookeddata["CreateFolder",id_,args_]:=Block[
	{params,rawdata,data, ext,res,url},
	params=filterparameters[args,getallparameters["RawCreateFolder"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path"};
	If[isValidPath["Path"/.params]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	rawdata=OAuthClient`rawoauthdata[id,"RawCreateFolder",params];
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]]
]

dropboxcookeddata[prop:("UserData"),id_,args_]:=Block[
	{params,rawdata,data},
	params=filterparameters[Join[args,{"ParameterlessBodyData"->"null"}],getallparameters["RawUserData"]];
	rawdata=OAuthClient`rawoauthdata[id,"RawUserData",params];
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
		Rule[".Tag", x_]->Rule["Tag", x]
	}]
]

dropboxcookeddata[prop:("QuotaInfo"),id_,args_]:=Block[
	{params,rawdata,data},
	params=filterparameters[Join[args,{"ParameterlessBodyData"->"null"}],getallparameters["RawQuotaInfo"]];
	rawdata=OAuthClient`rawoauthdata[id,"RawQuotaInfo",params];
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
		Rule[".Tag", x_]->Rule["Tag", x]
	}]
]


dropboxcookeddata[prop:("FileData"|"DirectoryData"),id_,args_]:=Block[
	{params,rawdata,data,fulldata,cursor,error,errormessage},
	params=filterparameters[args,getallparameters["RawPathData"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path"};
	If[isValidPath["Path"/.params]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	If[("Path"/.params)==="/",
			(*root folder is a special case*)
			data = Association[{".tag"->"folder"}];
			error = False,
			(*else not root*)
			rawdata=OAuthClient`rawoauthdata[id,"RawPathData",params];
			data=dropboximportjson[rawdata];
			error = Lookup[data,"error",False];	
	];
	If[error===False,
		If[Lookup[data,".tag"] === "folder",
			If[prop==="FileData",Message[ServiceExecute::nfile,"Path"/.params]];
			params=filterparameters[args/.("Path"->"/")->("Path"->""),getallparameters["RawListFolder"]]/.HoldPattern[Rule["Recursive",tf_]]:>Rule["Recursive",ToLowerCase[ToString[tf]]];
			params = addDefaultRecursiveOption[params];
			rawdata=OAuthClient`rawoauthdata[id,"RawListFolder",params];
			data=dropboximportjson[rawdata];
			fulldata = data["entries"];
			(*pagination implementation*)
			While[data["has_more"]===True,
				cursor = data["cursor"];
				params=filterparameters[Join[args,{"cursor"->cursor}],getallparameters["RawListFolderContinue"]];
				rawdata=OAuthClient`rawoauthdata[id,"RawListFolderContinue",params];
				data=dropboximportjson[rawdata];
				fulldata = Join[fulldata,data["entries"]];
			];
			Replace[Normal[fulldata],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
				Rule[".Tag", x_]->Rule["Tag", x],
				fval["ClientModified"->(readDate[#]&)],
				fval["ServerModified"->(readDate[#]&)]
				}
			,
			(*FileData*)
			If[prop==="DirectoryData",Message[ServiceExecute::ndir,"Path"/.params]];
			Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
				Rule[".Tag", x_]->Rule["Tag", x],
				fval["ClientModified"->(readDate[#]&)],
				fval["ServerModified"->(readDate[#]&)]
				}]
		],
		(*else*)
		(*error message is returned by dropbox service*)
		errormessage = Lookup[data,"error_summary","invalid path"];
		Message[ServiceExecute::svcerror,errormessage];
		$Failed
	]
]

dropboxcookeddata["FileNames",id_,args_]:=Block[
	{data,filenames,foldernames,foldersretmetadata,res},
	data=dropboxcookeddata["DirectoryData",id,args];
	(*To ensure consistency in the folder names returned by dropbox, we will use lower case folder and file names*)
	(filenames = "PathLower" /. Select[data, (("Tag" /. #) === "file") &];
		foldernames =  "PathLower" /. Select[data, (("Tag" /. #) === "folder") &];
		If[filenames==="PathLower",
			(*no file is present under given path*)
			filenames = {}
		];
		If[foldernames==="PathLower",
			(*no folder is present under given path*)
			foldernames = {}
		];
		(*when recursive is False, db api does not send metadata response for root folder*)
		If[StringMatchQ[Lookup[args,"Path"],"/"] || (Lookup[args,"Recursive",False] =!= True),AppendTo[foldernames,ToLowerCase["Path"/.args]]];
	
		res = Rule[#, {}] & /@ foldernames // Association;
		(*insert file names corresponds to each folder in above association*)
		AppendTo[res[FileNameDrop[#, -1,OperatingSystem -> "Unix"]], #] & /@ filenames;
		res//Dataset
	)/;(data=!=$Failed && ("Tag"/.data) =!= "file")
]

dropboxcookeddata["DirectoryTreePlot",id_,args_]:=Block[
	{params,rawdata,OAuthClient`$CacheResults=True,data,folders,treeplotdata={},root},
	data = dropboxcookeddata["FileNames",id,args]//Normal;
	(folders = Keys[data]//Normal;
		root = ToLowerCase["Path"/.args];
		(*fill treeplotdata list with folder->leaves/subfolders format*)
		Scan[Map[Function[arg, AppendTo[treeplotdata, # -> arg]], data[#]] &, folders];
		AppendTo[treeplotdata, FileNameDrop[#, -1, OperatingSystem -> "Unix"] -> #] & /@ DeleteCases[folders,root];
		If[treeplotdata=!={},
			TreePlot[treeplotdata, PlotStyle -> Directive[Blue, PointSize[Medium],Dashed],DirectedEdges -> True],
			(*else, empty directory*)
			Print["EmptyDirectory"]	
		]
	)/;data=!=$Failed
]

dropboxcookeddata["DataUpload",id_,args_]:=Block[
	{params,rawdata,data},
	If[FreeQ[args,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Data","Path"};
	If[isValidPath["Path"/.args]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	params=filterparameters[args/.{
		"Data"->"ParameterlessBodyData",("Mode"->x_)->("Mode"->ToLowerCase[x])},getallparameters["RawFileUpload"]];
	params=params/.{HoldPattern[Rule]["ParameterlessBodyData",d:Except[_String]]:>Rule["ParameterlessBodyData",ToString[InputForm[d]]],
		HoldPattern[Rule]["Autorename", tf_] :> Rule["Autorename", ToLowerCase[ToString[tf]]]};
	rawdata=OAuthClient`rawoauthdata[id,"RawFileUpload",params];
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
			fval["ClientModified"->(readDate[#]&)],
			fval["ServerModified"->(readDate[#]&)]
	}]
]

dropboxcookeddata["GraphicsUpload",id_,args_]:=Block[
	{params,rawdata,data, ext,res},
	If[FreeQ[args,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Graphics","Path"};
	If[isValidPath["Path"/.args]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	params=filterparameters[args/.{
		"Graphics"->"ParameterlessBodyData",("Mode"->x_)->("Mode"->ToLowerCase[x])},getallparameters["RawFileUpload"]];
	ext=FileExtension["Path"/.params];
	data="ParameterlessBodyData"/.params;
	data=Check[ImportString[ExportString[data, ext], "Byte"],
		Message[ServiceExecute::grext, ext];Throw[$Failed]
	];
	params=params/.{HoldPattern[Rule]["ParameterlessBodyData",g_]:>Rule["ParameterlessBodyData",data],
		HoldPattern[Rule]["Autorename", tf_] :> Rule["Autorename", ToLowerCase[ToString[tf]]]};
	rawdata=OAuthClient`rawoauthdata[id,"RawFileUpload",params];
	res=dropboximportjson[rawdata];
	Association[Replace[Normal[res],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
			fval["ClientModified"->(readDate[#]&)],
			fval["ServerModified"->(readDate[#]&)]
	}]
]

dropboxcookeddata["StartUploadSession",id_,args_]:=Block[
	{params,rawdata,data},
	If[FreeQ[args,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Data"};
	params=filterparameters[args/.{
		"Data"->"ParameterlessBodyData"},getallparameters["RawUploadSessionStart"]];
	params=params/.HoldPattern[Rule]["ParameterlessBodyData",d:Except[_String]]:>Rule["ParameterlessBodyData",ToString[InputForm[d]]];
	rawdata=OAuthClient`rawoauthdata[id,"RawUploadSessionStart",params];
	data=dropboximportjson[rawdata];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]]
]

dropboxcookeddata["AppendToUploadSession",id_,args_]:=Block[
	{params,rawdata,data,error,offsetvalue},
	If[FreeQ[args,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Data","SessionID"};
	params=filterparameters[args/.{
		"Data"->"ParameterlessBodyData",("SessionID"->sid_)->("cursor"->{"Session_ID"->sid,"offset"->0})},
		getallparameters["RawUploadSessionAppend"]];
	params=params/.HoldPattern[Rule]["ParameterlessBodyData",d:Except[_String]]:>Rule["ParameterlessBodyData",ToString[InputForm[d]]];
	rawdata=OAuthClient`rawoauthdata[id,"RawUploadSessionAppend",params];
	data=dropboximportjson[rawdata];
	error = Lookup[data,"error_summary",False];
	If[error =!= False && StringContainsQ[error,"incorrect_offset"],
		(*resend the request with correct offset*)
		offsetvalue = Cases[data,("correct_offset"->x_)->x,Infinity][[1]];
		params = params/.{("offset"->x_)->("offset"->offsetvalue)};
		rawdata=OAuthClient`rawoauthdata[id,"RawUploadSessionAppend",params];
		data=dropboximportjson[rawdata];
	];
	data
]


dropboxcookeddata["FinishUploadSession",id_,args_]:=Block[
	{params,rawdata,data,error,offsetvalue},
	If[FreeQ[args,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"SessionID","Path"};
	If[isValidPath["Path"/.args]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	params=filterparameters[args/.{
		("SessionID"->sid_)->("cursor"->{"Session_ID"->sid,"offset"->0}),
		("Path"->p_)->("commit"->{"Path"->p,"mode"->{".tag"->"overwrite"}})},
		getallparameters["RawUploadSessionFinish"]];
	rawdata=OAuthClient`rawoauthdata[id,"RawUploadSessionFinish",params];
	data=dropboximportjson[rawdata];
	error = Lookup[data,"error_summary",False];
	If[error =!= False && StringContainsQ[error,"incorrect_offset"],
		(*resend the request with correct offset*)
		offsetvalue = Cases[data,("correct_offset"->x_)->x,Infinity][[1]];
		params = params/.{("offset"->x_)->("offset"->offsetvalue)};
		rawdata=OAuthClient`rawoauthdata[id,"RawUploadSessionFinish",params];
		data=dropboximportjson[rawdata];
	];
	Association[Replace[Normal[data],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
			fval["ClientModified"->(readDate[#]&)],
			fval["ServerModified"->(readDate[#]&)]
	}]
]

dropboxcookeddata["FileSearch",id_,args_]:=Block[
	{params,rawdata, root,data},
	params=filterparameters[args,getallparameters["RawFileSearch"]];
	If[FreeQ[params,#],Message[ServiceExecute::nparam,#1];Throw[$Failed]]&/@{"Path","Query"};
	If[isValidPath["Path"/.args]===False,Message[ServiceExecute::npath,"Invalid path format"];Throw[$Failed]];
	(*special rule for root path*)
	params = params/.("Path"->"/")->("Path"->"");
	rawdata=OAuthClient`rawoauthdata[id,"RawFileSearch",params];
	data=dropboximportjson[rawdata];
	Dataset[Replace[data["matches"],HoldPattern[Rule][a_,b_]:>Rule[camelcase[a],b],Infinity]/.{
		Rule[".Tag", x_]->Rule["Tag", x]
	}]
]
    
dropboxcookeddata[___]:=$Failed 
(* Send Message *)

dropboxsendmessage[___]:=$Failed

(*** Service specific utilites ****)
filterparameters[given:{(_Rule|_RuleDelayed)...},accepted_,separators_:{"_"}]:=Module[{camel=camelcase[accepted,separators]},
	Cases[given,HoldPattern[Rule|RuleDelayed][Alternatives@@Join[accepted, camel],_],Infinity]/.Thread[camel->accepted]
]
filterparameters[___]:=Throw[$Failed]

camelcase[l_List, rest___]:=camelcase[#,rest]&/@l
camelcase[str_String, separators_:{"_"}]:=StringReplace[
 StringReplace[
  StringReplace[str, 
   Thread[separators -> " "]], {WordBoundary ~~ word_ :> 
    ToUpperCase[word]}], {"Id"~~WordBoundary->"ID",WhitespaceCharacter -> "","Url"~~WordBoundary->"URL","Urls"~~WordBoundary->"URLs"}]

fval[_[label_,fun_]]:=(Rule[label,value_]:>Rule[label,fun[value]])

readDate[date_, form_: DateObject] := 
 form@DateList[{date, {"Year", "-", "Month", "-", "Day", 
   "T", "Hour", ":", "Minute", ":", "Second", "Z"}}]
    
getallparameters[str_]:=DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.dropboxdata[str]],
	("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]

formatrootpath[root_,path_]:=StringJoin[stripslash[root],"/",stripslash[path]]

formatheaderpath[pvpairs_] :=Module[{pparams},
	pparams = pvpairs/.{"true"->True,"false"->False};
	pparams = Replace[Normal[pparams],HoldPattern[Rule][a_,b_]:>Rule[ToLowerCase[a],b],Infinity];
	If[Length[pparams]===0,"{}",ExportString[pparams,"JSON"]]
]
formatbodypath[pathvalue_] := ExportString[{"path"->pathvalue},"JSON"]
stripslash[""|"/"]="";
stripslash[str_]:=If[StringTake[#,1]==="/",StringDrop[#,1],#]&@If[StringTake[str,-1]==="/",StringDrop[str,-1],str]
tolowerfirstchar[str_String]:=StringReplacePart[str, ToLowerCase[StringTake[str, 1]], {1, 1}]
isValidPath[path_String]:=StringMatchQ[path, RegularExpression["(/(.|[\r\n])*|id:.*)|(rev:[0-9a-f]{9,})|(ns:[0-9]+(/.*)?)"]]
addDefaultRecursiveOption[p_List]:=If[!MemberQ[Keys[Association[p]],"Recursive"],Join[p,{"Recursive"->"false"}],p]

dbicon=Image[RawArray["Byte", {{{38, 38, 38, 217}, {99, 99, 99, 156}, {97, 97, 97, 158}, {99, 99, 99, 156}, {99, 99, 99, 
  156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 
  156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 
  156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 
  156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 156}, {99, 99, 99, 
  156}, {98, 98, 98, 157}, {99, 99, 99, 156}, {22, 22, 22, 233}}, {{108, 108, 108, 147}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {63, 63, 63, 192}}, {{97, 97, 97, 158}, {251, 251, 251, 4}, {247, 247, 247, 8}, {251, 251, 251, 4}, {251, 251, 251, 
  4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 
  251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 
  251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, 
  {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 4}, {251, 251, 251, 
  4}, {251, 251, 251, 4}, {248, 248, 248, 7}, {251, 251, 251, 4}, {57, 57, 57, 198}}, {{99, 99, 99, 156}, {255, 255, 
  255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 
  255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 
  156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {253, 254, 255, 1}, {253, 254, 255, 3}, 
  {253, 254, 255, 2}, {254, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {253, 254, 255, 1}, {255, 255, 255, 2}, {253, 254, 255, 3}, {254, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 
  252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {252, 253, 255, 3}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 1}, {254, 254, 
  255, 1}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 254, 255, 2}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 1}, {252, 254, 255, 2}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 
  197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {253, 254, 255, 1}, {253, 254, 255, 3}, {255, 255, 255, 0}, {218, 237, 251, 17}, 
  {71, 159, 236, 157}, {104, 178, 239, 126}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 254, 255, 2}, {251, 253, 
  255, 4}, {255, 255, 255, 0}, {222, 239, 252, 17}, {53, 149, 234, 171}, {121, 186, 241, 106}, {250, 253, 255, 0}, 
  {255, 255, 255, 0}, {251, 253, 255, 4}, {254, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 
  255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {253, 254, 255, 2}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {148, 200, 244, 81}, {14, 130, 230, 221}, {0, 118, 227, 255}, {0, 117, 227, 255}, 
  {72, 161, 236, 158}, {239, 247, 253, 5}, {255, 255, 255, 0}, {255, 255, 255, 0}, {191, 223, 248, 42}, {22, 133, 
  230, 213}, {0, 119, 227, 255}, {0, 118, 227, 255}, {52, 150, 234, 179}, {199, 227, 249, 37}, {255, 255, 255, 0}, 
  {255, 255, 255, 1}, {254, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 
  3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {254, 254, 255, 1}, {255, 255, 255, 0}, {219, 237, 252, 21}, {71, 160, 236, 160}, {0, 
  117, 227, 255}, {0, 125, 229, 255}, {3, 128, 229, 251}, {4, 128, 229, 251}, {0, 118, 227, 255}, {41, 145, 233, 
  193}, {202, 228, 250, 36}, {153, 203, 245, 80}, {7, 125, 229, 235}, {0, 123, 228, 255}, {3, 128, 229, 250}, {4, 
  128, 229, 252}, {0, 120, 228, 255}, {5, 125, 229, 240}, {123, 187, 241, 108}, {245, 250, 254, 0}, {255, 255, 255, 
  0}, {254, 254, 255, 1}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 
  58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {252, 254, 255, 2}, 
  {254, 255, 255, 1}, {163, 209, 246, 60}, {3, 122, 228, 241}, {0, 121, 228, 255}, {4, 128, 229, 252}, {1, 126, 229, 
  253}, {0, 126, 229, 254}, {4, 128, 229, 251}, {0, 124, 228, 255}, {0, 114, 227, 255}, {80, 166, 237, 126}, {37, 
  145, 233, 183}, {0, 114, 226, 255}, {1, 127, 229, 255}, {3, 128, 229, 252}, {0, 126, 229, 255}, {2, 127, 229, 252}, 
  {2, 127, 229, 254}, {0, 115, 227, 255}, {31, 139, 232, 193}, {224, 239, 252, 14}, {254, 254, 255, 1}, {254, 254, 
  255, 1}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 
  255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {254, 254, 255, 1}, {253, 254, 255, 2}, {220, 237, 251, 22}, 
  {75, 162, 236, 159}, {0, 117, 227, 255}, {2, 127, 229, 254}, {1, 127, 229, 252}, {0, 126, 229, 255}, {0, 117, 227, 
  255}, {33, 139, 231, 201}, {173, 213, 246, 60}, {255, 255, 255, 0}, {244, 249, 254, 2}, {123, 187, 241, 108}, {6, 
  126, 229, 236}, {0, 119, 227, 255}, {2, 128, 229, 253}, {2, 127, 229, 252}, {0, 123, 228, 255}, {1, 123, 228, 242}, 
  {128, 190, 242, 103}, {246, 250, 254, 5}, {253, 254, 255, 1}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 
  252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {254, 255, 255, 1}, {255, 255, 255, 0}, {254, 255, 255, 0}, {114, 182, 240, 110}, 
  {0, 119, 228, 251}, {0, 123, 228, 255}, {0, 123, 228, 242}, {114, 183, 241, 115}, {239, 247, 253, 4}, {255, 255, 
  255, 0}, {253, 254, 255, 2}, {255, 255, 255, 0}, {255, 255, 255, 0}, {204, 229, 250, 31}, {66, 156, 235, 166}, {0, 
  119, 227, 255}, {0, 122, 228, 255}, {16, 131, 230, 220}, {179, 217, 247, 51}, {255, 255, 255, 0}, {255, 255, 255, 
  1}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 
  58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {254, 254, 255, 1}, {250, 253, 255, 6}, {255, 255, 255, 0}, {154, 204, 245, 68}, {36, 137, 231, 
  183}, {214, 234, 251, 18}, {255, 255, 255, 0}, {254, 254, 255, 3}, {252, 253, 255, 3}, {255, 255, 255, 0}, {254, 
  255, 255, 0}, {250, 252, 255, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {151, 202, 244, 76}, {34, 137, 231, 184}, 
  {230, 243, 253, 6}, {255, 255, 255, 0}, {249, 252, 255, 5}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 
  255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 253, 255, 3}, {255, 
  255, 255, 0}, {229, 243, 253, 11}, {78, 164, 236, 151}, {15, 128, 230, 225}, {115, 182, 240, 115}, {233, 244, 253, 
  11}, {255, 255, 255, 0}, {253, 254, 255, 3}, {253, 254, 255, 1}, {252, 253, 255, 3}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {199, 226, 249, 40}, {72, 160, 236, 164}, {15, 128, 229, 226}, {139, 195, 243, 92}, {255, 255, 255, 0}, 
  {255, 255, 255, 1}, {253, 254, 255, 1}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 
  3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {253, 254, 255, 1}, {255, 255, 255, 0}, {201, 228, 250, 32}, {31, 139, 231, 200}, {0, 
  117, 227, 255}, {0, 126, 229, 255}, {0, 116, 227, 255}, {28, 138, 231, 209}, {158, 205, 245, 71}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {238, 247, 253, 5}, {108, 179, 240, 119}, {1, 123, 228, 241}, {0, 119, 
  227, 255}, {0, 125, 229, 255}, {0, 115, 227, 255}, {87, 168, 237, 140}, {243, 249, 254, 0}, {255, 255, 255, 0}, 
  {254, 254, 255, 1}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 
  197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {252, 254, 255, 2}, {255, 
  255, 255, 1}, {165, 209, 246, 56}, {0, 117, 227, 247}, {0, 123, 228, 255}, {5, 129, 230, 249}, {0, 126, 229, 255}, 
  {4, 128, 229, 251}, {0, 123, 228, 255}, {0, 118, 227, 255}, {71, 160, 236, 160}, {199, 227, 249, 40}, {162, 208, 
  245, 72}, {33, 139, 231, 202}, {0, 117, 227, 255}, {1, 126, 229, 255}, {2, 128, 229, 252}, {0, 126, 229, 254}, {7, 
  130, 230, 248}, {0, 115, 227, 255}, {33, 141, 232, 193}, {224, 240, 252, 13}, {253, 254, 255, 2}, {254, 254, 255, 
  1}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 
  255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {254, 254, 255, 1}, {254, 254, 255, 1}, {211, 233, 251, 26}, {85, 
  166, 237, 148}, {0, 121, 228, 253}, {0, 122, 228, 255}, {3, 128, 229, 251}, {0, 126, 229, 255}, {3, 128, 229, 251}, 
  {2, 126, 229, 255}, {0, 116, 227, 255}, {71, 162, 236, 125}, {37, 145, 233, 181}, {0, 115, 227, 255}, {5, 129, 229, 
  249}, {1, 126, 229, 254}, {1, 126, 229, 254}, {3, 128, 229, 253}, {0, 118, 227, 255}, {11, 128, 229, 226}, {128, 
  190, 242, 102}, {240, 248, 254, 7}, {253, 254, 255, 1}, {254, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, 
  {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {171, 212, 246, 64}, {29, 137, 
  231, 208}, {0, 118, 227, 255}, {1, 127, 229, 255}, {0, 123, 228, 255}, {11, 128, 229, 236}, {84, 169, 237, 119}, 
  {46, 147, 233, 185}, {65, 159, 236, 153}, {63, 158, 236, 152}, {0, 121, 228, 255}, {1, 126, 229, 255}, {0, 124, 
  228, 255}, {0, 117, 227, 255}, {73, 160, 236, 162}, {208, 231, 250, 24}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 
  197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {254, 254, 255, 1}, {252, 253, 255, 4}, {255, 255, 255, 0}, {233, 244, 253, 9}, {95, 172, 238, 122}, 
  {7, 125, 229, 242}, {23, 135, 231, 216}, {85, 170, 238, 116}, {28, 137, 231, 209}, {0, 120, 228, 255}, {0, 121, 
  228, 255}, {55, 152, 234, 169}, {75, 164, 236, 134}, {4, 125, 229, 248}, {27, 137, 231, 211}, {151, 201, 244, 77}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {251, 253, 255, 3}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 
  255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {250, 253, 255, 4}, {255, 255, 255, 0}, {90, 172, 238, 131}, {80, 165, 237, 129}, {89, 172, 238, 
  121}, {15, 129, 230, 232}, {0, 122, 228, 255}, {2, 127, 229, 252}, {2, 127, 229, 252}, {0, 119, 227, 255}, {37, 
  143, 232, 199}, {98, 176, 239, 104}, {63, 156, 235, 154}, {172, 213, 247, 80}, {254, 255, 255, 2}, {249, 252, 255, 
  5}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 
  252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {253, 254, 255, 2}, 
  {253, 254, 255, 1}, {130, 192, 242, 106}, {11, 127, 229, 224}, {0, 119, 227, 255}, {0, 125, 229, 255}, {3, 128, 
  229, 250}, {0, 126, 229, 255}, {0, 126, 229, 255}, {5, 129, 230, 249}, {0, 121, 228, 255}, {0, 120, 227, 252}, {33, 
  138, 231, 205}, {198, 226, 249, 52}, {254, 254, 255, 1}, {253, 254, 255, 2}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 
  58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {254, 254, 255, 1}, {255, 255, 255, 
  0}, {208, 231, 250, 27}, {74, 161, 236, 154}, {0, 119, 228, 254}, {0, 122, 228, 255}, {4, 129, 229, 251}, {3, 127, 
  229, 254}, {0, 118, 227, 255}, {11, 128, 229, 231}, {124, 187, 241, 112}, {245, 251, 254, 1}, {255, 255, 255, 0}, 
  {254, 255, 255, 1}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 
  255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 254, 255, 3}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {168, 210, 246, 61}, {37, 141, 232, 196}, {0, 116, 227, 255}, {0, 119, 227, 255}, {77, 163, 237, 155}, {212, 233, 
  251, 27}, {255, 255, 255, 0}, {255, 255, 255, 1}, {253, 254, 255, 2}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, 
  {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {253, 254, 255, 1}, {253, 254, 255, 3}, {255, 255, 255, 0}, {241, 248, 254, 0}, {124, 
  187, 241, 103}, {165, 209, 246, 69}, {255, 255, 255, 0}, {255, 255, 255, 0}, {251, 253, 255, 4}, {254, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 
  99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {252, 253, 255, 3}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {254, 254, 
  255, 3}, {253, 254, 255, 1}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 
  4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {254, 
  254, 255, 1}, {251, 253, 255, 4}, {252, 253, 255, 3}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 
  58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 255, 0}, {58, 58, 58, 197}}, {{99, 99, 99, 156}, {255, 255, 255, 
  0}, {251, 251, 251, 4}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {252, 252, 252, 3}, {255, 255, 
  255, 0}, {58, 58, 58, 197}}, {{98, 98, 98, 157}, {252, 252, 252, 3}, {248, 248, 248, 7}, {252, 252, 252, 3}, {252, 
  252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, 
  {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 
  3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 
  252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 252, 252, 3}, {252, 
  252, 252, 3}, {252, 252, 252, 3}, {249, 249, 249, 6}, {252, 252, 252, 3}, {57, 57, 57, 198}}, {{105, 105, 105, 
  150}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, 
  {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 
  0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 
  255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 255, 255, 0}, {255, 
  255, 255, 0}, {255, 255, 255, 0}, {62, 62, 62, 193}}, {{23, 23, 23, 232}, {58, 58, 58, 197}, {57, 57, 57, 198}, 
  {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, 
  {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, 
  {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, 
  {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, {58, 58, 58, 197}, 
  {58, 58, 58, 197}, {58, 58, 58, 197}, {57, 57, 57, 198}, {58, 58, 58, 197}, {13, 13, 13, 242}}}], "Byte", 
 ColorSpace -> "RGB", Interleaving -> True];

End[]
           		
End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{DropboxOAuth`Private`dropboxdata,DropboxOAuth`Private`dropboxcookeddata,DropboxOAuth`Private`dropboxsendmessage}
