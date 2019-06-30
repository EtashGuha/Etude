
(* Wolfram CURLLink Package *)

BeginPackage["CURLLink`HTTP`"]
System`CookieFunction
$WolframHTTPMultiHandle;
$WolframHTTPMultiHandleAsync;
$MaxConnections=32;
$MaxConnectionsPerHost=8
$PipeLining = 0;
$HTTPCredentialStorage = <||>
Needs["PacletManager`"]


Begin["`Private`"] (* Begin Private Context *)
Needs["CURLLink`"] 
CURLLink`Utilities`AddHandler[
	"http",
	<|
	"URLFetch"->httpFetch
	,
	"URLFetchAsynchronous"->httpFetchAsynchronous
	,
	"URLSave"->httpSave
	,
	"URLSaveAsynchronous"->httpSaveAsynchronous
	|>
	]
If[$VersionNumber <  9,
	Message[CURLLink::enable,  "CURLLink"]
]

Clear[$WolframHTTPMultiHandle];
Clear[$WolframHTTPMultiHandleAsync];

Needs["CURLLink`Cookies`"]
(****************************************************************************)

$MessageHead = HTTP;

(****************************************************************************)

$returnTypes={"Content","ContentData","DebugContentData","DebugContent","Cookies","Headers","StatusCode","Stream","HeadersReceived","DebugBodyByteArray","BodyByteArray"}
$storageForms={"Return","Proxies","FTP","URL","BaseURL","OPTIONS"}
$sslErrorCodes={35, 51, 53, 54, 58, 59, 60, 64, 66, 77, 80, 82, 83, 90, 91}

(*
Non fatal http status codes. 
For others http status codes URL* functions return $Failed
*)
$httpNonFailureReturnCodes=
	{
		100,101,
		200,201,202,203,204,205,206,
		300,301,302,303,304,305,307,
		400,401,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417
	}
	
$HTTPStorageForms={"Stream"}
$DeprecatedOptions = {
	"BodyData" -> "Body",
	"MultipartData" -> "MultipartElements"
}
Options[httpFetch] = Options[URLFetch]
$StandardOptions=Options[URLFetch]
Options[setStandardOptions] = $StandardOptions

deprecatedOptionFix[sym_, options___] := Sequence @@ Replace[
	{options}, 
	head_[key:Alternatives @@ Keys[$DeprecatedOptions], value_] :> (
		(* uncomment the next line to send a deprecation warning *)
		(* Message[sym::depropt, key, Lookup[$DeprecatedOptions, key]]; *)
		head[Lookup[$DeprecatedOptions, key], value]
	),
	{1}
]
(*Utility functions*)

isURL[url_,head_:URLFetch]:=StringQ[getURL[url,head]];
getURL[url_String,head_:URLFetch]:=url
getURL[URL[url_],head_:URLFetch]:=getURL[url,head]
getURL[IPAddress[url_],head_:URLFetch]:=getURL[url,head]
getURL[exp_,head_:URLFetch]:=(Message[head::invurl, exp];$Failed)

isFile[exp_,head_:URLSave]:=StringQ[getFile[exp,head]];
getFile[file_String,head_:URLSave]:=file
getFile[File[file_],head_:URLSave]:=getFile[file,head]
getFile[exp_,head_:URLSave]:=(Message[head::invfile, exp];$Failed)

unsetCURLHandle[handles_List]:=Map[unsetCURLHandle[#]&,handles]
unsetCURLHandle[handle_CURLLink`CURLHandle]:=Map[Quiet[Unset[handle[#]], Unset::norep]&,$storageForms]
unsetHTTPData[handle_CURLLink`CURLHandle]:=Map[Quiet[Unset[HTTPData[handle, #]], Unset::norep]&,$HTTPStorageForms]

deprecatedOptionQ[options___] := Cases[Keys @ {options}, Alternatives @@ Keys[$DeprecatedOptions]] =!= {}

httpFetch[urlExp_,opts:OptionsPattern[URLFetch]]/;initializeQ[]:=(httpFetch[urlExp,"Content",opts])	

httpFetch[urlExp_, res:(_String|_List|All), options___?OptionQ] /; deprecatedOptionQ[options] := 
	httpFetch[urlExp, res, deprecatedOptionFix[URLFetch, options]] 

allowCredentialDialog[opts:OptionsPattern[]] := (
	Lookup[Flatten[{opts}],"DisplayProxyDialog",True]
);


httpFetch[urlExp_/;isURL[urlExp,URLFetch], res:(_String|_List|All), opts:OptionsPattern[URLFetch]] /;(initializeQ[]) :=
	Module[{format,stream,httpConnectCode,responseCode,
		streamIsRequested,contentOrContentDataIsRequested,requestedElements=Flatten@{res},url,handle, output, error, stdOpts, elements, wellFormedURL,isblocking=True},
		
		url=getURL[urlExp,URLFetch];
		
		setMessageHead[URLFetch];
		
		$LastKnownCookies=CURLLink`Cookies`GetKnownCookies[];
		
		streamIsRequested=MemberQ[requestedElements,"Stream"];
		
		contentOrContentDataIsRequested=MemberQ[requestedElements,"Content"|"ContentData"];
		
		If[streamIsRequested,format="Stream";isblocking=False,format=None];

		If[streamIsRequested && contentOrContentDataIsRequested,Throw[$Failed,CURLLink`Utilities`Exception]];
		
		If[res===All,requestedElements=Complement[$URLFetchElements,{"Stream"}]];
				
		stdOpts = Flatten[{opts, FilterRules[Options[URLFetch], Except[{opts}]]}];
		
		error = Catch[
			handle = commonInit[<|"URL"->url,"Function"-> URLFetch,"Format"->format, "Options"->stdOpts|>];
			
			If[handle === $Failed,Return[$Failed]];
			(*If stream is not requested, wait*)
			If[Not[streamIsRequested],handle["Return"] = CURLLink`CURLWait[handle];];
			
			
			If[streamIsRequested,handle["Return"]=0;getResponseCodes[handle];stream=getResponseCodes[handle][[1]]];
			
			responseCode = CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
			
			httpConnectCode = CURLLink`CURLGetInfo[handle,"CURLINFO_HTTP_CONNECTCODE"];
			
			If[(responseCode === 407 ||httpConnectCode == 407)&& allowCredentialDialog[opts],
				If[Not[streamIsRequested],
					CURLLink`CURLMultiHandleRemove[$WolframHTTPMultiHandle,handle];
					CURLLink`CURLHandleUnload[handle];
				];
				handle=commonInit[<|"URL"->url,"Function"-> URLFetch,"Format"->format, "Options"->stdOpts,"Code"->407,"RequestAuthentication"->OptionValue["CredentialsProvider"]|>];
				If[Not[streamIsRequested],handle["Return"] = CURLLink`CURLWait[handle];];
				If[streamIsRequested,handle["Return"]=0;stream=getResponseCodes[handle][[1]]];
				responseCode = CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
				httpConnectCode = CURLLink`CURLGetInfo[handle,"CURLINFO_HTTP_CONNECTCODE"];
				If[responseCode === 407 ||httpConnectCode == 407,
					$proxyCache = False;
				  ];

			];
			
			If[(responseCode === 401 ||httpConnectCode == 401) && allowCredentialDialog[opts],
				If[Not[streamIsRequested],
					CURLLink`CURLMultiHandleRemove[$WolframHTTPMultiHandle,handle];
					CURLLink`CURLHandleUnload[handle];
				];
				handle=commonInit[<|"URL"->url,"Function"-> URLFetch,"Format"->format, "Code"->401,"Options"->stdOpts,"RequestAuthentication"->OptionValue["CredentialsProvider"]|>];
				If[Not[streamIsRequested],handle["Return"] = CURLLink`CURLWait[handle];];
				responseCode = CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
				httpConnectCode = CURLLink`CURLGetInfo[handle,"CURLINFO_HTTP_CONNECTCODE"];
				If[streamIsRequested,handle["Return"]=0;stream=getResponseCodes[handle][[1]]];
				
			];
			
			If[CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"] === 401, 
				wellFormedURL = If[StringMatchQ[url, {"http://*", "https://*", "ftp://*", "ftps://*"}], 
					URIJoin[Flatten@{URISplit[url]}]
				(*else*), 
					URIJoin[Flatten@{URISplit["http://" <> url]}]
				];				
				
				clearCredentials[url];
			];

			
			
			If[handle["Return"] =!= 0,
				Which[
					MemberQ[$sslErrorCodes, handle["Return"]],
					Message[URLFetch::ssl,CURLLink`CURLErrorData[handle],handle["Return"]]
					,
					True,
					Message[URLFetch::invhttp, CURLLink`CURLErrorData[handle],handle["Return"]];
					
					
				];
				
				If[!successQ[handle],
					CURLLink`CURLHandleUnload[handle];
					Return[$Failed]];
				
			];
		
			If[(OptionValue[System`CookieFunction]=!=None)&&OptionValue["StoreCookies"] ,
				storeCookies[HTTPData[handle, "Cookies"]]
			]; 
		
			elements = If[handle["FTP"]===True, $FTPFetchElements, $URLFetchElements];
			(* Perhaps the user just wants to know the available output types. *)
			If[res === "Elements",
					Return[elements]
			];
			output = parseElements[handle, requestedElements, elements];
			(*
			if non blocking,i.e stream element is 
			requested then unload happens in c-code
			when Close[InputStream[..]] is called
			*)
			If[isblocking,CURLLink`CURLHandleUnload[handle]];
			unsetCURLHandle[handle];
			unsetHTTPData[handle];
		,CURLLink`Utilities`Exception];
		
		If[error === $Failed,
			$Failed,
			cookiefunction[OptionValue[System`CookieFunction]];
			output
		]
	]
	
	
URLFetch::invhttp = "A library error occurred. The raw details are: \"libcurl error (`2`): `1`\"";
URLFetch::ssl="An SSL error occurred. The raw details are: \"libcurl error (`2`): `1`\"";
URLFetch::noelem = "The element \"`1`\" is not allowed."
URLFetch::invurl="`1` is not a valid URL";
URLFetchAsynchronous::invurl="`1` is not a valid URL";
	
(****************************************************************************)
(* URLSave... *)
Options[httpSave]=Options[URLSave] 

httpSave[urlExp_, options___?OptionQ] := 
	httpSave[urlExp, Automatic, options]

httpSave[urlExp_/;isURL[urlExp,URLSave], Automatic|None|Null, rest___] := 
	httpSave[urlExp, FileNameJoin[{$TemporaryDirectory, CreateUUID[] <> ".tmp"}], rest]

httpSave[urlExp_, file_, res:(_String|_List|All):"Content", options___?OptionQ] /; deprecatedOptionQ[options] := 
	httpSave[urlExp, file, deprecatedOptionFix[URLSave, options]]  
	
httpSave[urlExp_/;isURL[urlExp,URLSave], fileExp_/;isFile[fileExp,URLSave], res:(_String|_List|All):"Content", opts:OptionsPattern[URLSave]] /; initializeQ[] :=
	Module[{responseCode,httpConnectCode,oldhandle,handle, output, error,file ,stdOpts, elements, wellFormedURL,url},
		setMessageHead[URLSave];
		
		url=getURL[urlExp];
		
		file=getFile[fileExp];
		
		$LastKnownCookies=CURLLink`Cookies`GetKnownCookies[];
		
		stdOpts = Flatten[{opts, FilterRules[Options[URLSave], Except[{opts}]]}];
		error = Catch[
			handle = commonInit[<|"URL"->url, "Function"->URLSave,"FileName"->ExpandFileName[file], "Format"->OptionValue[BinaryFormat], "Options"->stdOpts|>];
			If[handle === $Failed,
				Return[$Failed]
			];
			handle["Return"] = CURLLink`CURLWait[handle];
			
			responseCode = CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
			
			httpConnectCode = CURLLink`CURLGetInfo[handle,"CURLINFO_HTTP_CONNECTCODE"];
			
			If[(responseCode === 407 ||httpConnectCode == 407)&& allowCredentialDialog[opts],
				CURLLink`CURLMultiHandleRemove[$WolframHTTPMultiHandle,handle];
				CURLLink`CURLHandleUnload[handle];
				handle=commonInit[<|"URL"->url,"Function"-> URLSave,"FileName"->ExpandFileName[file],"Format"->OptionValue[BinaryFormat], "Options"->stdOpts,"Code"->407,"RequestAuthentication"->OptionValue["CredentialsProvider"]|>];
				handle["Return"] = CURLLink`CURLWait[handle];
				responseCode = CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
				httpConnectCode = CURLLink`CURLGetInfo[handle,"CURLINFO_HTTP_CONNECTCODE"];

			];
			If[CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"] === 407||CURLLink`CURLGetInfo[handle,"CURLINFO_HTTP_CONNECTCODE"] === 407,
				$proxyCache = False;
				
			];
			If[(responseCode === 401 ||httpConnectCode == 401) && allowCredentialDialog[opts],
				CURLLink`CURLMultiHandleRemove[$WolframHTTPMultiHandle,handle];
				CURLLink`CURLHandleUnload[handle];
				handle=commonInit[<|"URL"->url,"Function"-> URLSave,"FileName"->ExpandFileName[file],"Format"->OptionValue[BinaryFormat], "Code"->401,"Options"->stdOpts,"RequestAuthentication"->OptionValue["CredentialsProvider"]|>];
				handle["Return"] = CURLLink`CURLWait[handle];
				responseCode = CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
				httpConnectCode = CURLLink`CURLGetInfo[handle,"CURLINFO_HTTP_CONNECTCODE"];
				handle=retryIfSSLError[handle,url,URLSave,ExpandFileName[file],stdOpts];
				If[streamIsRequested,Close[stream],CURLLink`CURLHandleUnload[oldhandle]];
			];
			If[CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"] === 401, 
				wellFormedURL = If[StringMatchQ[url, {"http://*", "https://*", "ftp://*", "ftps://*"}], 
					URIJoin[Flatten@{URISplit[url]}]
				(*else*), 
					URIJoin[Flatten@{URISplit["http://" <> url]}]
				];				
				clearCredentials[url];
			];
			
			If[CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"] === 407,
				$proxyCache = False;
			];
			
			
			If[handle["Return"] =!= 0,
				Which[
					MemberQ[$sslErrorCodes, handle["Return"]],
					Message[URLSave::ssl,CURLLink`CURLErrorData[handle],handle["Return"]]
					,
					True,
					Message[URLSave::invhttp, CURLLink`CURLErrorData[handle],handle["Return"]];
					
					
				];
				
				If[!successQ[handle],
					CURLLink`CURLHandleUnload[handle];
					Return[$Failed]];
				
			];
		
			If[(OptionValue[System`CookieFunction]=!=None)&&OptionValue["StoreCookies"] && (OptionValue["Cookies"] =!= Automatic),
				storeCookies[HTTPData[handle, "Cookies"]]
			]; 
		
			elements = If[handle["FTP"]===True, $FTPSaveElements, $URLSaveElements];
			(* Perhaps the user just wants to know the available output types. *)
			If[res === "Elements",
				Return[elements]
			];
	
			output = 
				If[res === "Content",
					file
				(*else*), 
					parseElements[handle, res, elements]
				];
		
			CURLLink`CURLHandleUnload[handle];
			unsetCURLHandle[handle];
		,CURLLink`Utilities`Exception];
		
		If[error === $Failed,
			$Failed,
			cookiefunction[OptionValue[System`CookieFunction]];
			output
		]
	]

URLSave::invhttp = "A library error occurred. The raw details are: \"libcurl error (`2`): `1`\"";
URLSave::ssl="An SSL error occurred. The raw details are: \"libcurl error (`2`): `1`\"";
URLSave::noelem = "The element \"`1`\" is not allowed.";
URLSave::invurl="`1` is not a valid URL";
URLSaveAsynchronous::invurl="`1` is not a valid URL";
URLSave::invfile="`1` is not a valid File";
URLSaveAsynchronous::invfile="`1` is not a valid File";
(****************************************************************************)
(* Useful functions for both URLFetch and URLSave *)

setMessageHead[head_] := $MessageHead = head;
curlMessage[head_, tag_, args___] := Message[MessageName[head, tag], args]

connectQ[] :=
	If[$AllowInternet,
		True
	(*else*),
		Message[URLFetch::offline];
		False
	];
	
(* Check all the options passed are valid. *)
validOptionsQ[opts_, func_] := 
	Module[{},
		If[opts === {},
			Return[True]
		];
		
		If[FilterRules[opts, Except[Options[func]]] =!= {},
			Message[General::optx, First[#], InString[$Line]] & /@ FilterRules[opts, Except[Options[func]]];
			Return[False];	
		];

		If[!MatchQ[(Method /. opts),_String|Automatic] || (Method /. opts) === "",
			If[!StringQ[("Method" /. opts)] || StringMatchQ[( "Method"/. opts), "Method"] || ("Method" /. opts) === "",
				Message[General::erropts, (Method /. opts /. Method -> "Method" ) /. opts, "Method"];
				Return[False];
			];
		];
		
		If[!MatchQ[("Headers" /. opts), List[Rule[_String, _String]...]],
			Message[General::erropts, "Headers" /. opts, "Headers"];
			Return[False];
		];
		
		If[!(StringQ[("Username" /. opts)]||MatchQ[("Username" /. opts),None]),
			Message[General::erropts, "Username" /. opts, "Username"];
			Return[False];
		];
		
		If[!(StringQ[("Password" /. opts)]||MatchQ[("Password" /. opts),None]),
			Message[General::erropts, "Password" /. opts, "Password"];
			Return[False];
		];
		If[!(StringQ[("ProxyUsername" /. opts)]||MatchQ[("ProxyUsername" /. opts),None]),
			Message[General::erropts, "ProxyUsername" /. opts, "ProxyUsername"];
			Return[False];
		];
		
		If[!(StringQ[("ProxyPassword" /. opts)]||MatchQ[("ProxyPassword" /. opts),None]),
			Message[General::erropts, "ProxyPassword" /. opts, "ProxyPassword"];
			Return[False];
		];
		If[("UserAgent" /. opts)=!= Automatic && !StringQ[("UserAgent" /. opts)],
			Message[General::erropts, "UserAgent" /. opts, "UserAgent"];
			Return[False];
		];
	
		If[!MatchQ[("VerifyPeer" /. opts), True|False],
			Message[General::erropts, "VerifyPeer" /. opts, "VerifyPeer"];
			Return[False];
		];
		
		If[!MatchQ[("StoreCookies" /. opts), True|False],
			Message[General::erropts, "StoreCookies" /. opts, "StoreCookies"];
			Return[False];
		];
		
		If[("Parameters" /. opts) === "Parameters",
			Return[True];
		];
		
		If[!MatchQ["Parameters" /. opts, List[Rule[_String, _String]...]],
			Message[General::erropts, "Parameters" /. opts, "Parameters"];
			Return[False];
		];

		If[!MatchQ[("Body" /. opts),None| _String|List[___Integer]|_?ByteArrayQ],
			Message[General::erropts, "Body" /. opts, "Body"];
			Return[False];
		];
		
		If[("MultipartElements" /. opts) =!= {} && (
                                                    !MatchQ[("MultipartElements" /. opts), {{_String, _String,___String, {__Integer}}..}] &&
                                                    !MatchQ[("MultipartElements" /. opts), {Rule[{_String, _String|Automatic,___String}, {__Integer}|File[_String]|_String]..}] &&
                                                    !MatchQ[("MultipartElements" /. opts), {Rule[{_String, _String|Automatic,___String}, _String]..}]
                                                    
                                                    ),
			Message[General::erropts, "MultipartElements" /. opts, "MultipartElements"];
			Return[False];
		];
		
		If[!NonNegative[("ConnectTimeout" /. opts)],
			Message[General::erropts, "ConnectTimeout" /. opts, "ConnectTimeout"];
			Return[False];
		];
		
		If[!NonNegative[("ReadTimeout" /. opts)],
			Message[General::erropts, "ReadTimeout" /. opts, "ReadTimeout"];
			Return[False];
		];
		
		If[!validConnectionSettingsQ[ConnectionSettings/.opts],
			Return[False]
		];		
	
		If[!BooleanQ[("Debug" /. opts)],
			Message[General::erropts, "Debug" /. opts, "Debug"];
			Return[False];
		];
		(* If we made it here, all the options should be valid. *)
		True
	]

validConnectionSettingsQ[connectionSettings_]:=
Module[
	{maxUploadSpeed,maxDownloadSpeed,res=True},
	res=
	Catch[
		If[!AssociationQ[Association[connectionSettings]] ,
			Message[General::erropts, connectionSettings, ConnectionSettings];
			Throw[False];
			,
			maxUploadSpeed = Lookup[connectionSettings,"MaxUploadSpeed",Automatic];
			maxDownloadSpeed = Lookup[connectionSettings,"MaxDownloadSpeed",Automatic];
		];
			
		If[QuantityQ[maxDownloadSpeed],
			If[!CompatibleUnitQ[maxDownloadSpeed,"Bytes"/"Seconds"],
				Message[General::erropts, maxDownloadSpeed, "MaxDownloadSpeed"];
				Throw[False];
			]
		];
		
		If[QuantityQ[maxUploadSpeed],
			If[!CompatibleUnitQ[maxUploadSpeed,"Bytes"/"Seconds"],
				Message[General::erropts, maxUploadSpeed, "MaxUploadSpeed"];
				Throw[False];
			]
		];
		
		If[!TrueQ[NonNegative[N[maxDownloadSpeed]]] && maxDownloadSpeed =!= Automatic,
				Message[General::erropts, maxDownloadSpeed, "MaxDownloadSpeed"];
				Throw[False];
		];
		
		If[!TrueQ[NonNegative[N[maxUploadSpeed]]] &&  maxUploadSpeed =!= Automatic,
				Message[General::erropts, maxUploadSpeed, "MaxUploadSpeed"];
				Throw[False];
		];
			
		True
	];
	
	res
]
	
configureWolframMultiHandle[multihandle_,options_?AssociationQ]:=
Module[
	{maxConnections,maxConnectionsPerHost},
	
	maxConnections = Lookup[options, "MaxConnections", $MaxConnections];
			
	maxConnectionsPerHost = Lookup[options, "MaxConnectionsPerHost", $MaxConnectionsPerHost];
	
	CURLLink`CURLOption[multihandle,"CURLMOPT_PIPELINING",$PipeLining];
					
	CURLLink`CURLOption[multihandle,"CURLMOPT_MAX_PIPELINE_LENGTH",16];
					
	CURLLink`CURLOption[multihandle,"CURLMOPT_MAX_TOTAL_CONNECTIONS",maxConnections];
				
	CURLLink`CURLOption[multihandle,"CURLMOPT_MAX_HOST_CONNECTIONS",maxConnectionsPerHost];
]

renameKeys = 
Function[
	{key},
	StringReplace[
					key, 
					{
						"MaxSequentialConnections" -> "MaxConnections"
						,
						"MaxSequentialConnectionsPerHost" -> "MaxConnectionsPerHost"
						,
						"MaxConcurrentConnections" -> "MaxConnections"
						,
						"MaxConcurrentConnectionsPerHost" -> "MaxConnectionsPerHost"
					}
				 ]
		]
getConnectionOptions[func_/;MatchQ[func,URLFetch|URLSave]]:=
Module[
	{hostConnectionOptions},
	hostConnectionOptions = Association[Replace["HostConnectionOptions", SystemOptions["HostConnectionOptions"]]];
	hostConnectionOptions = KeyTake[hostConnectionOptions,{"MaxSequentialConnections","MaxSequentialConnectionsPerHost"}];
	hostConnectionOptions = KeyMap[renameKeys,hostConnectionOptions];
	hostConnectionOptions
]
getConnectionOptions[func_/;MatchQ[func,URLFetchAsynchronous|URLSaveAsynchronous]]:=
Module[
	{hostConnectionOptions},
	hostConnectionOptions = Association[Replace["HostConnectionOptions", SystemOptions["HostConnectionOptions"]]];
	hostConnectionOptions = KeyTake[hostConnectionOptions,{"MaxConcurrentConnections","MaxConcurrentConnectionsPerHost"}];
	hostConnectionOptions = KeyMap[renameKeys,hostConnectionOptions];
	hostConnectionOptions

]
getConnectionOptions[___]:=$Failed

createWolframMultiHandle[func_]:=
Module[
	{multihandle},
	multihandle = CURLLink`CURLMultiHandleCreate[];
	configureWolframMultiHandle[multihandle,getConnectionOptions[func]];			
	CURLLink`CURLCreateRunThread[multihandle];
	multihandle
]
initializeGlobalMultiHandles[]:=
(
If[Not[TrueQ[ValueQ[$WolframHTTPMultiHandleAsync]]],$WolframHTTPMultiHandleAsync = createWolframMultiHandle[URLFetchAsynchronous];];
If[Not[TrueQ[ValueQ[$WolframHTTPMultiHandle]]],	$WolframHTTPMultiHandle = createWolframMultiHandle[URLFetch]];

)
getGlobalMultiHandle[func_]:=
(
initializeGlobalMultiHandles[];
Which[
	MatchQ[func,URLFetch|URLSave],
	$WolframHTTPMultiHandle,
	MatchQ[func,URLFetchAsynchronous|URLSaveAsynchronous],
	$WolframHTTPMultiHandleAsync,
	True,
	$Failed
	]
)
(* Initialization routines common to both URLSave and URLFetch. *)
CheckAndSetCredentials[handle_,url_]:=
Module[
	{credHandle=None},
	credHandle = getCredentialHandle[url];
	If[credHandle =!=None,
		CURLLink`CURLAssociateCredentialHandle[credHandle,handle];
	];	
]
CheckAndSetCredentials[handle_,url_,"Proxy"]:=
Module[
	{credHandle=None,proxyuser,proxypass},
	credHandle = getCredentialHandle[url];
	If[credHandle =!=None,
		proxyuser = CURLGetCredentials[credHandle,"Username"];
		proxypass = CURLGetCredentials[credHandle,"Password"];
		CURLLink`CURLOption[handle, "CURLOPT_PROXYUSERNAME", proxyuser];
		CURLLink`CURLOption[handle, "CURLOPT_PROXYPASSWORD", proxypass];
	]
]
getCredentialHandle[url_]:=
Module[
	{domain,credHandle},
	domain = getDomain[url];
	credHandle = Lookup[$HTTPCredentialStorage,domain,Lookup[$HTTPCredentialStorage,url,None]];
	credHandle
]

commonInit[data_Association] := 
	Module[
		{handle,multiHandle,norevoke=2,code=Lookup[data,"Code",200],
		url=Lookup[data,"URL",""],func=Lookup[data,"Function",URLFetch],fileName=Lookup[data,"FileName",None],format=Lookup[data,"Format",None],opts=Lookup[data,"Options",{}],
		requestAuthentication=Lookup[data,"RequestAuthentication",False],cert
		},
		cert = CURLLink`CURLGetClientCertificate[url];
		(* First determine if we're allowed to connect to the internet. *)	
		If[!connectQ[],
			Return[$Failed]
		];
		(* Now check all the options passed are valid. *)
		If[!validOptionsQ[Flatten[opts], func],
			Return[$Failed]	
		];
		
		Which[
			format==="Stream",
			multiHandle = CURLLink`CURLMultiHandleCreate[];
			,
			True,
			multiHandle = getGlobalMultiHandle[func];
			];
		handle = CURLLink`CURLHandleLoad[];
		
		handle["FTP"] = StringMatchQ[url, {"ftp://"~~___, "ftps://"~~___}];
		If[MatchQ[("UseProxy" /. PacletManager`$InternetProxyRules),True ],
			handle["Proxies"] = getProxies[url, "UseProxy" /. PacletManager`$InternetProxyRules];
			If[handle["Proxies"]=!={} && handle["Proxies"]=!={""},
				CURLLink`CURLSetProxies[handle, #] & /@ handle["Proxies"];
				CURLLink`CURLOption[handle, "CURLOPT_PROXY",handle["Proxies"][[1]]];
				If[MatchQ[requestAuthentication,Automatic|True] && code === 407,proxyCredentials[handle, url,handle["Proxies"][[1]]]]
			]
			,
			handle["Proxies"] = {}
		];
		If[(("UseProxy" /. PacletManager`$InternetProxyRules) === Automatic),
			handle["Proxies"] = getProxies[url, "UseProxy" /. PacletManager`$InternetProxyRules];
			If[handle["Proxies"]=!={} && handle["Proxies"]=!={""},
				CURLLink`CURLSetProxies[handle, #] & /@ handle["Proxies"];
				CURLLink`CURLOption[handle, "CURLOPT_PROXY",handle["Proxies"][[1]]];
				If[MatchQ[requestAuthentication,Automatic|True] && code === 407,
					proxyCredentials[handle, url,handle["Proxies"][[1]]];
					
					]
			]
			,
			handle["Proxies"] = {}	
		];

		CURLLink`CURLOption[handle, "CURLOPT_PROXYAUTH", 15];
		(* A bit mask passed to libcurl to indicated HTTP,HTTPS,FTP, and FTPS are the only allowed protocols *)
		CURLLink`CURLOption[handle, "CURLOPT_PROTOCOLS", 15]; 
		CURLLink`CURLOption[handle, "CURLOPT_NOSIGNAL", True];
		If[((System`CookieFunction/.opts)===None || !("StoreCookies" /. opts)) && ("Cookies" /. opts) === Automatic,
			setStandardOptions[handle, url, FilterRules[{Flatten[FilterRules[opts, Except["Cookies"]]], "Cookies"->$HTTPCookies}, $StandardOptions]]
		(*else*),
			setStandardOptions[handle, url, FilterRules[Flatten[opts], $StandardOptions]]
		];
		If[MatchQ[requestAuthentication,Automatic|True] && code === 401 ,credWrapper[handle, url, requestAuthentication]];
		
		CheckAndSetCredentials[handle,url];
		(*If using proxies, check if proxy credentials are saved or need to be saved*)
		If[ListQ[handle["Proxies"]] && handle["Proxies"]=!={} && handle["Proxies"]=!={""},
			CheckAndSetCredentials[handle,handle["Proxies"][[1]],"Proxy"];
		];
		CURLLink`CURLOption[handle, "CURLOPT_ACCEPT_ENCODING", ""];
		CURLLink`CURLOption[handle, "CURLOPT_TCP_KEEPALIVE", 1];

		CURLLink`CURLOption[handle,"CURLOPT_SSLCERT",cert];

		
		(*
		Turn on storage for more precise error messages.
		Mostly useful when transfer fails due to ssl error
		*)
		CURLLink`CURLOption[handle, "CURLOPT_ERRORBUFFER", 1];
		
		If[$OperatingSystem==="Windows",CURLLink`CURLOption[handle, "CURLOPT_SSL_OPTIONS",norevoke]];
		
		CURLLink`CURLOption[handle, "CURLOPT_HEADERFUNCTION", "WRITE_HEADER"];
		CURLLink`CURLOption[handle, "CURLOPT_WRITEHEADER", "MemoryPointer"];
		
		Which[
			MatchQ[func,URLSave|URLSaveAsynchronous],
			CURLLink`CURLFileInfo[handle, fileName, format];
			CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "WRITE_FILE"];
			CURLLink`CURLOption[handle, "CURLOPT_WRITEDATA", "FilePointer"];
			,
			True,
			CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "WRITE_MEMORY"];
			CURLLink`CURLOption[handle, "CURLOPT_WRITEDATA", "MemoryPointer"];
			If["Debug"/.opts,
			CURLLink`CURLOption[handle, "CURLOPT_VERBOSE", True];
			CURLLink`CURLOption[handle, "CURLOPT_DEBUGFUNCTION", "WRITE_MEMORY_DEBUG"];
			CURLLink`CURLOption[handle, "CURLOPT_DEBUGDATA", "MemoryPointer"];
			];
			
		];
		
		If[MatchQ[func,URLFetchAsynchronous|URLSaveAsynchronous],
			CURLLink`CURLOption[handle,"CURLOPT_NOPROGRESS",Not["Progress"/.opts]];
			CURLLink`CURLOption[handle, "CURLOPT_XFERINFOFUNCTION", "SHOW_PROGRESS"];
			CURLLink`CURLOption[handle, "CURLOPT_XFERINFODATA", "MemoryPointer"];
			If[Replace["Transfer",opts]==="Chunks",
				CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "SEND_CHUNKS"];
				CURLLink`CURLOption[handle, "CURLOPT_WRITEDATA", "MemoryPointer"];
			];		
			handle,
		CURLLink`CURLMultiHandleAdd[multiHandle,{handle}];
		handle]
	]

(****************************************************************************)
(* Parse elements to return correct data *)
parseElements[handlesAndElements_List,out_]:=Map[parseElements[#[[1]],out,#[[2]]]&,handlesAndElements]
parseElements[handle_, out_, elements_] :=
	Module[{},
		Which[
			out === All, parseAll[handle, out, elements],
			StringQ[out], parseString[handle, out, elements],
			ListQ[out], parseList[handle, out, elements],
			True, Throw[$Failed,CURLLink`Utilities`Exception]
		]
	]

parseAll[handle_, All, elements_] := HTTPData[handle, #] & /@ elements;

parseString[handle_, "All", elements_] := parseAll[handle, All, elements] 
parseString[handle_, "Rules", elements_] := Rule @@@ Partition[Riffle[elements, (HTTPData[handle, #] & /@ elements)], 2]
parseString[handle_, str_String, elements_] := If[MemberQ[elements, str], HTTPData[handle, str],  curlMessage[$MessageHead, "noelem", ToString[str]]; Throw[$Failed,CURLLink`Utilities`Exception]]

parseList[handle_, list_List, elements_] :=
	Module[{subList},
		If[Length[list] === 1,
			Which[
				StringQ[First[list]], parseString[handle, First[list], elements],
				ListQ[First[list]], 
					subList = First[list];
					If[Length[subList] === 1,
						If[StringQ[First[subList]], 
							parseString[handle, First[subList], elements]
						(*else*),
							Return[$Failed]
						]
					(*else*),
						HTTPData[handle, #] & /@ subList	
					]
				,
				True, Return[$Failed]
			]
		(*else*),
			(*Special Case: {"Rules", {...}}*)
			If[MatchQ[list, {"Rules", List[_String ...]}],
				parseString[handle, "Rules", Last[list]]
			(*else*),
				parseString[handle, #, elements] & /@ list
			]
		]
	]

(****************************************************************************)

buildData[handle_CURLLink`CURLHandle, data_List, method_String] /; initializeQ[] := 
	Quiet[
		Check[
			StringExpression[ 
				Sequence @@ 
				Riffle[CURLLink`CURLEscape[ToString[First[#]]] <> "=" <> CURLLink`CURLEscape[ToString[Last[#]]] & /@ data, "&"]
			], 
			Throw[$Failed,CURLLink`Utilities`Exception]
		]
	]

HTTPData[handles_List,element_String]:=Map[HTTPData[#,element]&,handles]	
(****************************************************************************)

(* stream *)
HTTPData[handle : CURLLink`CURLHandle[id_], "Stream"] /;successQ[handle] :=HTTPData[ CURLLink`CURLHandle[id], "Stream"]=
 (OpenRead[handle["URL"],Method -> {"HTTPStreamElement", "CURLHandle" -> id}])
 
(****************************************************************************)

(* Return the headers of a CURLHandle as a list of rules. *)
HTTPData[handle_CURLLink`CURLHandle, "Headers"] /; successQ[handle] := 
	impHeaders[CURLLink`CURLHeaderData[handle]];

HTTPData[handle_CURLLink`CURLHandle, "HeadersReceived"] /; successQ[handle] :=
With[{lastheader=Last[StringSplit[CURLLink`CURLHeaderData[handle],"\r\n\r\n"]]}, 
	impHeaders[lastheader]
]
impHeaders[string_]:=Cases[StringSplit[StringSplit[string, "\r\n"], ": ", 2], {_, _}]
(****************************************************************************)
HTTPData[handle_CURLLink`CURLHandle, "Cookies"] /; successQ[handle] := 
	Cases[
		{	"Domain"-> First[#], 
			If[#[[2]] === "FALSE",
				"MachineAccess"-> #[[2]]
			],
			"Path"->#[[3]], 
			"Secure"->#[[4]], 
			"ExpirationDate"->DateString[ToExpression[#[[5]]] + AbsoluteTime[{1970, 1, 1, 0, 0, 0}]], 
			"Name"->#[[6]], 
			"Value"->Last[#]
		}, 
		Except[Null]] & /@ StringSplit[StringSplit[CURLLink`CURLCookies[handle], "\n"], "\t"];

(****************************************************************************)
addHeaders[handle_CURLLink`CURLHandle, headers_List] :=
	CURLLink`CURLAddHeader[handle, StringReplace[ToString[First[#]] <> ": " <> ToString[Last[#]], "\n"->""]] & /@ headers


(****************************************************************************)
addCookies[handle_CURLLink`CURLHandle, ucookies_List] := 
	Module[{errs,cookies=ucookies},
		errs = Catch[
			If[cookies === {},
				CURLLink`CURLOption[handle, "CURLOPT_COOKIELIST", ""];
				Return[]		
			];
			(*if list of Assocs. is passed covert it to list of rules*)
			CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST","SESS"];
			CURLLink`CURLOption[handle,"CURLOPT_COOKIEJAR",Quiet@Check[getFile@System`$CookieStore,""]];
			CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST","FLUSH"];
			CURLLink`CURLOption[handle,"CURLOPT_COOKIEJAR",""];
			cookies=Map[Association,ucookies];
			cookies=Map[CURLLink`Cookies`Private`toOldCookie, cookies];
			CURLLink`CURLOption[handle, "CURLOPT_COOKIELIST", 
				StringJoin[
					ReleaseHold[{
							"Domain", "\t", 
							If[("MachineAccess" /. #) === "MachineAccess",
								"TRUE",
								"MachineAccess"
							], "\t",  			
							"Path", "\t", 
							"Secure", "\t", 
							Hold[ToString[AbsoluteTime["Expires"] - AbsoluteTime[{1970, 1, 1, 0, 0, 0}]]], "\t", 
							"Name", "\t", 
							"Value"
						} /. Rule @@@ #
					]
				]
			] & /@ cookies;
			(*Save to $CookieStore*)
			
			
			
			
		,CURLLink`Utilities`Exception];
	]




	
storeCookies[cookies_List] /; initializeQ[] :=
	Module[{handle},
		handle = CURLLink`CURLHandleLoad[];
		setStandardOptions[handle, ""];
		addCookies[handle, cookies];
		
		CURLLink`CURLHandleUnload[handle];
	]

$LastKnownCookies="";

cookiefunction[f_] := 
	Module[{cookiesreceived,allcookies,handle,res},
		handle = CURLLink`CURLHandleLoad[];
		setStandardOptions[handle, ""];
		allcookies=CURLLink`CURLCookies[handle];
		CURLLink`CURLHandleUnload[handle];
		cookiesreceived=StringDelete[allcookies,$LastKnownCookies];
		res=CURLLink`Cookies`Private`cookiesToAssociation[cookiesreceived];
		Map[System`ClearCookies[#]&,res];     
		If[f===Automatic,System`SetCookies[res],Map[f,res]]

	]


$HTTPCookies /; initializeQ[] :=
	Module[{cookies, handle, error},
		error = Catch[
			handle = CURLLink`CURLHandleLoad[];
			setStandardOptions[handle, ""];
			handle["Return"] = 0;
			cookies = HTTPData[handle, "Cookies"];
			CURLLink`CURLHandleUnload[handle];
		,CURLLink`Utilities`Exception];
		If[error === $Failed, $Failed, cookies]
	]
	

(****************************************************************************)

(* Return the headers of a CURLHandle as a list of rules. *)
HTTPData[handle_CURLLink`CURLHandle, "Headers"] /; successQ[handle] := 
	Cases[StringSplit[StringSplit[CURLLink`CURLHeaderData[handle], "\r\n"], ": ", 2], {_, _}];
	

(* Return the cookies used for by this CURLHandle. *)
HTTPData[handle_CURLLink`CURLHandle, "Cookies"] /; successQ[handle] := 
	Cases[
		{	"Domain"-> First[#], 
			If[#[[2]] === "FALSE",
				"MachineAccess"-> #[[2]]
			],
			"Path"->#[[3]], 
			"Secure"->#[[4]], 
			"ExpirationDate"->DateString[ToExpression[#[[5]]] + AbsoluteTime[{1970, 1, 1, 0, 0, 0}]], 
			"Name"->#[[6]], 
			"Value"->Last[#]
		}, 
		Except[Null]] & /@ StringSplit[StringSplit[CURLLink`CURLCookies[handle], "\n"], "\t"];
		
HTTPData[handle_CURLLink`CURLHandle, "CookiesReceived"] /; successQ[handle]:=
Module[
	{allcookies,cookiesreceived,chandle},
	chandle = CURLLink`CURLHandleLoad[];
	setStandardOptions[chandle, ""];
	allcookies=CURLLink`CURLCookies[chandle];
	chandle["Return"] = 0;
	CURLLink`CURLHandleUnload[chandle];
	
	cookiesreceived=StringDelete[allcookies,$LastKnownCookies];
	CURLLink`Cookies`Private`cookiesToAssociation[cookiesreceived]
]


(* Return the content as a list of bytes of a given CURLHandle. *)
HTTPData[handle_CURLLink`CURLHandle, "ContentData"] /; successQ[handle] := 
		Normal[HTTPData[handle, "BodyByteArray"]]

HTTPData[handle_CURLLink`CURLHandle, "DebugContentData"] /; successQ[handle] := 
		Normal[HTTPData[handle, "DebugBodyByteArray"]]


HTTPData[handle_CURLLink`CURLHandle, "BodyByteArray"] /; successQ[handle] :=
		CURLLink`CURLRawContentDataRawArray[handle]

HTTPData[handle_CURLLink`CURLHandle, "DebugBodyByteArray"] /; successQ[handle] :=
		CURLLink`CURLDebugRawContentDataRawArray[handle]	
(* Return the content as a String of a given CURLHandle. *)
HTTPData[handle_CURLLink`CURLHandle, "Content"] /; successQ[handle] :=
Module[
	{bytes,   mCharset="ISO8859-1"}, 
	
	bytes = HTTPData[handle, "BodyByteArray"];
	Which[
		handle["FTP"] === True,
			Quiet[Check[ByteArrayToString[bytes, "UTF-8"], ByteArrayToString[bytes, "ISO8859-1"]]]
		,
		True,
			mCharset=getCharacterEncoding[HTTPData[handle, "Headers"]];
			Quiet[Check[ByteArrayToString[bytes, mCharset], ByteArrayToString[bytes,"ISO8859-1"]]]
		]
	]
HTTPData[handle_CURLLink`CURLHandle, "DebugContent"] /; successQ[handle] :=
Module[
	{bytes,   mCharset="ISO8859-1"}, 
	
	bytes = HTTPData[handle, "DebugBodyByteArray"];
	Which[
		handle["FTP"] === True,
			Quiet[Check[ByteArrayToString[bytes, "UTF-8"], ByteArrayToString[bytes, "ISO8859-1"]]]
		,
		True,
			mCharset=getCharacterEncoding[HTTPData[handle, "Headers"]];
			Quiet[Check[ByteArrayToString[bytes, mCharset], ByteArrayToString[bytes,"ISO8859-1"]]]
		]
	]	
getCharacterEncoding[headers_]:=
Module[
	{contentTypeHeader,contentType,charset,mCharset="ISO8859-1"},
	contentTypeHeader=Select[headers, StringMatchQ[First[#], "Content-Type", IgnoreCase -> True] &];
	If[MatchQ[contentTypeHeader,{{_String,_String}..}],
				(*then*)
				contentType = contentTypeHeader[[-1,2]];
				charset = StringCases[contentType,StartOfString ~~ ___ ~~(" "|";")~~ "charset=" ~~ c__ ~~ (WhitespaceCharacter | EndOfString) :> c,IgnoreCase->True ];
				mCharset = If[charset=!={},charsetToMCharset[First[charset]],"ISO8859-1"];
				];
	mCharset		
]
charsetToMCharset["utf-8"]="UTF-8"
charsetToMCharset["ISO-8859-1"]="ISO8859-1"
charsetToMCharset[charset_] := (
    CloudObject; (* make sure the CloudObject paclet is loaded *)
    CloudObject`ToCharacterEncoding[charset, "ISO8859-1"]
)

		
(* Return the status code as an Integer of a given CURLHandle. *)
HTTPData[handle_CURLLink`CURLHandle, "StatusCode"] /; successQ[handle] := 
	Module[{code},
		code=CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
		If[	code===0,
			code = CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
		 ];
		 code
		]
(* Catch all for bad types *)
HTTPData[handle_CURLLink`CURLHandle, unknown_String/;!MemberQ[$returnTypes,unknown]] := (curlMessage[$MessageHead, "noelem", ToString[unknown]]; Throw[$Failed,CURLLink`Utilities`Exception])

(****************************************************************************)
URISplit[uri_String] := 
	Flatten[
		StringCases[
			uri, 
			RegularExpression[ "^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?"] -> 
			   	{"Scheme" -> "$2", "Authority" -> "$4"}
		]
	]

URIJoin[uri : List[_Rule, _Rule]] := 
	Module[{scheme, authority},
		If[!Fold[And, True, Map[MatchQ[#, Rule[_String, _String]] &, uri]], Return[$Failed]]; 
		{scheme, authority} = Map[Last, uri];
		StringJoin[
			Cases[{
				If[scheme =!= "", StringJoin[scheme, ":"]],
				Which[
					authority =!= "" && scheme =!= "", StringJoin["//", authority],
					authority === "" && scheme =!= "", authority
				]
			}, Except[Null]]
		]
	]

(****************************************************************************)

buildProxy[{scheme_String, url_String}] := If[StringMatchQ[url, scheme <> "://*"], url, scheme <> "://" <> url]
buildProxy[{url_String}] := url
buildProxy[url_String] := url	
tourl[{url_String,port_Integer}]:=url<>":"<>ToString[port]
tourl[{url_String}]:=url
tourl[___]:=Nothing
wlproxy[proto_String]:=
	Module[
		{url,fullurl},
		fullurl=proto/.PacletManager`$InternetProxyRules;
		url=tourl[fullurl];
		url
		
		]	
getProxies[url_String, False] = {""}

(*when UseProxy->True,in $InternetProxyRules, and scheme isn't https or ftp,
 getProxies defaults to HTTP and Socks proxies given by $InternetProxyRules *)
getProxies[url_String, True] :=
	Module[
		{proxies,scheme = URLParse[url,"Scheme"]},
		proxies=Which[
					scheme==="https",
					{wlproxy["HTTPS"],wlproxy["Socks"]}
					,
					scheme==="ftp",
					{wlproxy["FTP"],wlproxy["Socks"]}
					,
					True,
					{wlproxy["HTTP"],wlproxy["Socks"]}
					];
		If[proxies==={},{""},proxies]
	]
getProxies[url_String, Automatic] :=
	getSystemProxies[url, $OperatingSystem]
	
getSystemProxies[url_String, "Windows"] :=
Module[
	{rawProxies, proxies,useWPAD,res}
	,
	useWPAD=If[Lookup[PacletManager`$InternetProxyRules,"UseWPAD",False],1,0];
	
	If[StringMatchQ[url, {"http://*", "https://*", "ftp://*", "ftps://*"}],
		rawProxies = CURLLink`CURLGetProxies[url,useWPAD]
		,
		(*else*)
		rawProxies = CURLLink`CURLGetProxies["http://" <> url,useWPAD]
	];
	
	proxies = StringSplit[StringSplit[rawProxies, ";"], "=", 2];

	res=buildProxy[#] & /@ proxies;

	res
]	
	
getSystemProxies[url_String, "MacOSX"] := 
	Module[{},	
		If[StringMatchQ[url, {"http://*", "https://*", "ftp://*", "ftps://*"}],
			Flatten@{Quiet[Check[CURLLink`CURLGetProxies[URIJoin[Flatten@{URISplit[url]}]], {}]]}
		(*else*),
			Flatten@{Quiet[Check[CURLLink`CURLGetProxies[URIJoin[Flatten@{URISplit["http://" <> url]}]], {}]]}
		]
	]
	
getSystemProxies[url_String, _] := {}

setProxies[handle_CURLLink`CURLHandle] := CURLLink`CURLSetProxies[handle, StringJoin@Riffle[handle["Proxies"], "\n"]] 
(****************************************************************************)
setOutput[easyhandles_List,"String"]:=Map[setOutput[#,"String"]&,easyhandles]
setOutput[handle_CURLLink`CURLHandle, "String"] := 
	(
		If[!(handle["FTP"]===True),
			CURLLink`CURLOption[handle, "CURLOPT_HEADERFUNCTION", "WRITE_HEADER"];
			CURLLink`CURLOption[handle, "CURLOPT_WRITEHEADER", "MemoryPointer"];
		];
		CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "WRITE_MEMORY"];
		CURLLink`CURLOption[handle, "CURLOPT_WRITEDATA", "MemoryPointer"];
	)
	
setOutput[handle_CURLLink`CURLHandle, "File", fileName_String, format:(True|False)] := 
	(
		If[!(handle["FTP"]===True),
			CURLLink`CURLOption[handle, "CURLOPT_HEADERFUNCTION", "WRITE_HEADER"];
			CURLLink`CURLOption[handle, "CURLOPT_WRITEHEADER", "MemoryPointer"];
		];
		
		CURLLink`CURLFileInfo[handle, fileName, format];
		CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "WRITE_FILE"];
		CURLLink`CURLOption[handle, "CURLOPT_WRITEDATA", "FilePointer"];
	)
	
setOutput[handle_CURLLink`CURLHandle, "WriteFunction", func_String] :=
	(
		If[!(handle["FTP"]===True),
			CURLLink`CURLOption[handle, "CURLOPT_HEADERFUNCTION", "WRITE_HEADER"];
			CURLLink`CURLOption[handle, "CURLOPT_WRITEHEADER", "MemoryPointer"];	
		];
		CURLLink`CURLWriteInfo[handle, func];
		CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "WRITE_USER"];
	)

(****************************************************************************)
saveCredentials[assoc_]:=Module[
	{credhandle,uname,pwd,url,domain,proxyQ},
	uname = Lookup[assoc,"Username",""];
	pwd = Lookup[assoc,"Password",""];
	url = Lookup[assoc,"URL",""];
	If[uname =!= "" || pwd =!= "",
		credhandle = CURLLink`CURLCredentialHandleCreate[];
		CURLLink`CURLFillCredentialHandle[credhandle,<|"Username"->uname,"Password"->pwd|>];
		proxyQ = MatchQ[Lookup[assoc,"Proxy",None],_String];
		If[Not[proxyQ],domain = getDomain[url],domain=Lookup[assoc,"Proxy",None]];
		AssociateTo[$HTTPCredentialStorage,domain->credhandle];
	]
]

clearCredentials[url_]:=
Module[
	{domain},
	domain = getDomain[url];
	If[domain=!="",		 
		$HTTPCredentialStorage = KeyDrop[$HTTPCredentialStorage,domain];
	  ]
]

getDomain[url_]:= getDomain[url] = Quiet[HostLookup[url,"FullName"],{HostLookup::nohost}]

setStandardOptions[handle_CURLLink`CURLHandle, iurl_String, opts:OptionsPattern[]] := 
	Module[{prxyuname,prxypwd,uname,pwd,multipartElements,parameters,
			url=iurl,finalURL, method = ToUpperCase[OptionValue["Method"]], 
			baseURL,optionValueCookies,
			maxUploadSpeed,maxDownloadSpeed,
			body
			}, 
		finalURL = url;
		
		body = OptionValue["Body"];
		
		optionValueCookies=OptionValue["Cookies"];
		
		If[OptionValue["UserAgent"] === Automatic,
			CURLLink`CURLOption[handle, "CURLOPT_USERAGENT", "Wolfram HTTPClient " <> ToString[$VersionNumber]],
			CURLLink`CURLOption[handle, "CURLOPT_USERAGENT", OptionValue["UserAgent"]];
		];
		multipartElements=OptionValue["MultipartElements"];
		parameters=OptionValue["Parameters"];
		
		(*add to parameters to multipart-elements*)
		If[ MatchQ[parameters, {__}] && MatchQ[ multipartElements, {__}],
			multipartElements=appendParametersToMultipartElements[parameters,multipartElements];
			parameters={};
		];
		
		CURLLink`CURLOption[handle, "CURLOPT_SSL_VERIFYPEER", OptionValue["VerifyPeer"]]; 
		CURLLink`CURLOption[handle, "CURLOPT_FOLLOWLOCATION", OptionValue["FollowRedirects"]];
		CURLLink`CURLOption[handle, "CURLOPT_POSTREDIR", CURLInfo`Private`$CURLPostRedir]; 
		CURLLink`CURLOption[handle, "CURLOPT_TIMEOUT_MS", Round[1000*N[OptionValue["ReadTimeout"]]]];
		CURLLink`CURLOption[handle, "CURLOPT_CONNECTTIMEOUT_MS", Round[1000*N[OptionValue["ConnectTimeout"]]]];
		
		maxDownloadSpeed = getSpeed[OptionValue[ConnectionSettings],"MaxDownloadSpeed"];
		CURLLink`CURLOption[handle, "CURLOPT_MAX_RECV_SPEED_LARGE", maxDownloadSpeed];
		
		maxUploadSpeed = getSpeed[OptionValue[ConnectionSettings],"MaxUploadSpeed"];
		CURLLink`CURLOption[handle, "CURLOPT_MAX_SEND_SPEED_LARGE", maxUploadSpeed];
				
		Which[
			method === "GET",
			CURLLink`CURLOption[handle, "CURLOPT_CUSTOMREQUEST", "GET"]
			,
			method === "POST",
			CURLLink`CURLOption[handle, "CURLOPT_POST", True];
			,
			method === "HEAD",
			CURLLink`CURLOption[handle, "CURLOPT_NOBODY", True];
			,
			StringQ[method],
			CURLLink`CURLOption[handle, "CURLOPT_CUSTOMREQUEST", method];
			,
			True,
			Null;(*libcurl will choose the appropriate method*)	
		];
		uname = OptionValue["Username"]/.{None->""};
		pwd = OptionValue["Password"]/.{None->""};
		prxyuname =OptionValue["ProxyUsername"]/.{None->""};
		prxypwd = OptionValue["ProxyPassword"]/.{None->""};
		If[!MatchQ[uname,None|""],
			CURLLink`CURLOption[handle, "CURLOPT_USERNAME", uname]
		];
		If[!MatchQ[pwd,None|""],
			CURLLink`CURLOption[handle, "CURLOPT_PASSWORD", pwd]
		];
		If[!MatchQ[prxyuname,None|""],
			CURLLink`CURLOption[handle, "CURLOPT_PROXYUSERNAME", prxyuname];
		];
		If[!MatchQ[prxypwd,None|""],
			CURLLink`CURLOption[handle, "CURLOPT_PROXYPASSWORD", prxypwd];
		];
		If[!MatchQ[uname,None|""] || !MatchQ[pwd,None|""],
			saveCredentials[<|"URL"->url,"Username"->uname,"Password"->pwd|>]
		];
		If[!MatchQ[prxyuname,None|""] || !MatchQ[prxypwd,None|""],
			saveCredentials[<|"Proxy"->url,"Username"->prxyuname,"Password"->prxypwd|>]
		];
		
		Switch[optionValueCookies,
			Automatic, CURLLink`CURLAutoCookies[handle], 
			_, addCookies[handle, optionValueCookies]
		];
		If[OptionValue["Headers"] =!= {},
			addHeaders[handle, Join[OptionValue["Headers"],{"Connection"->"keep-alive"}]]
			,
			addHeaders[handle,{"Connection"->"keep-alive"}]
		];


		
		
		If[MatchQ[parameters, {__}],
			If[method === "GET",
				finalURL  = url <> "?" <> buildData[handle, parameters, method],
				If[MemberQ[{"POST","PUT","DELETE"},method],
					CURLLink`CURLOption[handle, "CURLOPT_COPYPOSTFIELDS", buildData[handle, parameters, method]]
				]
			]
		];
		If[MemberQ[{"POST","PUT","DELETE"},method],
			(* If the parameters are set, then don't set the body. *)
			If[StringQ[body] && (parameters === {}) && (multipartElements === {}),
				CURLLink`CURLOption[handle, "CURLOPT_POSTFIELDSIZE", StringLength[body]]; 
				CURLLink`CURLOption[handle, "CURLOPT_COPYPOSTFIELDS", body];
			];
			
			If[ByteArrayQ[body] && (parameters === {}) && (multipartElements === {}),
				CURLLink`CURLOption[handle, "CURLOPT_POSTFIELDSIZE", Length[body]]; 
				CURLLink`CURLOption[handle, "CURLOPT_COPYPOSTFIELDS", body];
			]
		];
		CURLLink`CURLCredentialsProvider[handle, ToString[OptionValue["CredentialsProvider"]]];

		(*Handles the old List cases of Multipart Requests*)
         If[MatchQ[multipartElements, {{_String, _String, {__Integer}}..}],
            CURLLink`CURLForm[handle,
                    #[[1]],
                    #[[2]],
                     "",
                    #[[3]],
                     Length[#[[3]]],
                     ""
                     ] & /@ multipartElements
            ];
         (*Handles a List of Rules of MutipartData*)
         Which[MatchQ[multipartElements, {Rule[{_String, _String}, _String]..}],
               CURLForm[handle,
                    #[[1]][[1]],
                    #[[1]][[2]],
                    "",
                    #[[2]],
                        Length[#[[2]]],
                        ""
                        ] & /@ ((Rule[#[[1]],ToCharacterCode[#[[2]]]])& /@ multipartElements)
               ,
               MatchQ[multipartElements, {Rule[{_String, _String|Automatic,___String}, {__Integer}|File[_String]|_String]..}],
               CURLForm[handle,
                        #[[1]][[1]],
                        #[[1]][[2]],
                        If[Length[#[[1]]]===3,#[[1]][[3]],""],
                        If[StringQ[#[[2]]],ToCharacterCode[#[[2]]],#[[2]]],
                        If[StringQ[#[[2]]],Length[ToCharacterCode[#[[2]]]],Length[#[[2]]]],
                        ""
                        ] & /@multipartElements
               ,
               MatchQ[multipartElements, {Rule[{_String, _String|Automatic}, File[_String]]..}],
               CURLLink`CURLForm[handle,
                        #[[1]][[1]],
                        #[[1]][[2]],
                        If[Length[#[[1]]]===3,#[[1]][[3]],""],
                        #[[2]],
                        Length[#[[2]]],
                        ""
                        ] & /@multipartElements
               
               ];

		(* If the Parmeters are set then, we don't want to set the body. *)
		If[MatchQ[body, {__Integer}	]&& ( parameters ==={}) && (multipartElements === {}) ,
			CURLLink`CURLOption[handle, "CURLOPT_POSTFIELDSIZE", Length[body]];
			If[MemberQ[{"POST","PUT","DELETE"},method],
				CURLLink`CURLOption[handle, "CURLOPT_COPYPOSTFIELDS", body]
			]
		];
		handle["URL"] = finalURL;
		CURLLink`CURLSetURL[handle, finalURL];
		CURLLink`CURLOption[handle, "CURLOPT_URL", finalURL];	
		
		baseURL = If[StringMatchQ[url, {"http://*", "https://*", "ftp://*", "ftps://*"}], 
			URIJoin[Flatten@{URISplit[url]}]
		(*else*), 
			URIJoin[Flatten@{URISplit["http://" <> url]}]
		];				
		handle["BaseURL"] = baseURL;
		CURLLink`CURLSetBaseURL[handle, baseURL];
	]
(*helper function: join  parameters to multipart elements *)
(*Since the syntax for parameters option does not allow for
 "content/type" we hard-code "text/plain" this is in accordance with rfc 2388.
 http://www.ietf.org/rfc/rfc2388.txt
 *)
appendParametersToMultipartElements[parameters_,me_]:=
Module[
	{keys,values,formattedParameters},
	keys=Keys[parameters];
	(*the Values[parameters] must be converted to bytes automatically
	  because the user gives these parameters as strings. Encoding for
	  single argument ToCharacterCode is Unicode, and we cannot use that.
	  
	  Hard-code UTF-8 as the character encoding. 
	  Reason:
	  "The first 128 characters of Unicode, 
	  which correspond one-to-one with ASCII, are encoded using a single octet 
	  with the same binary value as ASCII, 
	  making valid ASCII text valid UTF-8-encoded Unicode as well."
	  https://en.wikipedia.org/wiki/UTF-8
	 *)
	values=If[Head[#]===String,ToCharacterCode[#,"UTF-8"],#]&/@Values[parameters];
	(*check multipart-element pattern and format parameters accordingly*)
	Which[
		MatchQ[me,{{_String, _String, {__Integer}}..}],
		formattedParameters=MapThread[{#1,"text/plain; charset=\"utf-8\"",#2}&,{keys,values}]
		,
		MatchQ[me,{Rule[{_String, _String|Automatic,___String}, {__Integer}|_File]..}],
		formattedParameters=MapThread[Rule[{#1,"text/plain; charset=\"utf-8\""},#2]&,{keys,values}];
		];
	(*join multipart elements with parameters*)
	Return[me~Join~formattedParameters];
	]

getSpeed[settings_,key_]:=
Module[
	{speed},
	speed = Lookup[settings,key,Automatic];
	Which[
		(*Use HoldPattern to avoid loading QuantityUnits.*)
		MatchQ[speed,Infinity|Automatic|HoldPattern[Quantity][Infinity,_]],
		speed = 0
		,
		QuantityQ[speed],
		speed = UnitConvert[speed,"Bytes"/"Seconds"];
		speed = Max[QuantityMagnitude[speed]];
		,
		True,
		speed
		
	];
	
	Ceiling[N[speed]]
			
]

(****************************************************************************)
(* helper functions for HTTP streams *)
streamInit[url_String, opts_List] :=
	Module[{stdOpts, error, handle},
		Quiet[
			stdOpts = FilterRules[Flatten[{opts, FilterRules[Options[URLFetch], Except[opts]]}], Except[BinaryFormat]];
			error = Catch[
				handle = commonInit[<|"URL"->url, "Function"->URLFetch,"Format"->"Stream", "Options"->stdOpts|>];
				If[handle === $Failed,
					Return[$Failed]
				];
				
				If[TrueQ[$proxyCache],
					CURLLink`CURLProxyCache[handle];
				];
		
			,CURLLink`Utilities`Exception]
		];
		handle["OPTIONS"] = stdOpts;
		
		If[error === $Failed,
			$Failed
		(*else*),
			First[handle]
		]
	]
	
Options[streamCookies] = $StandardOptions
streamCookies[id_Integer] :=
	streamCookies[id, Sequence@@CURLLink`CURLHandle[id]["OPTIONS"]]

streamCookies[id_Integer, opts:OptionsPattern[]] :=
	Module[{error},
		Quiet[
			error = Catch[
				If[OptionValue["StoreCookies"] && OptionValue["Cookies"] =!= Automatic,
					storeCookies[HTTPData[CURLLink`CURLHandle[id], "Cookies"]]
				] 
			,CURLLink`Utilities`Exception]
		];
		
		If[error === $Failed,
			$Failed,
			True
		]
	]
	
streamStore[id_Integer] := 
	Module[{wellFormedURL, handle},
		handle = CURLLink`CURLHandle[id];
		wellFormedURL = If[StringMatchQ[handle["URL"], {"http://*", "https://*", "ftp://*", "ftps://*"}], 
			URIJoin[Flatten@{URISplit[handle["URL"]]}]
		(*else*), 
			URIJoin[Flatten@{URISplit["http://" <> handle["URL"]]}]
		];
		sessionStore[wellFormedURL] := False
	]
(****************************************************************************)
sessionStore[_] := False;

credWrapper[handle_CURLLink`CURLHandle, url_String, func_] :=
	credWrapper[First[handle], url, func]
	
credWrapper[id_Integer, url_String, func_] :=
	Module[{credProvider, wellFormedURL, defaultQ, handle = CURLLink`CURLHandle[id], res,useWPAD,credHandle=CURLLink`CURLCredentialHandleCreate[]},
		useWPAD=If[Lookup[PacletManager`$InternetProxyRules,"UseWPAD",False],1,0];
		defaultQ = func === Automatic; 
		credProvider = If[defaultQ,passwordDialog, func];
		wellFormedURL = If[StringMatchQ[url, {"http://*", "https://*", "ftp://*", "ftps://*"}], 
			URIJoin[Flatten@{URISplit[url]}]
		(*else*), 
			URIJoin[Flatten@{URISplit["http://" <> url]}]
		];
		
		res = credProvider[url];
		Which[
			res === $Canceled,False,
			MatchQ[res, List[_String, _String]], 
				CURLLink`CURLOption[handle, "CURLOPT_USERNAME", First[res]];
				CURLLink`CURLOption[handle, "CURLOPT_PASSWORD", Last[res]];	
				saveCredentials[<|"URL"->url,"Username"->First[res],"Password"->Last[res]|>];
				True,
				True,False	
		]
	]
	
$proxyCache = False;

proxyCredentials[handle_CURLLink`CURLHandle, url_String,proxy_] :=	
	Module[{result},
		If[$proxyCache === True,
			Return[True];	
		];
		
		result = proxyDialog[url];
		Which[
			result === $Canceled, False,
			MatchQ[result, List[_String, _String]],
				CURLLink`CURLOption[handle, "CURLOPT_PROXYUSERNAME", First[result]];
				CURLLink`CURLOption[handle, "CURLOPT_PROXYPASSWORD", Last[result]];
				saveCredentials[<|"Proxy"->proxy,"Username"->First[result],"Password"->Last[result]|>];
				$proxyCache = True;
				True
		]
	]
	
(* Old default Wolfram System password dialog *)
If[!ValueQ[$allowDialogs], $allowDialogs = True]
hasFrontEnd[] := ToString[Head[$FrontEnd]] === "FrontEndObject"
$pwdDlgResult;
$legacyPwdDlg := ($VersionNumber < 11.3) (* AuthenticationDialog is new in 11.3 *)

passwordDialogStandalone[prompt1_, prompt2_, prompt3_] /; $legacyPwdDlg :=
(
	Print[prompt1];
	Print[prompt2];
	Print[prompt3];
	{InputString["username: "], InputString["password (will echo as cleartext): "]}
)

passwordDialogStandalone[prompt1_, prompt2_, prompt3_] :=
	AuthenticationDialog[
		"UsernamePassword",
		($pwdDlgResult = If[AssociationQ[#], {#Username, #Password}, $Canceled])&,
		AppearanceRules -> Association[
			"Description" -> Column[{prompt1, prompt2, prompt3}]
		]
	]

passwordDialogFE[title_, prompt1_, prompt2_, prompt3_] /; $legacyPwdDlg :=
	Module[{cells, uname = "", pwd = "", createDialogResult},
		cells = {
			TextCell[prompt1, NotebookDefault, "DialogStyle", "ControlStyle"],
			TextCell[prompt2, NotebookDefault, "DialogStyle", "ControlStyle"],
			ExpressionCell[Grid[{ {TextCell["Username:  "], InputField[Dynamic[uname], String, ContinuousAction -> True, 
         		ImageSize -> 200, BoxID -> "UserNameField"]}, {TextCell["Password:  "], 
					InputField[Dynamic[pwd], String, ContinuousAction -> True, 
						ImageSize -> 200, FieldMasked -> True]}}], "DialogStyle", "ControlStyle"],
				TextCell[prompt3, NotebookDefault, "DialogStyle", "ControlStyle"],
                
				ExpressionCell[ Row[{DefaultButton[$pwdDlgResult = {uname, pwd}; 
					DialogReturn[], ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]]], Spacer[{2.5`, 42, 16}],
				CancelButton[$pwdDlgResult = $Canceled; DialogReturn[], 
					ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]]]}], TextAlignment -> Right] };
			createDialogResult = DialogInput[DialogNotebook[cells], 
				WindowTitle -> title, WindowSize -> {400, FitAll}, Evaluator -> CurrentValue["Evaluator"], 
				LineIndent -> 0, PrivateFontOptions -> {"OperatorSubstitution" -> False} ];
			If[createDialogResult === $Failed,
				Null,
			(* else *)
				MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[ FE`BoxReference[createDialogResult, {{"UserNameField"}}, 
					FE`BoxOffset -> {FE`BoxChild[1]}]]];
				$pwdDlgResult
			]
	]
	
passwordDialogFE[title_, prompt1_, prompt2_, prompt3_] :=
	AuthenticationDialog[
		"UsernamePassword",
		($pwdDlgResult = If[AssociationQ[#], {#Username, #Password}, $Canceled])&,
		AppearanceRules -> Association[
			"Description" -> Style[Column[{prompt1, prompt2, prompt3, ""}, Spacings -> 1], "FormDescription"]
		],
		WindowTitle -> title,
		Evaluator -> CurrentValue["Evaluator"]
	]
	
coreDialog[url_String, prompt2_String] :=
	Module[{title, prompt1, prompt3},
	    title = "Authentication Required";
        Clear[$pwdDlgResult];
        Which[
            !TrueQ[$allowDialogs],
                Null,
            hasFrontEnd[],
                (* Use FE dialog box *)
                prompt1 = Row[{"You are attempting to read from the domain:\n", Hyperlink[getDomain[url], BaseStyle -> "ControlStyle"]}];
                prompt3 = "(These values are kept for this session only.)";
                passwordDialogFE[title, prompt1, prompt2, prompt3],
            True,
                prompt1 = "You are attempting to read from the URL:\n" <> url;
                prompt3 = "(These values are kept for this session only.)";
                passwordDialogStandalone[prompt1, prompt2, prompt3]
        ]
	]
	
passwordDialog[url_String] := coreDialog[url, "The server is requesting authentication."]
proxyDialog[url_String] := coreDialog[url, "The proxy server is requesting authentication."]


(****************************************************************************)
$AsyncEnum = {
	"Progress" -> 0,
	"Transfer" -> 1	
}

callBackWrapper[obj_, "headers", data_] := {obj, "headers", Cases[StringSplit[StringSplit[FromCharacterCode[First[data]], "\r\n"], ": ", 2], {_, _}]}
callBackWrapper[obj_, "cookies", data_] := 
	{obj, "cookies",      	
		Cases[
			{	"Domain"-> First[#], 
				If[#[[2]] === "FALSE",
					"MachineAccess"-> #[[2]]
				],
				"Path"->#[[3]], 
				"Secure"->#[[4]], 
				"Expires"->DateString[ToExpression[#[[5]]] + AbsoluteTime[{1970, 1, 1, 0, 0, 0}]], 
				"Name"->#[[6]], 
				"Value"->Last[#]
			}, 
			Except[Null]] & /@ StringSplit[StringSplit[FromCharacterCode[First[data]], "\n"], "\t"]
	}
	
callBackWrapper[obj_, "credentials", data_] := 
	Module[{error, credProvider, handleID, url, output},
		Catch[error,
			handleID = data[[1]];
			url = data[[2]];
			credProvider = data[[3]];	
			CURLLink`CURLReset[CURLLink`CURLHandle[handleID]];
			output = ToExpression[credProvider][handleID, url];
			If[MatchQ[output, {_String, _String}],
				CURLLink`CURLOption[CURLLink`CURLHandle[handleID], "CURLOPT_USERNAME", output[[1]]];
				CURLLink`CURLOption[CURLLink`CURLHandle[handleID], "CURLOPT_PASSWORD", output[[2]]];
			];
			CURLLink`CURLSetCheckQ[CURLLink`CURLHandle[handleID], True];
		,CURLLink`Utilities`Exception];
		
		If[error === $Failed,
			{obj, "credentials", {False}}
		(*else*),
			{obj, "credentials", {True}} 
		]
	]
callBackWrapper[obj_, "progress", data_] := {obj, "progress", Flatten[data]}	
callBackWrapper[obj_, name_, data_] := {obj, name, data}



urlSubmitCallbackWrapper[obj_,"headers",data_,opts:OptionsPattern[URLFetchAsynchronous]]:=
Module[{headers},
	headers=impHeaders[Last[StringSplit[FromCharacterCode[First[data]],"\r\n\r\n"]]];
	{obj,"HeadersReceived",headers}
]

urlSubmitCallbackWrapper[obj_,"cookies",data_,opts:OptionsPattern[URLFetchAsynchronous]]:=
Module[
	{cookiesreceived,cookiefunction},
	cookiesreceived=Complement[CURLLink`Cookies`Private`cookiesToAssociation[FromCharacterCode[First[data]]],CURLLink`Cookies`Private`cookiesToAssociation[$LastKnownCookies]];
	cookiefunction=OptionValue[CookieFunction];
	Which[
		cookiefunction===Automatic,
		SetCookies[cookiesreceived];
		,
		cookiefunction===None,
		ClearCookies[cookiesreceived];
		,
		True,
		Map[cookiefunction,cookiesreceived];
	
	];
	{obj,"CookiesReceived",cookiesreceived}
]

urlSubmitCallbackWrapper[obj_,"data",data_,opts:OptionsPattern[URLFetchAsynchronous]]:=
	If[OptionValue["Transfer"]==="Chunks",{obj,"chunk",data},{obj,"data",data}]
	
urlSubmitCallbackWrapper[obj_, event_, data_,opts:OptionsPattern[URLFetchAsynchronous]] := {obj, event, data}


(****************************************************************************)
Options[httpFetchAsynchronous]=Options[URLFetchAsynchronous]

httpFetchAsynchronous[urlExp_, func:Except[_Rule|_RuleDelayed|_String], options___?OptionQ] /; deprecatedOptionQ[options] := 
	httpFetchAsynchronous[urlExp, func, deprecatedOptionFix[URLFetchAsynchronous, options]] 

httpFetchAsynchronous[urlExp_/;isURL[urlExp,URLFetchAsynchronous], func:Except[_Rule|_RuleDelayed|_String], opts:OptionsPattern[URLFetchAsynchronous]] /; initializeQ[] := 
	Module[{handle, stdOpts, error, output,url},
		url=getURL[urlExp];
		If[OptionValue["DisplayProxyDialog"],
			URLFetch[url,{"StatusCode","ContentData"},"Method"->"HEAD"];
		];
		$LastKnownCookies=CURLLink`Cookies`GetKnownCookies[];
		stdOpts = Flatten[{opts, FilterRules[Options[URLFetchAsynchronous], Except[{opts}]]}];	
		error = Catch[
			(* handle is freed in c code *)
			handle = commonInit[<|"URL"->url,"Function"-> URLFetchAsynchronous,"Options"->stdOpts|>];
			If[handle === $Failed,
				Return[$Failed]
			];
		
			CURLLink`CURLSetAsync[handle, True];
		
			If[OptionValue["StoreCookies"] && (OptionValue["Cookies"] =!= Automatic),
				CURLLink`CURLAsyncCookies[handle, True]
			]; 
			
			(* Set which async events will be raised. *)
			
		,CURLLink`Utilities`Exception];
		output = If[error === $Failed,
			$Failed
		(*else*),
			Internal`CreateAsynchronousTask[
				CURLLink`Private`curlMultiAdd, {Extract[$WolframHTTPMultiHandleAsync,1],Extract[handle,1]}, 
				CallbackFunction[func,OptionValue["Events"],#1,#2,#3,opts]&,
				"TaskDetail"->url, 
				"UserData"->OptionValue["UserData"]]
		];
		unsetCURLHandle[handle];
		output
	]
	
CallbackFunction[userfunc_,events_,obj_,event_,data_,opts:OptionsPattern[URLFetchAsynchronous]]:=Module[{},
Which[
	events===Full,
	userfunc[Sequence@@urlSubmitCallbackWrapper[obj,event,data,opts]],
	True,
	userfunc[Sequence@@callBackWrapper[obj,event,data]]
	]
]	

(****************************************************************************)
Options[httpSaveAsynchronous]=Options[URLSaveAsynchronous]
httpSaveAsynchronous[urlExp_, func:Except[_Rule|_RuleDelayed|_String], options___?OptionQ] := 
	httpSaveAsynchronous[urlExp, Automatic, options]

httpSaveAsynchronous[urlExp_, Automatic|None|Null, rest___] := 
	httpSaveAsynchronous[urlExp, FileNameJoin[{$TemporaryDirectory, CreateUUID[] <> ".tmp"}], rest]

httpSaveAsynchronous[urlExp_, file_, func:Except[_Rule|_RuleDelayed|_String], options___?OptionQ] /; deprecatedOptionQ[options] := 
	httpSaveAsynchronous[urlExp, file, func, deprecatedOptionFix[URLSaveAsynchronous, options]] 

httpSaveAsynchronous[urlExp_/;isURL[urlExp,URLSaveAsynchronous], fileExp_/;isFile[fileExp,URLSaveAsynchronous], func:Except[_Rule|_RuleDelayed|_String], opts:OptionsPattern[URLSaveAsynchronous]] /; initializeQ[] := 
	Module[{handle, stdOpts,file, error, output,url},
		url=getURL[urlExp];
		file=getFile[fileExp];
		$LastKnownCookies=CURLLink`Cookies`GetKnownCookies[];
		If[OptionValue["DisplayProxyDialog"],
			URLFetch[url,{"StatusCode","ContentData"},"Method"->"HEAD"];
		];
		stdOpts = Flatten[{opts, FilterRules[Options[URLSaveAsynchronous], Except[{opts}]]}];
		error = Catch[
			(* handle is freed in c code *)
			handle = commonInit[<|"URL"->url,"Function"-> URLSaveAsynchronous,"FileName"->ExpandFileName[file],"Format"-> OptionValue[BinaryFormat],"Options"-> stdOpts|>];
			If[handle === $Failed,
				Return[$Failed]
			];
		
			CURLLink`CURLSetAsync[handle, True];
		
			If[OptionValue["StoreCookies"] && (OptionValue["Cookies"] =!= Automatic),
				CURLLink`CURLAsyncCookies[handle, True]
			]; 
		
			
		,CURLLink`Utilities`Exception];
		output = If[error === $Failed,
			$Failed
		(*else*),
			Internal`CreateAsynchronousTask[
				CURLLink`Private`curlMultiAdd, {Extract[$WolframHTTPMultiHandleAsync,1],Extract[handle,1]}, 
				CallbackFunction[func,OptionValue["Events"],#1,#2,#3,opts]&,
				"TaskDetail"->url, 
				"UserData"->OptionValue["UserData"]]
		];
		unsetCURLHandle[handle];
		output
	]
(*Helper Functions*)
getNewHandle[url_,func_,file_,opts:OptionsPattern[URLSave]]:=Module[
	{handle},
	(*initialize multi handle*)
				
	(*Throw an error, in case handle cannot be initialized*)
	If[handle === $Failed,Return[$Failed]];
			
	(*output should be held in memory*)
	If[func===URLFetch,handle=commonInit[<|"URL"->url,"Function"-> func,"Options"-> opts|>]];
	(*output should be held in a file*)
	If[func===URLSave,handle=commonInit[<|"URL"->url,"Function"->  func, "FileName"->ExpandFileName[file],"Format"-> OptionValue[BinaryFormat],"Options"->opts|>]];
	handle
]
getResponseCodes[handle_]:=Module[
	{stream,statusCode,connectCode},
	(*get the stream*)					
	stream=HTTPData[handle,"Stream"];
	(*
	read one byte, this is necessary hack
	to get proper status code
	*)
	
	Read[stream,Byte];

	(*reset stream position to zero*)
	SetStreamPosition[stream,0];

	(*get status code*)
	statusCode=CURLLink`CURLGetInfo[handle,"CURLINFO_RESPONSE_CODE"];
	connectCode=CURLLink`CURLGetInfo[handle,"CURLINFO_HTTP_CONNECTCODE"];
	{stream,statusCode,connectCode,handle["Return"]}
					
	]
retryIfSSLError[ihandle_,url_,func_,file_,opts___]:=
Module[{sslConnectError=35,handle=ihandle,SSLVersion=5},
	While[MatchQ[handle["Return"],sslConnectError|56],
			CURLLink`CURLHandleUnload[handle];
			handle=getNewHandle[url,func,file,opts];
			(*CURLLink`CURLOption[handle, "CURLOPT_SSL_OPTIONS", noRevoke];*)
			CURLLink`CURLOption[handle, "CURLOPT_SSLVERSION",SSLVersion];
			CURLLink`CURLMultiHandleAdd[getGlobalMultiHandle[func],{handle}];
			handle["Return"]=CURLLink`CURLWait[handle];
			If[SSLVersion<=2,Break[]];
			SSLVersion--;
	];
		handle	

]
httpFetch[args___]:=Throw[False,CURLLink`Utilities`Exception]
httpSave[args___]:=Throw[False,CURLLink`Utilities`Exception]
httpFetchAsynchronous[args___]:=Throw[False,CURLLink`Utilities`Exception]
httpSaveAsynchronous[args___]:=Throw[False,CURLLink`Utilities`Exception]
(****************************************************************************)
If[ValueQ[initialize[]],initialize[]=.]
initialize[] := initialize[] = 
	Catch[	
		CURLLink`CURLInitialize[];
		CURLLink`CURLSetCert[$CACERT];
		System`$CookieStore;
		CURLLink`Cookies`LoadPersistentCookies[System`$CookieStore];
	,CURLLink`Utilities`Exception] =!= $Failed

initializeQ[] := initialize[];
	
(****************************************************************************)
(* List of all possible output types *)
$URLFetchElements = {
	"Content",
	"DebugContent",
	"ContentData",
	"DebugContentData",
	"Headers",
	"Cookies",
	"StatusCode",
	"Stream",
	"CookiesReceived",
	"HeadersReceived",
	"BodyByteArray",
	"DebugBodyByteArray"
}

$FTPFetchElements = {
	"Content",
	"ContentData",
	"StatusCode"
}

$URLSaveElements = {
	"Headers",
	"Cookies",
	"StatusCode",
	"CookiesReceived",
	"HeadersReceived"	
}

$FTPSaveElements = {
	"StatusCode"
};

(****************************************************************************)
successQ[obj_CURLLink`CURLHandle] := (obj["Return"] === 0 ||(MemberQ[$httpNonFailureReturnCodes,CURLLink`CURLGetInfo[obj,"CURLINFO_RESPONSE_CODE"]]))

(****************************************************************************)
End[] (* End Private Context *)
EndPackage[]
