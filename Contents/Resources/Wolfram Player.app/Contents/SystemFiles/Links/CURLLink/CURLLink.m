(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["CURLLink`",{"PacletManager`"}]

(* Exported symbols added here with SymbolName::usage *)  
CURLWait
CURLMultiHandleInit
CURLCreateRunThread
CURLMultiHandleAdd
CURLMultiHandleRemove
CURLHTTPConnectCodes
CURLMultiHandleCreate
CURLCredentialHandleCreate
CURLFillCredentialHandle
CURLAssociateCredentialHandle
CURLGetCredentials
CURLCredentialHandle
CURLInitialize::usage = "CURLInitialize[size] initializes internal hashtables of size size."
CURLStatusCodes
CURLErrorData
CURLGetClientCertificate
CURLHandleLoad::usage = "CURLHandleLoad	create a new CURLHandle."
CURLHandleDuplicate
CURLReset::usage = "CURLReset reset a CURLHandle."
CURLHandle::usage = "CURLHandle	"
CURLMultiHandle::usage = "Adds an easy handle to multi handle"
CURLMultiPerform::usage = "Adds an easy handle to multi handle"
CURLSetURL::usage = "CURLSetURL[hdl, url] set the actual URL used for the connection."
CURLSetBaseURL::usage = "CURLSetBaseURL[hdl, url] set the base URL used for the connection."
CURLCredentialsProvider::usage = "CURLCredentialsProvider[hdl, url] set the callback function for credentials."
CURLStore::usage = "CURLStore[hdl, session] store session information."
CURLSessionCache::usage = "CURLSessionCache[hdl, url] check cache for session information."
CURLProxyCache::usage = "CURLProxyCache[hdl] use cached proxy information for a handle."
CURLSetCheckQ::usage = "CURLSetCheckQ[hdl, True|False] flag if credentials have been set."
CURLGetInfo::usage = "extract information from a curl handle"
CURLCookies::usage = "CURLCookies[hdl] get a list of cookies used by a handle."
CURLCerts::usage = "CURLCerts[hdl] get certificate information from a handle."
CURLSetCert::usage = "CURLSetCert[cert] store the cerficite file in global memory, for use by the stream handler."
CURLSetProxies::usage = "CURLSetProxies[hdl, proxies] set a list of proxies to be used by streams and async handles."
CURLGetProxies::usage = "CURLGetProxies[url] get valie proxies for OS."
CURLAutoCookies::usage = "CURLAutoCookies[hdl] if called, cookies will be shared/handled automatically."
CURLFileInfo::usage = "CURLFileInfo[hdl, name] store filename to be used to store contents of connection."
CURLUploadFileInfo::usage = "CURLFileInfo[hdl, name] loacal file to be uploaded"
CURLUploadContent::usage = "CURLUploadContent[hdl, name] data to be uploaded"
CURLWriteInfo::usage = "CURLWriteInfo[hdl, name] store function name to be used for user defined write function."
CURLHandleUnload::usage = "CURLHandleUnload[handle]	free a CURLHandle from memory"
CURLHeaderData::usage = "CURLHeaderData[handle] get headers after CURLPerform that is stored in memory (proper CURLOptions must be set)."
CURLStatusCode::usage = "CURLStatusCode[handle] get the status code after the connection has competed."
CURLHTTPConnectCode::usage = "CURLHTTPConnectCode[handle] get the  HTTP proxy response code to a CONNECT request"
CURLRawContentData::usage = "CURLRawContentData[handle] get raw data as an array of bytes."
CURLRawContentDataRawArray::usage = "CURLRawContentData[handle] get raw data as an array of bytes."
CURLDebugRawContentDataRawArray::usage = "CURLDebugRawContentData[handle] get raw data as an array of bytes."
CURLOption::usage = "CURLOption[handle, option, value] set a handle option to value."

CURLLinkInformation 

CURLEscape::usage = "CURLEscape[handle, url] url encodes a string."
CURLUnescape::usage = "CURLUnescape[handle, url] url decodes a string."

CURLPerform::usage = "CURLPerform[handle] perform actions with a CURLHandle."
CURLPerformNoBlock::usage = "CURLPerform[handle] perform actions with a CURLHandle."
CURLForm::usage = "CURLForm[...] set multi-form POST data."
CURLError::usage = "CURLError[cc] get error string for given CURLcode cc."

CURLAddHeader::usage = "CURLAddHeader[handle, header] add a custom header to an CURLHandle."
CURLSetAsync::usage = "CURLSetAsync[handle, value] set if a CURLHandle is to be asynchronous. "
CURLAsyncOption::usage = "CURLAsyncOpiton[handle, opt, val] set async events to be rasied."
CURLAsyncCookies::usage = "CURLAsyncCookies[handle, val] set a flag to inform the async connection to manually store cookies."

CURLLink`$CURLOptions::usage = "$CURLOptions list containing all available options for CURLOption."
CURLLink`CURLOptionQ::usage = "CURLOptionQ[opt] returns True if a valid CURL option."
$WolframHandleIndex;




Begin["`Private`"] (* Begin Private Context *) 

If[ValueQ[$WolframHandleIndex],Nothing,$WolframHandleIndex=0;]
If[ValueQ[initialize[]],Clear[initialize]]
(****************************************************************************)

(* Supported option types that may be sent to LibraryLink functions *)
$CURLOptionTypes = {
	"Integer",
	"String",
	"CallbackPointer",	
	"FilePointer",
	"MemoryPointer",
	"Tensor"
}

(* Enumerate the option types to match the LibraryLink enum values *)
Do[
	opTypes[$CURLOptionTypes[[ii]]] = ii - 1,
	{ii, Length[$CURLOptionTypes]}
]

(* Different types of callback functions that HTTPClient supports *)
$CURLCallbacks = {
	"WRITE_MEMORY",
	"WRITE_FILE",
	"WRITE_USER",
	"READ_MEMORY",
	"READ_FILE",
	"WRITE_HEADER",
	"SHOW_PROGRESS",
	"SEND_CHUNKS",
	"WRITE_MEMORY_DEBUG"
}

(* Enumerate the callback functions to match LibraryLink enum values *)
Do[
	callbackEnum[$CURLCallbacks[[ii]]] = ii - 1;
	curlCallbackQ[$CURLCallbacks[[ii]]] := True,
	{ii, Length[$CURLCallbacks]}
]

curlCallbackQ[_] := False

$handleType = 
<|

"WOLFRAM_MULTI_HANDLE"->0,
"WOLFRAM_EASY_HANDLE"->1

|>
(****************************************************************************)
(* Load the required dynamic libraries *)

$LibraryResourcesPath =
	FileNameJoin[{
		If[TrueQ[Developer`$ProtectedMode] && $VersionNumber < 10.2,
		    (* This branch is a fix for bug 294005. *)
		    FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "CURLLink"}],
		(* else *)
            DirectoryName[System`Private`$InputFileName]
		],
		"LibraryResources",
		$SystemID
	}]

If[FreeQ[$LibraryPath, $LibraryResourcesPath],
	PrependTo[$LibraryPath, $LibraryResourcesPath]
]

newPath={FileNameJoin[Most[FileNameSplit[$InputFileName]]~Join~{"LibraryResources"}~Join~{$SystemID}],FileNameJoin[{$InstallationDirectory,"SystemFiles","Libraries",$SystemID}]};
Block[{$LibraryPath = If[$OperatingSystem === "iOS", $LibraryPath, newPath]},
 		If[ValueQ[$librariesLoadedQ]===False || $librariesLoadedQ===False,
 			Check[
 				(*expr*)
 				$CURLLinkLibrary=FindLibrary["libcurllink"];
 				Which[
					$OperatingSystem === "Windows",
						LibraryLoad[FindLibrary["libssh2"]];
						LibraryLoad[FindLibrary["libcurl"]];
						$librariesLoadedQ=True;,
					True,								
						$librariesLoadedQ=True;
					 ];
				 ,
				(*failure expr*)
				$librariesLoadedQ=False;
				 ]
 			]
		]

(****************************************************************************)
(* Load all the functions required by HTTPClient *)
initialize[] := initialize[] =
(
	If[!StringQ[$CURLLinkLibrary] || !FileExistsQ[$CURLLinkLibrary],
		Throw[$Failed,CURLLink`Utilities`Exception]
	];
	
	If[$VersionNumber < 9,
		Throw[$Failed,CURLLink`Utilities`Exception]
	];
	Check[	
		curlInitialize = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_initialize", {_Integer}, "Void"];
		createWolframEasyHandle = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_createWolframEasyHandle", {_Integer}, _Integer];
		
		createDuplicateWolframEasyHandle = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_createDuplicateWolframEasyHandle", {_Integer,_Integer}, _Integer];
		
		wolframEasyHandleReset = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_resetWolframEasyHandle", {_Integer}, _Integer];
		
		createWolframMultiHandle = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_createWolframMultiHandle", {_Integer}, _Integer];
		
		curlPerform = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_perform", {_Integer,_Integer}, _Integer];
		
		createRunThread = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_createRunThread", {_Integer,_Integer}, _Integer];
		
		curlMultiAdd=LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_multiadd", {_Integer,_Integer}, _Integer];
		
		curlWait = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_wait", {_Integer,_Integer}, _Integer];
		
		curlMultiRemove=LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_multiremove", {_Integer,_Integer}, _Integer];
		curlVersionFeatures = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getVersionFeatures", {}, _Integer];
		curlVersion = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getVersion", {_Integer}, "UTF8String"];
		curlProtocols = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getSupportedProtocols", {_Integer}, "UTF8String"];
		getIdentityPreference = LibraryFunctionLoad[$CURLLinkLibrary,"get_identity_preference",{"UTF8String"},"UTF8String"];
		createCredentials = LibraryFunctionLoad[$CURLLinkLibrary,"curlLink_createWolframCredentialHandle",{_Integer},_Integer];
		setCredentialInfo = LibraryFunctionLoad[$CURLLinkLibrary,"curlLink_setCredentialInfo",{_Integer,"UTF8String","UTF8String","UTF8String"},_Integer];
		getCredentialInfo = LibraryFunctionLoad[$CURLLinkLibrary,"curlLink_getCredentialInfo",{_Integer,_Integer},"UTF8String"];
		associateCredentialHandleToEasyHandle = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_associateCredentials", {_Integer,_Integer}, _Integer];
		getCookies = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getCookies", {_Integer}, "UTF8String"];
		handleUnload = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_cleanup", {_Integer}, "Void"];
		getHeaders = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getHeaders", {_Integer}, {_Integer, 1, "Automatic"}];
		getErrorString = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getErrorString", {_Integer},"UTF8String"];
		getInfoString = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getInfoString", {_Integer,_Integer},"UTF8String"];
		getInfoSList = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getInfoSList", {_Integer,_Integer},"UTF8String"];
		getInfoReal = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getInfoReal", {_Integer,_Integer}, _Real];
		getInfoInteger = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getInfoInteger", {_Integer,_Integer},_Integer];
		curlMultiOptionInteger = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_multiOptionInteger", {_Integer,_Integer, _Integer, _Integer}, _Integer];
		getRawContent = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getRawContent", {_Integer}, {_Integer, 1, "Automatic"}];
		getRawContentRawArray = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getRawContentRawArray", {_Integer},"ByteArray"];
		getDebugRawContentRawArray = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getDebugRawContentRawArray", {_Integer},"ByteArray"];
		curlOptionString = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_optionString", {_Integer, _Integer, "UTF8String"}, _Integer];
		curlOptionByteArray = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_optionByteArray", {_Integer, _Integer,"ByteArray"}, _Integer];
		curlOptionTensor = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_optionTensor", {_Integer, _Integer, {_Integer, 1, "Shared"}}, _Integer];
		curlOptionInteger = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_optionInteger", {_Integer,_Integer, _Integer, _Integer}, _Integer];
		curlEscape = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_escape", {"UTF8String"}, "UTF8String"];
		curlUnescape = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_unescape", {"UTF8String"}, "UTF8String"];
		curlAddHeader = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_addHeader", {_Integer, "UTF8String"}, _Integer];
		curlMultiForm = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_multiPartForm", {_Integer, "UTF8String", "UTF8String","UTF8String","UTF8String", {_Integer, 1, "Shared"}, _Integer, "UTF8String"}, _Integer];
		curlError = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_errorStr", {_Integer}, "UTF8String"];
		curlGetProxies = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_getProxies", {"UTF8String",_Integer}, "UTF8String"];
		curlSetProxies = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setProxies", {_Integer, "UTF8String"}, "Void"];
		curlSetURL = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setURL", {_Integer, "UTF8String"}, "Void"];
		curlSetBaseURL = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setBaseURL", {_Integer, "UTF8String"}, "Void"];
		curlCredentialsProvider = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setCredentialsProvider", {_Integer, "UTF8String"}, "Void"];
		curlSessionStore = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_sessionStore", {_Integer, "UTF8String", "UTF8String", "UTF8String",_Integer}, "Void"];
		curlSessionCache = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_sessionCache", {_Integer, "UTF8String"}, "Void"];
		curlProxyCache = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_proxyCache", {_Integer}, "Void"];
		curlSetCheckQ = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_curlSetCheckQ", {_Integer, True|False}, "Void"];
		curlSetCert = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setCert", {"UTF8String"}, "Void"];
		curlAutoCookies = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_autoCookies", {_Integer}, _Integer];
		curlFileInfo = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setFileInfo", {_Integer, "UTF8String", True|False}, "Void"];
		curlUploadFileInfo = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setUploadFileInfo", {_Integer, "UTF8String"}, "Void"];
		curlUploadContent = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setUploadContent", {_Integer, "UTF8String"}, "Void"];
		curlWriteInfo = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setWriteFunction", {_Integer, "UTF8String"}, "Void"];
		curlSetAsync = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_setAsync", {_Integer,_Integer, True|False}, "Void"];
		curlAsyncCookies = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_asyncCookies", {_Integer, True|False}, "Void"];
		cSuccessQ = LibraryFunctionLoad[$CURLLinkLibrary, "curlLink_successQ", {}, True|False],
		
		Throw[$Failed,CURLLink`Utilities`Exception]
	]
)

initializedQ := Catch[initialize[]] =!= $Failed

successQ := cSuccessQ[]
Developer`RegisterInputStream["HTTPStreamElement", StringMatchQ[#, "HTTPStreamElement"] &,Null];

(****************************************************************************)

(* 
	CURLInitialize[] sets up the LibraryLink data structures needed to use
	HTTPClient.
*)   
CURLInitialize[] := CURLInitialize[]=CURLInitialize[5] (* this should be plenty for the average user (famous last words) *)

CURLInitialize[hashSize_Integer] /; initializedQ :=
	(
		curlInitialize[hashSize];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* Add a curlHandle_t to the HTTPClient hash table *)
CURLHandleLoad[] /; initializedQ :=
	Module[{id},
		id = createWolframEasyHandle[++$WolframHandleIndex];
		If[successQ, CURLHandle[id], Throw[$Failed,CURLLink`Utilities`Exception]]
	]
CURLHandleDuplicate[CURLHandle[idToClone_]] /; initializedQ :=
	Module[{id},
		id = createDuplicateWolframEasyHandle[++$WolframHandleIndex,idToClone];
		If[successQ, CURLHandle[id], Throw[$Failed,CURLLink`Utilities`Exception]]
	]	
CURLHandle /: Format[CURLHandle[id_Integer], StandardForm] :=
	CURLHandle["<" <> ToString[id] <> ">"] 
   
(****************************************************************************)
(****************************************************************************)
(* Add a curlHandle_t to the HTTPClient hash table *)
CURLMultiHandleCreate[] /; initializedQ :=
	Module[{id},
		id = createWolframMultiHandle[++$WolframHandleIndex];
		If[successQ, CURLMultiHandle[id], Throw[$Failed,CURLLink`Utilities`Exception]]
	]
	
CURLMultiHandle /: Format[CURLMultiHandle[id_Integer], StandardForm] :=
	CURLMultiHandle["<" <> ToString[id] <> ">"]

CURLCredentialHandleCreate[] /; initializedQ :=
	Module[{id},
		id = createCredentials[++$WolframHandleIndex];
		If[successQ, CURLCredentialHandle[id], Throw[$Failed,CURLLink`Utilities`Exception]]
	]
CURLFillCredentialHandle[CURLCredentialHandle[id_Integer],assoc_?AssociationQ]:=
Module[
	{uname, pwd, domain},
	uname = Lookup[assoc,"Username",""];
	pwd = Lookup[assoc,"Password",""];
	domain = Lookup[assoc,"Domain",""];
	setCredentialInfo[id,uname, pwd, domain]
]
CURLAssociateCredentialHandle[CURLCredentialHandle[id_Integer],CURLHandle[id2_Integer]]:=
Module[
	{},
	associateCredentialHandleToEasyHandle[id,id2]
]
CURLGetCredentials[CURLCredentialHandle[id_Integer],field_String]:=
Module[
	{},
	Which[
		field==="Username",
		getCredentialInfo[id,0]
		,
		field==="Password",
		getCredentialInfo[id,1]
		,
		field==="Domain",
		getCredentialInfo[id,2]
	]
	
]	
CURLCredentialHandle /: Format[CURLCredentialHandle[id_Integer], StandardForm] :=
	CURLCredentialHandle["<" <> ToString[id] <> ">"] 
	 
   
(****************************************************************************)

(* 
	Remove a curlHandle_t from HTTPClient hash table, and free
	all memory associated with the handle.
*)
CURLHandleUnload[easyhandles_List] /; initializedQ :=
	Map[CURLHandleUnload[#]&,easyhandles]
CURLHandleUnload[CURLHandle[id_Integer]] /; initializedQ :=
	(	
		Quiet[CURLHeaderData[CURLHandle[id]]=.];
		Quiet[CURLRawContentData[CURLHandle[id]]=.];
		Quiet[CURLErrorData[CURLHandle[id]]=.];
		Quiet[CURLRawContentDataRawArray[CURLHandle[id]]=.];
		handleUnload[id];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* 
	Set the final URL used for the connection.
*)
CURLSetURL[CURLHandle[id_Integer], url_String] /; initializedQ :=
	(
		curlSetURL[id, url];	
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* 
	Set the base URL used for the connection.
*)
CURLSetBaseURL[CURLHandle[id_Integer], url_String] /; initializedQ :=
	(
		curlSetBaseURL[id, url];	
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* 
	Set callback function for providing credentials.
*)
CURLCredentialsProvider[CURLHandle[id_Integer], credentials_String] /; initializedQ :=
	(
		curlCredentialsProvider[id, credentials];	
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* 
	Store session information.
*)

CURLStore[CURLHandle[id_Integer], url_String, user_String, pass_String,useWPAD_Integer] /; initializedQ :=
	(
		curlSessionStore[id, url, user, pass,useWPAD];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* 
	Check cache for session	information.
*)
CURLSessionCache[CURLHandle[id_Integer], url_String] /; initializedQ :=
	(
		curlSessionCache[id, url];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* 
	Check cache for proxy information.
*)
CURLProxyCache[CURLHandle[id_Integer]] /; initializedQ :=
	(
		curlProxyCache[id];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* 
	Set credentials flag.
*)
CURLSetCheckQ[CURLHandle[id_Integer], flag:(True|False)] /; initializedQ :=
	(
		curlSetCheckQ[id, flag];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
	
(****************************************************************************)
(* 
	Get the cookies received from a CURLHandl]e, the cookies will
	be returned as a single string with '\n' seperating the cookies.
*)
CURLCookies[CURLHandle[id_Integer]] /; initializedQ :=
	Module[{str},
		str = getCookies[id];
		If[successQ, str, Throw[$Failed,CURLLink`Utilities`Exception]]	
	]
	
(****************************************************************************)
(* 
	Get the contents of the headers received from a CURLHandle,
	note that this will include the HTTP status code as well.
*)
CURLHeaderData[CURLHandle[id_Integer]] /; initializedQ:=
CURLHeaderData[CURLHandle[id]]=
	Module[{str},
		str=FromCharacterCode@getHeaders[id];
		If[successQ, str, Throw[$Failed,CURLLink`Utilities`Exception]]	
	]
	
(****************************************************************************)
(* 
	Reset a CURLHandle to its default state.
*)
CURLReset[CURLHandle[id_Integer]] /; initializedQ :=
	Module[{},
		Quiet[CURLHeaderData[CURLHandle[id]]=.];
		Quiet[CURLRawContentData[CURLHandle[id]]=.];
		Quiet[CURLErrorData[CURLHandle[id]]=.];
		Quiet[CURLRawContentDataRawArray[CURLHandle[id]]=.];
		Quiet[CURLDebugRawContentDataRawArray[CURLHandle[id]]=.];
		wolframEasyHandleReset[id];
		
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]	
	]

(****************************************************************************)
(* Returns the content of CURLHandle as a list of bytes. *)
CURLRawContentData[CURLHandle[id_Integer]] /; initializedQ :=
CURLRawContentData[CURLHandle[id]]=
	Module[{lst},
		lst = getRawContent[id];
		If[successQ, lst, Throw[$Failed,CURLLink`Utilities`Exception]]	
	]

CURLErrorData[CURLHandle[id_Integer]] /; initializedQ :=
CURLErrorData[CURLHandle[id]]=
	Module[{lst},
		lst = getErrorString[id];
		If[successQ, lst, Throw[$Failed,CURLLink`Utilities`Exception]]	
	]

CURLRawContentDataRawArray[CURLHandle[id_Integer]] /; initializedQ :=
CURLRawContentDataRawArray[CURLHandle[id]]=
	Module[{lst},
		lst = getRawContentRawArray[id];
		If[successQ, lst, Throw[$Failed,CURLLink`Utilities`Exception]]	
	]

CURLDebugRawContentDataRawArray[CURLHandle[id_Integer]] /; initializedQ :=
CURLDebugRawContentDataRawArray[CURLHandle[id]]=
	Module[{lst},
		lst = getDebugRawContentRawArray[id];
		If[successQ, lst, Throw[$Failed,CURLLink`Utilities`Exception]]	
	]

CURLGetInfo[CURLHandle[id_Integer], element_String] /; initializedQ :=
Module[{info,value=CURLLink`$CURLInfo[element]["Value"]},
		info=
		Switch[
			CURLLink`$CURLInfo[element]["Type"],
			"String",
			getInfoString[id,value]
			,
			"Real",
			getInfoReal[id,value]
			,
			"Integer",
			getInfoInteger[id,value]
			,
			"SList",
			getInfoSList[id,value]
		];
		If[successQ, info, Throw[$Failed,CURLLink`Utilities`Exception]]
	]
(****************************************************************************)
(* Connect to an HTTP server and gather content/headers *)
Options[CURLPerform]={"Blocking"->True}
CURLPerform[CURLHandle[id_Integer],opts:OptionsPattern[]] /; initializedQ :=
	Module[{cc,type = $handleType["WOLFRAM_EASY_HANDLE"]},
		
		If[OptionValue["Blocking"],cc = curlPerform[id,type],cc = curlPerformnoblock[id]];
		
		If[successQ, cc, Throw[$Failed,CURLLink`Utilities`Exception]]
	]
CURLPerform[CURLMultiHandle[id_Integer],opts:OptionsPattern[]] /; initializedQ :=
	Module[{cc,type = $handleType["WOLFRAM_MULTI_HANDLE"]},
		
		If[OptionValue["Blocking"],cc = curlPerform[id,type],cc = curlPerformnoblock[id]];
		
		If[successQ, cc, Throw[$Failed,CURLLink`Utilities`Exception]]
	]

(******************************************************************************)
(*curl-retry*)
(*
This function is a work around, for a bug in windows schannel see bug#308724 for details.
When TLSv1.2 gives ssl connect error, we set a lower ssl version, and keep trying.
SSLVersions:CURL_SSLVERSION_TLSv1_1 (5),CURL_SSLVERSION_TLSv1_0 (4),CURL_SSLVERSION_SSLv3 (3),CURL_SSLVERSION_SSLv2 (2)
*)
CURLRetry[CURLHandle[id_Integer]] /; initializedQ :=
	Module[
		{cc,SSLVersion=5,noError=0,noRevoke=2(*CURLSSLOPT_NO_REVOKE*)},
		CURLLink`CURLOption[CURLHandle[id], "CURLOPT_SSL_OPTIONS", noRevoke];
		While[SSLVersion >= 2,
		CURLLink`CURLOption[CURLHandle[id], "CURLOPT_SSLVERSION",SSLVersion];
		cc=curlPerform[id];
		If[cc==noError,
			(*then*)
			Break[]
			];
		SSLVersion--;
	];
	cc
	]
(*GetIdentity Preference - OSX only*)
CURLGetClientCertificate[url_String]/; initializedQ:= Module[
	{cert},
	Quiet@Check[cert=getIdentityPreference[url],cert=0];
	cert
]
(******************************************************************************)
(*
	This will get a list of possible proxies valid for a url.
*)
CURLGetProxies[url_String,useWPAD_Integer:0] /; initializedQ :=
	Module[{str},
		str = curlGetProxies[url,useWPAD];
		If[successQ, str, Throw[$Failed,CURLLink`Utilities`Exception]]
	]

(******************************************************************************)
(*
	This will set an internal list of possible proxies, for streams and async handles..
*)
CURLSetProxies[CURLHandle[id_Integer], proxies_String] /; initializedQ :=
	Module[{},
		curlSetProxies[id, proxies];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	]

(******************************************************************************)
(* 
	Use Global Cookie share.
*)
CURLAutoCookies[CURLHandle[id_Integer]] /; initializedQ :=
	Module[{cc},
		cc = curlAutoCookies[id];
		If[successQ, cc, Throw[$Failed,CURLLink`Utilities`Exception]]
	]
	
(******************************************************************************)
(*
	If output is to be saved to a file, this function must be used to specify 
	the filename, as well as cause some triggers to tell libcurlLink what is 
	going on.
*)
CURLFileInfo[CURLHandle[id_Integer], fileName_String, format:(True|False)] /; initializedQ :=
	(
		curlFileInfo[id, fileName, format];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
(******************************************************************************)
(*
	If input is to be read from a file, this function must be used to specify 
	the filename, as well as cause some triggers to tell libcurlLink what is 
	going on.
*)
CURLUploadFileInfo[CURLHandle[id_Integer], fileName_String] /; initializedQ :=
	(
		curlUploadFileInfo[id, fileName];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)
(******************************************************************************)
(*
	If input is to be read from a file, this function must be used to specify 
	the filename, as well as cause some triggers to tell libcurlLink what is 
	going on.
*)
CURLUploadContent[CURLHandle[id_Integer], content_String] /; initializedQ :=
	(
		curlUploadContent[id, content];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)	
(******************************************************************************)
CURLWriteInfo[CURLHandle[id_Integer], function_String] /; initializedQ :=
	(
		curlWriteInfo[id, function];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	)

(******************************************************************************)

CURLSetCert[cert_String] /; initializedQ :=
	(
		curlSetCert[cert];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]];
	)
(****************************************************************************)
(* 
	upload multi-form post data.
*)

 CURLForm[CURLHandle[id_Integer], name_String, type:(_String|Automatic),fln:(_String|Automatic), udata:(_List|File[_String]), ln_Integer, headers_String] /; initializedQ :=
	Module[
{cc,mimetype,filepath,filename,data,length},
           Which[
                 fln===Automatic,
                 filename="Automatic";
                 ,
                 fln==="",
                 filename="";
                 ,
                 True,
                 filename=fln
                 
                 ];
           If[Head[udata]===File,
              
              filepath=udata[[1]];
              data={0};
              length=0;
              ,
              filepath="";
              data=udata;
              length=ln;
              ];
           
           If[!StringQ[type],mimetype="",mimetype=type];

           cc = curlMultiForm[id, name, mimetype,filename,filepath, Developer`ToPackedArray[data], length, headers];

              
		If[successQ, cc, Throw[$Failed,CURLLink`Utilities`Exception]]
	]
	
(****************************************************************************)
(* 
	Get a human readable error message from libcurl indicating 
	what went wrong during the connection (if anything).
*)
CURLError[cc_Integer] /; initializedQ :=
	Module[{str}, str = curlError[cc];	If[successQ, str, Throw[$Failed,CURLLink`Utilities`Exception]]]

(****************************************************************************)

(* URL Encode a String *)
CURLEscape[url_String] /; initializedQ :=
	Module[{str},
		str = curlEscape[url];	
		If[successQ, str, Throw[$Failed,CURLLink`Utilities`Exception]]
	]
	
(****************************************************************************)

(* URL Decode a String *)
CURLUnescape[url_String] /; initializedQ :=
	Module[{str},
		str = curlUnescape[url];	
		If[successQ, str, Throw[$Failed,CURLLink`Utilities`Exception]]
	]

(****************************************************************************)
(* 
	The following functions pass different options to curl_easy_setopt().  
	Since curl_easy_setopt is a macro designed to take several different 
	types of parameters, including kinds that are not easily passed from 
	Mathematica to a LibraryLink function (function pointers) care must 
	be taken to ensure HTTPClient's c code can interpert these properly.
	
	Note: the order in which these are defined is important.
*)

(* Most basic type of option.  To be used when the option takes an integer value *)
CURLOption[CURLMultiHandle[id_Integer], option_?CURLLink`CURLMultiOptionQ, param_Integer] /; initializedQ :=
	Module[{cc},
		cc = curlMultiOptionInteger[id, CURLLink`$CURLMultiOptions[option], opTypes["Integer"], Min[param,Developer`$MaxMachineInteger]];
		If[cc =!= 0 || !successQ, 
			Throw[$Failed,CURLLink`Utilities`Exception]
		];
	]	
CURLOption[CURLHandle[id_Integer], option_?CURLLink`CURLOptionQ, param_Integer] /; initializedQ :=
	Module[{cc},
		cc = curlOptionInteger[id, Replace[option, CURLLink`$CURLOptions], opTypes["Integer"], Min[param,Developer`$MaxMachineInteger]];
		If[cc =!= 0 || !successQ, 
			Throw[$Failed,CURLLink`Utilities`Exception]
		];
	]
	
(* Option to set the address where the data downloaded is to be stored. *)
CURLOption[CURLHandle[id_Integer], option:("CURLOPT_DEBUGDATA"|"CURLOPT_HEADERDATA"|"CURLOPT_XFERINFODATA"|"CURLOPT_WRITEDATA"|"CURLOPT_WRITEHEADER"|"CURLOPT_READDATA"), opType_String] /; initializedQ := 
	Module[{cc},
		cc = curlOptionInteger[id,Replace[option, CURLLink`$CURLOptions], opTypes[opType], 0];
		If[cc =!= 0 || !successQ, 
			Throw[$Failed,CURLLink`Utilities`Exception]
		];
	]

(* Set various callback functions for libcurl *)
CURLOption[CURLHandle[id_Integer], option_?CURLLink`CURLOptionQ, param_?curlCallbackQ] /; initializedQ :=
	Module[{cc},
		cc = curlOptionInteger[id, Replace[option, CURLLink`$CURLOptions], opTypes["CallbackPointer"], callbackEnum[param]];
		If[cc =!= 0 || !successQ, 
			Throw[$Failed,CURLLink`Utilities`Exception]
		];
	]
	
(* Set any option that requires a String as the parameter, such as setting the URL *)
CURLOption[CURLHandle[id_Integer], option_?CURLLink`CURLOptionQ, param_String] /; initializedQ :=
	Module[{cc}, 
		cc = curlOptionString[id, Replace[option, CURLLink`$CURLOptions], param];
		If[cc =!= 0 || !successQ,
			Throw[$Failed,CURLLink`Utilities`Exception]
		];
	]
	
(* Set options where the parameter consist of a list of bytes. *)  
CURLOption[CURLHandle[id_Integer], option_?CURLLink`CURLOptionQ, param_List] /; initializedQ :=
	Module[{cc}, 
		cc = curlOptionTensor[id, Replace[option, CURLLink`$CURLOptions], Developer`ToPackedArray[param]]; 
		If[cc =!= 0 || !successQ,
			Throw[$Failed,CURLLink`Utilities`Exception]
		];
	]

(* Set options where the parameter consist of a byte array. *)  
CURLOption[CURLHandle[id_Integer], option_?CURLLink`CURLOptionQ, param_?ByteArrayQ] /; initializedQ :=
	Module[{cc}, 
		cc = curlOptionByteArray[id, Replace[option, CURLLink`$CURLOptions], param]; 
		If[cc =!= 0 || !successQ,
			Throw[$Failed,CURLLink`Utilities`Exception]
		];
	]


(* Set True/False Options *)
CURLOption[handle_CURLHandle, option_?CURLLink`CURLOptionQ, param:(True|False)] /; initializedQ :=
	CURLOption[handle, option, Boole[param]]
	
(****************************************************************************)
(* Supply a custom header to be sent when doing CURLPerform *)
CURLAddHeader[CURLHandle[id_Integer], header_String] /; initializedQ :=
	Module[{cc}, 
		cc = curlAddHeader[id, header];
		If[successQ, cc, Throw[$Failed,CURLLink`Utilities`Exception]]
	]


(****************************************************************************)
(* Inform the Async handle to store its cookies when finished. *)
CURLAsyncCookies[CURLHandle[id_Integer], val:(True|False)] /; initializedQ :=
	Module[{}, 
		curlAsyncCookies[id, val];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	]
	
(****************************************************************************)
(* Change the default behavior of a handle to be asynchronous. *)
CURLSetAsync[CURLHandle[id_Integer], val:(True|False)] /; initializedQ :=
	Module[{type = $handleType["WOLFRAM_EASY_HANDLE"]}, 
		curlSetAsync[id, type,val];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	]
CURLSetAsync[CURLMultiHandle[id_Integer], val:(True|False)] /; initializedQ :=
	Module[{type = $handleType["WOLFRAM_MULTI_HANDLE"]}, 
		curlSetAsync[id, type,val];
		If[!successQ, Throw[$Failed,CURLLink`Utilities`Exception]]
	]	
(*
	Add handles to a multihandle
*)

getIDs[CURLHandle[id_Integer]]:=id
getIDs[CURLMultiHandle[id_Integer]]:=id
CURLMultiHandleInit[] /; initializedQ :=CURLHandleLoad[]

CURLMultiHandleAdd[multihandle_CURLMultiHandle,easyhandle_CURLHandle]/; initializedQ:=
Module[{mID,easyID},
		mID = getIDs[multihandle];
		easyID=getIDs[easyhandle];
		curlMultiAdd[mID, easyID];
		If[successQ, multihandle,Throw[$Failed,CURLLink`Utilities`Exception]]
	]
CURLMultiHandleAdd[multihandle_CURLMultiHandle,easyhandles_List]/; initializedQ:=
	(Map[CURLMultiHandleAdd[multihandle,#]&,easyhandles])

CURLMultiHandleRemove[multihandle_CURLHandle,easyhandles_List]/; initializedQ:=
Module[{mID,easyIDs},
		mID = getIDs[multihandle];
		easyIDs=Map[getIDs[#]&,easyhandles];
		Map[curlMultiRemove[mID, #]&,easyIDs];
		If[successQ, multihandle,Throw[$Failed,CURLLink`Utilities`Exception]]
	]
CURLCreateRunThread[multihandle_CURLMultiHandle]/; initializedQ:=
Module[{mID,type = $handleType["WOLFRAM_MULTI_HANDLE"]},
		mID = getIDs[multihandle];
		createRunThread[mID,type];
		If[successQ, multihandle,Throw[$Failed,CURLLink`Utilities`Exception]]
	]
CURLCreateRunThread[easyHandle_CURLHandle]/; initializedQ:=
Module[{eID,type = $handleType["WOLFRAM_EASY_HANDLE"]},
		eID = getIDs[easyHandle];
		createRunThread[eID,type];
		If[successQ, easyHandle,Throw[$Failed,CURLLink`Utilities`Exception]]
	]

CURLWait[easyHandle_CURLHandle]/; initializedQ:=
Module[{eID,type = $handleType["WOLFRAM_EASY_HANDLE"]},
		eID = getIDs[easyHandle];
		curlWait[eID,type]
	]

curllinkFetaures[]/;initializedQ:=
Module[{features,sel},
	features = curlVersionFeatures[];
	sel = Positive[BitAnd[features, 2^Range[0, Length[CURLInfo`libcurlFeatures] - 1]]];
	AssociationThread[CURLInfo`libcurlFeatures, sel]
]

curllinkProtocols[]/;initializedQ:=
Module[{supportedProtocols={}},
	Do[AppendTo[supportedProtocols, Quiet[Check[ToUpperCase[curlProtocols[i]]->True,Break[]]]],{i,1,30}];
	
	AssociateTo[CURLInfo`protocols,Association[supportedProtocols]]
]

curllinkVersion[]/;initializedQ:=
Module[{},
	<|
	"libcurl"->curlVersion[0],
	"ssl" ->curlVersion[1],
	"Host"->curlVersion[2],
	"libz"->curlVersion[3],
	"libssh2"->curlVersion[4]
	|>
]
CURLLinkInformation[]:=
<|
	"Version" -> curllinkVersion[], 
	"Protocols" -> curllinkProtocols[], 
	"Features" -> curllinkFetaures[]
 |>		
(******************************************************************************)

End[] (* End Private Context *)

EndPackage[]
