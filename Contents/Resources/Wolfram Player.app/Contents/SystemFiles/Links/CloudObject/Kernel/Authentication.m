(* ::Package:: *)

(* Mathematica package *)
BeginPackage["CloudObject`"]

System`CloudConnect;
System`CloudDisconnect;

Hold[System`$CloudConnected];
Hold[System`$UserURLBase];
Hold[System`$WolframID];
Hold[System`$WolframUUID];
Hold[System`$CloudUserID];
Hold[System`$CloudUserUUID];
Hold[System`$RegisteredUserName];
Hold[System`$RequesterWolframID];
Hold[System`$RequesterWolframUUID];

Hold[System`$CloudVersion];
Hold[System`$CloudVersionNumber];
Hold[System`$CloudWolframEngineVersionNumber];

Hold[CloudObject`$CloudDebug];

Begin["`Private`"]

Unprotect[CloudConnect];
Unprotect[CloudDisconnect];
Unprotect[$CloudConnected];

$AuthenticationMethod = "XAuth";
$tag = "CloudObjectCatchThrowTag";
$Flag = True;
$PurgeCredFileOnFailure = True;
$credsDir = FileNameJoin[{$UserBaseDirectory, "ApplicationData", "CloudObject", "Authentication"}];
$credsFile ="config.pfx";(*TODO:make this an actual pfx file*)


CloudObject`$LogConnectionInformation := CloudObject`$LogConnectionInformation = With[{v = fileLoggingEnabledQ[]},
	If[
		TrueQ[v],
		v,
		False
	]
]

$cloudConnectLoggingFile := $cloudConnectLoggingFile = Quiet[CreateFile[FileNameJoin[{$TemporaryDirectory, "CloudConnect", CreateUUID[]}]]]

fileLoggingEnabledQ[] := FileExistsQ[FileNameJoin[{$credsDir, "logging.conf"}]]

connectionLog[message__] /; CloudObject`$LogConnectionInformation := Quiet[PutAppend[
	{FrontEnd`Private`$KernelName, $ProcessID, DateString[]}, 
	message,
	"-------",
	$cloudConnectLoggingFile
]]

RememberMeValueFromDialogCheckbox[]:= If[SameQ[Head[$FrontEnd], FrontEndObject],
	TrueQ[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "DialogSettings", "Login", "RememberMe"}]],
	True
]

SetRememberMeValueFromDialogCheckbox[value_] := If[SameQ[Head[$FrontEnd], FrontEndObject],
	CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "DialogSettings", "Login", "RememberMe"}] = value
]

frontEndCredsQ[] := TrueQ[And[$checkFECredsQ, $Notebooks, CurrentValue["WolframCloudConnected"]]]

getAndProcessFEAuth[base_] := Catch[
	processFEAuthInfo[getFEAuthInfo[base]],
	$tag
]

getFEAuthInfo[base_] := Quiet[
	connectionLog["Fetching Authentication from FrontEnd for ", base];
	MathLink`CallFrontEnd@FrontEnd`WolframCloud`GetAuthentication[]
	]

processFEAuthInfo[res:{_String}] := Module[{tmp = extractAuthInfo@@res},
	If[getValueFromAuthInfo[tmp, "CloudBase"] === $CloudBase,
			$checkPFXCacheQ = False;
			CloudObject`Internal`SetAuthentication@@res,
			(*otherwise call CloudConnectStatus again to possibly load PFX configuration*)
			CloudObject`Internal`CloudConnectStatus[$CloudBase]
		]
]
processFEAuthInfo[_] := CloudObject`Internal`CloudConnectStatus[$CloudBase]

$connectedCloud = None;
$checkPFXCacheQ = True;
$checkFECredsQ = True;
(*first time $CloudConnected is called check PFX file for cached credentials*)
CloudObject`Internal`CloudConnectStatus[args___]/;$checkPFXCacheQ := CheckAbort[
	If[frontEndCredsQ[],
		$checkFECredsQ = False;
		getAndProcessFEAuth[args],
		$checkPFXCacheQ = False;
		If[TrueQ[$hasCredsFile],
			
			Block[{$CacheLogin = True, $PurgeCredFileOnFailure = False},
				StringQ[Catch[iConnectAndVerify[{}, False, {}, 60], $tag]]
			],
			iCloudConnectStatus[args]
		]
	]
	, AbortProtect[flushCredentials[]; False]
]

extractURLDomain[url_String] := With[{d = Quiet[URLParse[url, "Domain"]]},
	If[StringQ[d],
		(*memoizing domain extraction to avoid URLParse overhead*)
		extractURLDomain[url] = d,
		url
	]
]
extractURLDomain[other__] := other

CloudObject`Internal`CloudConnectStatus[args___] := iCloudConnectStatus[args]

(* the various $*Status associations store connection status and other information by $CloudBase domains *)
initializeCloudConnectionInformation[] := Module[{},
	$CCStatus = Association[];
	$WIDStatus = Association[];
	$WUUIDStatus = Association[];
	$CUIDStatus = Association[];
	$CUUUIDStatus = Association[];
	$RUserNameStatus = Association[];
	$UserURLBaseStatus = Association[];
	$CloudVersionStatus = Association[];
	$CloudVersionNumberStatus = Association[];
	$CloudWolframEngineVersionNumberStatus = Association[];
	$OAuthSigVersion = Association[];
]

initializeCloudConnectionInformation[]

iCloudConnectStatus[] := $CCStatus
iCloudConnectStatus[server_String] := TrueQ[$CCStatus[extractURLDomain[server]]]
SetCloudConnectStatus[s_String, status_] := With[{server = extractURLDomain[s]},
	If[$checkPFXCacheQ, $checkPFXCacheQ = False];
	$connectedCloud = server;
	$CCStatus[server] = status
]

CloudObject`Internal`WolframIDStatus[] := $WIDStatus
CloudObject`Internal`WolframIDStatus[server_String] := Replace[$WIDStatus[extractURLDomain@server], _Missing :> None]
SetWolframIDStatus[server_String, status_] := $WIDStatus[extractURLDomain@server] = status

CloudObject`Internal`CloudUserIDStatus[] := $CUIDStatus
CloudObject`Internal`CloudUserIDStatus[server_String] := Replace[$CUIDStatus[extractURLDomain@server], _Missing :> None]
SetCloudUserIDStatus[server_String, status_] := $CUIDStatus[extractURLDomain@server] = status

CloudObject`Internal`WolframUUIDStatus[] := $WUUIDStatus
CloudObject`Internal`WolframUUIDStatus[server_String] := Replace[$WUUIDStatus[extractURLDomain@server], _Missing :> None]
SetWolframUUIDStatus[server_String, status_] := $WUUIDStatus[extractURLDomain@server] = status

CloudObject`Internal`CloudUserUUIDStatus[] := $CUUUIDStatus
CloudObject`Internal`CloudUserUUIDStatus[server_String] := Replace[$CUUUIDStatus[extractURLDomain@server], _Missing :> None]
SetCloudUserUUIDStatus[server_String, status_] := $CUUUIDStatus[extractURLDomain@server] = status

CloudObject`Internal`RegisteredUserNameStatus[] := $RUserNameStatus
CloudObject`Internal`RegisteredUserNameStatus[server_String] := Replace[$RUserNameStatus[extractURLDomain@server], _Missing :> ""]
SetRegisteredUserNameStatus[server_String, status_] := $RUserNameStatus[extractURLDomain@server] = status

CloudObject`Internal`OAuthSignatureVersionStatus[] := $OAuthSigVersion
CloudObject`Internal`OAuthSignatureVersionStatus[server_String] := $OAuthSigVersion[extractURLDomain@server]
SetOAuthSignatureVersionStatus[server_String, status_] := $OAuthSigVersion[extractURLDomain@server] = status

CloudObject`Internal`UserURLBaseStatus[] := $UserURLBaseStatus
CloudObject`Internal`UserURLBaseStatus[server_String] := Replace[$UserURLBaseStatus[extractURLDomain@server], _Missing :> None]
SetUserURLBaseStatus[server_String, status_] := $UserURLBaseStatus[extractURLDomain@server] = status

CloudObject`Internal`CloudVersionStatus[] := $CloudVersionStatus
CloudObject`Internal`CloudVersionStatus[server_String] := Replace[$CloudVersionStatus[extractURLDomain@server], _Missing :> None]
SetCloudVersionStatus[server_String, status_] := $CloudVersionStatus[extractURLDomain@server] = status

CloudObject`Internal`CloudVersionNumberStatus[] := $CloudVersionNumberStatus
CloudObject`Internal`CloudVersionNumberStatus[server_String] := Replace[$CloudVersionNumberStatus[extractURLDomain@server], _Missing :> None]
SetCloudVersionNumberStatus[server_String, status_] := $CloudVersionNumberStatus[extractURLDomain@server] = status

CloudObject`Internal`CloudWolframEngineVersionNumberStatus[] := $CloudWolframEngineVersionNumberStatus
CloudObject`Internal`CloudWolframEngineVersionNumberStatus[server_String] := Replace[$CloudWolframEngineVersionNumberStatus[extractURLDomain@server], _Missing :> None]
SetCloudWolframEngineVersionNumberStatus[server_String, status_] := $CloudWolframEngineVersionNumberStatus[extractURLDomain@server] = status
Options[CloudConnect] = {CloudBase :> $CloudBase};

$cloudconnecthiddenoptions = {
	"RememberMe"-> Automatic, 
	"Prompt":> TrueQ[PacletManager`$AllowInternet](*TODO: update once this is a real system symbol*), 
	"Notifications" -> {}, 
	"TimeConstraint" -> 60
};


CloudConnect[args___] := With[{res = Catch[cloudConnect[args], $tag]},
	res /; res =!= "CCUnevaluatedTag"]
	
	
ensureCloudConnected[] := If[
	TrueQ[$CloudConnected] (*will automatically connect with cached credentials if needed*),
	$CloudUserID,
	CloudConnect[] (*return whatever CloudConnect would; $CloudUserID, $Canceled, or $Failed*)
]
	
$cloudBaseAbbreviations = {
	"local" -> "http://localhost",
	"devel" -> "https://www.devel.wolframcloud.com",
	"test" -> "https://www.test.wolframcloud.com",
	"prd" -> "https://www.wolframcloud.com",
	"production" -> "https://www.wolframcloud.com"
};

cloudConnect[___]/;$CloudEvaluation := "CCUnevaluatedTag"
cloudConnect[args___, opts___?OptionQ]:=Block[{$hasFailed=False, $PurgeCredFileOnFailure = False,
	ov = Check[OptionValue[{CloudConnect, $cloudconnecthiddenoptions}, {opts}, "RememberMe"], Throw[$Failed,$tag]], 
	prompt = OptionValue[{CloudConnect, $cloudconnecthiddenoptions}, {opts}, "Prompt"],
	notes = OptionValue[{CloudConnect, $cloudconnecthiddenoptions}, {opts}, "Notifications"],
	timelimit = OptionValue[{CloudConnect, $cloudconnecthiddenoptions}, {opts}, "TimeConstraint"],
	base = Replace[OptionValue[{CloudConnect, $cloudconnecthiddenoptions}, {opts}, CloudBase], URL[server_] :> server]
}, With[{res=Catch[
Block[{$CloudBase},
	$CloudBase = Replace[base, $cloudBaseAbbreviations];
	If[!FreeQ[{opts}, CloudBase], SetRememberMeValueFromDialogCheckbox[ov = True]];
	If[
		Not[MatchQ[ov, Automatic|True|False]],
		Message[CloudConnect::opttf,"RememberMe",ov];connectionLog["Invalid \"RememberMe\" Value: Exitting"];Throw[$Failed,$tag]
	];
	Block[{$CacheLogin = TrueQ[Replace[OptionValue[{CloudConnect, $cloudconnecthiddenoptions}, {opts}, "RememberMe"], Automatic -> RememberMeValueFromDialogCheckbox[]]]},
	iConnectAndVerify[{args}, prompt, notes, timelimit]]],
	$tag]},
	establishCloudConnect[base];
	res]
]

initiateCloudConnection[] /; TrueQ[$Notebooks] := CompoundExpression[
	connectionLog["Initiating Connection"],
	MathLink`CallFrontEnd@FrontEnd`WolframCloud`ConnectionInitiated[]
]
establishCloudConnect[server_String] /; TrueQ[$Notebooks] := Block[{$CloudBase = server},
	If[SameQ[$CloudUserID, None], (*if user didn't log in*)
			FEConnectFail[1700],
			With[{cloudbase = $CloudBase, username = $CloudUserID, displayname = $RegisteredUserName, uuid = $CloudUserUUID},
				connectionLog["Connection Established with ", server];
				MathLink`CallFrontEnd@FrontEnd`WolframCloud`ConnectionEstablished[cloudbase, username, displayname, uuid]]
			]
		]

$verificationURL := If[SameQ[StringTake[$CloudBase,-1], "/"],
			StringJoin[$CloudBase, "files/auth"],
			StringJoin[$CloudBase, "/files/auth"]
		]

(*ping server to verify credentials are valid and server is up*)
pingCloudServer[]:= Module[{res,status},
	res=authenticatedURLFetch[$verificationURL, {"StatusCode", "ContentData"}, "Headers" -> {"Content-Type" -> "text/plain"}, "VerifyPeer" -> False, "DisplayProxyDialog" -> False];
	If[MatchQ[res, {_, _}], {status, res} = res, FEConnectFail[1600]; CloudDisconnect[]; Throw[$Failed,$tag]];
	res = fcc[res];
	If[
		UnsameQ[status, 200],
		handleServerResponse[{status, res}]; CloudDisconnect[]; Throw[$Failed,$tag],
		res = importFromJSON[res];
		setUserURLBaseFromContentData[res];
		setCloudVersionInfoFromContentData[res];
		setWolframEngineVersionFromContentData[res];
		SetCloudDirectory[];
		connectionLog["Cloud Server Ping Successful for ", $verificationURL];
		$CloudUserID
	]
]

setUserURLBaseFromContentData[res:{__Rule}] := Module[{userURLBase},
	userURLBase = Lookup[res, "userURLBase"];
	If[StringQ[userURLBase], setUserURLBase[userURLBase]]
]

setCloudVersionInfoFromContentData[res:{__Rule}] := Module[{cloudVersionNumber, cloudReleaseDate},
	cloudVersionNumber = Lookup[res, "cloudVersion"];
	cloudReleaseDate = Lookup[res, "cloudReleaseDate"];
	SetCloudVersionNumber[cloudVersionNumber];
	SetCloudVersion[cloudVersionNumber, cloudReleaseDate];
]

cloudVersionStringFromDate[date_String] := StringJoin["(",
	DateString[date, {"MonthName", " ", "Day", ", ", "Year"}],
	")"
]
cloudVersionStringFromDate[___] := ""

setWolframEngineVersionFromContentData[res:{__Rule}] := 
	With[{wolframEngineVersion = Interpreter["Number"][Lookup[res, "wolframEngineVersion"]]},
		SetCloudWolframEngineVersionNumber[wolframEngineVersion]
	]

iConnectAndVerify[args___] := If[
	(* if FE has creds try loading them and see if they match the login info provided*)
	And[frontEndCredsQ[], frontEndCredsMatchInputQ[args]],
	getAndProcessFEAuth[$CloudBase];
	pingCloudServer[],
	
	initiateCloudConnection[];
	With[{res = iCloudConnect[args]},
		If[MatchQ[res, $CloudUserID],
			pingCloudServer[],
			res
		]
	]
]

frontEndCredsMatchInputQ[args___] := Catch[
	With[{authInfo = extractAuthInfo@@getFEAuthInfo[$CloudBase]},
		(* only check FECreds once *)
		$checkFECredsQ = False; 
		(* fetch creds from FrontEnd via MathLink *)
		SameQ[getValueFromAuthInfo[authInfo, "CloudBase"], $CloudBase]; 
		(* check fetched creds against input arguments *)
		matchingCloudUserIDQ[getValueFromAuthInfo[authInfo, "WolframID"], args]
	],
	$tag
]

matchingCloudUserIDQ[_, {}, ___] := True (* FE has creds, and user called CloudConnect[] without a user name *)
matchingCloudUserIDQ[id_String, {id_String, ___}, ___] := True
matchingCloudUserIDQ[___] := False

canceledCloudConnect[] := CompoundExpression[
	If[TrueQ[$Notebooks], MathLink`CallFrontEnd @ FrontEnd`WolframCloud`ConnectionCancelled[]],
	connectionLog["CloudConnect Canceled"],
	$Failed
]

noCacheLogin[] := CompoundExpression[
	connectionLog["Login called without pw and \"RememberMe\" -> False"],
	$Failed
]

iCloudConnect[{}, prompt:(True|False), notes_, timelimit_] := With[{r = If[TrueQ[$CacheLogin], fetchCredentials[], noCacheLogin[]]},
	If[FreeQ[r, $Failed],
		establishCloudConnect[$CloudBase]; $CloudUserID,
		connectionLog["fetchCredentials Failed"];
		If[TrueQ[prompt],
			authenticate["", notes, timelimit],
			canceledCloudConnect[]
		]
	]
]
iCloudConnect[{username_String}, True, notes_, timelimit_] :=
	With[
		{r=If[TrueQ[$CacheLogin], fetchCredentials[], noCacheLogin[]]},
		If[
			FreeQ[r, $Failed],
			If[SameQ[username, $CloudUserID],
				establishCloudConnect[$CloudBase]; $CloudUserID,
				connectionLog["Cached Credentials conflict with provided login; Disconnecting"];
				CloudDisconnect[]; authenticate[username, notes, timelimit]
			],
			connectionLog["fetchCredentials Failed"];
			authenticate[username, notes, timelimit]
		]
	]

iCloudConnect[{username_String,password_String}, _, notes_, timelimit_] := CompoundExpression[
	If[UnsameQ[username, $CloudUserID], CloudDisconnect[]]; connectionLog["Cached Credentials conflict with provided login; Disconnecting"];
	authenticate[username, password, notes, timelimit]
]
iCloudConnect[{args__},___] := CompoundExpression[
	System`Private`Arguments[CloudConnect[args], {0, 2}],
	canceledCloudConnect[]
]
iCloudConnect[{___}, False, __] := canceledCloudConnect[]

(*removes connection information for current $CloudBase*)
flushCredentials[] := With[{username = $CloudUserID},
If[$Notebooks==True,(*if FrontEnd is available*)
MathLink`CallFrontEnd@FrontEnd`WolframCloud`ConnectionTerminated[username]];
connectionLog["Flushing Credentials for ", $CloudBase];
setCloudConnected[False];
setCloudUserID[None];
setCloudUserUUID[None];
setRegisteredUserName[""];
SetCloudWolframEngineVersionNumber[None];
SetCloudVersionNumber[None];
SetCloudVersion[None];
setUserURLBase[None];
setAccessData["",""];
clearCookies[$CloudBase];
True
]
(*removes all connection information, not just for current $CloudBase*)
flushCredentials[All] := With[{username = $CloudUserID},
If[$Notebooks==True,(*if FrontEnd is available*)
MathLink`CallFrontEnd@FrontEnd`WolframCloud`ConnectionTerminated[username]];
connectionLog["Flushing All Credentials"];
clearCookies[All];
initializeCloudConnectionInformation[];
setAccessData["",""];
True
]

clearCookies[url_String?StringQ] /; TrueQ[$VersionNumber >= 11] := ClearCookies[
StringReplace[url, {RegularExpression["^(.*?)\\."] -> "", "/" -> ""}]
]

clearCookies[All] /; TrueQ[$VersionNumber >= 11] := With[{urls = Keys[iCloudConnectStatus[]]},
	ClearCookies[StringReplace[urls, {RegularExpression["^(.*?)\\."] -> "", "/" -> ""}]]
]

Options[CloudDisconnect] = {CloudBase :> $CloudBase}


CloudDisconnect[args___] := With[{res = Catch[iCloudDisconnect[args], "CloudDC"|$tag]},
	res /; res =!= $Failed
]
(*in-cloud operations handled independently*)
iCloudDisconnect[___] /; $CloudEvaluation := Throw[$Failed, "CloudDC"]
(*remove all login info, regardless of server*)
iCloudDisconnect[All, OptionsPattern[]] := Module[{},
	flushCredentials[All];
	logoutFE[];
	Block[{$PurgeCredFileOnFailure = True}, purgeCreds[All]];
]
iCloudDisconnect[opts___?OptionQ] := Block[{
	$CloudBase = Replace[Check[
		OptionValue[CloudDisconnect, {opts}, CloudBase],
		Throw[$Failed, "CloudDC"], 
		{OptionValue::nodef, OptionValue::optnf}
	], $cloudBaseAbbreviations]
},
	flushCredentials[];
	logoutFE[];
	purgeCreds[$CloudBase];
]

iCloudDisconnect[args__] := Module[{}, System`Private`Arguments[CloudDisconnect[args], {0, 0}];
  Throw[$Failed, "CloudDC"]
]

logoutFE[]/;TrueQ[$Notebooks] := If[
	MathLink`CallFrontEnd[FrontEnd`Value["$NotebookVersionNumber"]] >= 11,
	MathLink`CallFrontEnd@FrontEnd`WolframCloud`Logout[];
	connectionLog["Logging out FrontEnd"]
]

purgeCreds[All] /; $PurgeCredFileOnFailure:= Quiet[DeleteFile[FileNameJoin[{$credsDir, $credsFile}]]]
purgeCreds[server_String, dir_:$credsDir, file_:$credsFile, key_:$storageKey] /; $PurgeCredFileOnFailure:= Block[{
	keychain = encrypt[removeFromKeyChain[getKeyChain[dir, file, key], server], key]
	},
	storeKeyChain[{dir, file}, keychain]
]
(*anything else we don't evaluate*)

fetchCredentials[] := Catch[
	If[Not[FreeQ[getCredentials[], $Failed]], Throw[$Failed, $tag]];
	If[Not[StringQ[$CloudBase]], Throw[$Failed, $tag]]
	,$tag]

(* prompt the user for login credentials, either in an interactive dialog in FE, or terminal dialog when there is no FE *)
getLoginCredentials[username_, notes_, timelimit_] := Module[{creds},
	TimeConstrained[
		If[
			TrueQ[$Notebooks],
			creds = loginDialog[username, notes],
			If[username === "",
				(* username of "" indicates none was provided earlier *)
				creds = AuthenticationDialog[];
				If[AssociationQ[creds],
					creds = {creds["Username"], creds["Password"]},
					creds = $Failed
				],
				creds = AuthenticationDialog["Password"];
				If[AssociationQ[creds],
					creds = {username, creds["Password"]},
					creds = $Failed	
				]
			]
		];
		creds,
		timelimit
	]
]

authenticate[args___] := authenticateUsingCredentials[args]

authenticateUsingCredentials[__] /; SameQ[PacletManager`$AllowInternet, False] := CompoundExpression[
	FEConnectFail[1601],
	Message[CloudConnect::offline],
	$Failed
](*internet connectivity disabled*)
authenticateUsingCredentials[username_String, notes_List, timelimit_]:=With[{creds = getLoginCredentials[username, notes, timelimit]},
    (*placeholder while we wait on real authentication*)
If[creds === $Canceled || Not[MatchQ[creds, {_String, _String}]],
	connectionLog["getLoginCredentials Canceled"]; $Canceled,
    If[TrueQ[And[authenticateWithServer[creds, $AuthenticationMethod], authenticatedQ[]]],
    	If[TrueQ[$CacheLogin] || RememberMeValueFromDialogCheckbox[], storeCredentials[]];
    $CloudUserID,
    Message[CloudConnect::notauth]; $Failed]]
]

authenticateUsingCredentials[username_String,password_String, __]:=With[{creds = {username, password}},
    (*placeholder while we wait on real authentication*)
If[TrueQ[And[authenticateWithServer[creds, $AuthenticationMethod], authenticatedQ[]]],
		If[TrueQ[$CacheLogin](*TODO: re-check dialog status*), storeCredentials[]];
    $CloudUserID,
    Message[CloudConnect::notauth]; $Failed]
]

authenticateUsingCredentials[___] := $Failed

$randomCharacters = Join[
	CharacterRange["0", "9"],
	CharacterRange["A", "Z"],
	CharacterRange["a", "z"]
];

If[Not[ValueQ[$LocalhostAuthURL]], Set[$LocalhostAuthURL, "https://user.devel.wolfram.com/oauth/access_token"]];

getDomain[value_] :=
	StringReplace[value, ("http://" | "https://" | "") ~~ Shortest[domain_] ~~
        RegularExpression["(:[0-9]+)?(/.*)?"] :> domain]

makeAuthURL["localhost"|"localhost:8080/app"] := $LocalhostAuthURL
makeAuthURL[Alternatives[
	"www.devel.wolframcloud.com",
	"www.devel-xkernel.wolframcloud.com"
]] := "https://user.devel.wolfram.com/oauth/access_token"
makeAuthURL[Alternatives[
	"www.test.wolframcloud.com",
	"www.test2.wolframcloud.com",
	"www.test-xkernel.wolframcloud.com",
	"www.prd-masterkernel.wolframcloud.com",
	"datadrop.test.wolframcloud.com"
]] := "https://user.test.wolfram.com/oauth/access_token"
makeAuthURL[Alternatives[
	"datadrop.wolframcloud.com",
	"www.wolframcloud.com",
	"oadeployment.wolframcloud.com"
]] = "https://user.wolfram.com/oauth/access_token"
makeAuthURL[url_String] /; Not[TrueQ[$getDomainFlag]] := Block[{$getDomainFlag=True, res},
	res = Catch[makeAuthURL[getDomain[url]], $tag];
	If[res === $Failed,
	automaticAuthURL[url],
	res
	]
]
makeAuthURL[___] := Throw[$Failed,$tag]

makeSignatureURL["localhost"] := "http://localhost"
makeSignatureURL["localhost:8080/app"] := "http://localhost:8080"
makeSignatureURL["www.devel.wolframcloud.com"] := "http://devel.wolframcloud.com"
makeSignatureURL["www.devel-xkernel.wolframcloud.com"] := "http://devel-xkernel.wolframcloud.com"
makeSignatureURL["www.test.wolframcloud.com"] := "http://test.wolframcloud.com"
makeSignatureURL["www.test2.wolframcloud.com"] := "http://test2.wolframcloud.com"
makeSignatureURL["www.test-xkernel.wolframcloud.com"] := "http://test-xkernel.wolframcloud.com"
makeSignatureURL["www.prd-masterkernel.wolframcloud.com"] := "http://www.prd-masterkernel.wolframcloud.com"
makeSignatureURL["datadrop.test.wolframcloud.com"] = "http://datadrop.test.wolframcloud.com"
makeSignatureURL["datadrop.wolframcloud.com"] = "http://datadrop.wolframcloud.com"
makeSignatureURL["www.wolframcloud.com"] := "http://wolframcloud.com"
makeSignatureURL["oadeployment.wolframcloud.com"] := "http://oadeployment.wolframcloud.com"
makeSignatureURL[url_String]/;Not[TrueQ[$getDomainFlag]] := Block[{$getDomainFlag=True, res},
	res = Catch[makeSignatureURL[getDomain[url]], $tag];
	If[res === $Failed,
	automaticSignatureURL[url],
	res
	]
]
makeSignatureURL[___] := Throw[$Failed,$tag]

$authUrl := makeAuthURL[System`$CloudBase](*"https://user.test.wolfram.com/oauth/access_token"*)
$signatureURL := makeSignatureURL[System`$CloudBase](*"http://test.wolframcloud.com"*)
$oauthVersion = "1.0";
$unixtimebase = AbsoluteTime[{1970, 1, 1, 0, 0, 0}];
unixtime[] :=  Round[AbsoluteTime[TimeZone -> 0] - $unixtimebase];
nonce[] := StringJoin[RandomChoice[$randomCharacters, 20]]
$sigMethod = "HMAC-SHA1";

handleAuthType[Automatic] := handleAuthType["XAuth"]
handleAuthType["XAuth"] := "XAuth"
handleAuthType[type_] :=  Module[{}, Message[CloudConnect::atype, type]; False]

handleCloudBase[Automatic] := "www.wolframcloud.com";
handleCloudBase[url_String] := getDomain[url]
handleCloudBase[other_] := Module[{}, Message[CloudConnect::cbase, other]; False]

handleAuthURL[Automatic] := "https://user.wolfram.com/oauth/access_token"
handleAuthURL[Automatic, base_String] := automaticAuthURL[base]
handleAuthURL[url_String, ___] := url
handleAuthURL[other_] := Module[{}, Message[CloudConnect::aurl, other]; False]

handleSigURL[Automatic] := "http://wolframcloud.com"
handleSigURL[Automatic, base_String] := automaticSignatureURL[base]
handleSigURL[url_String, ___] := url
handleSigURL[other_] := Module[{}, Message[CloudConnect::surl, other]; False]

processAuthInformation["XAuth"][base_String, auth_String, 
  sig_String] := Module[{},
  Set[makeSignatureURL[base], sig];
  Set[makeAuthURL[base], auth];
  True
  ]

processAuthInformation[___][___] := Module[{},
  Message[CloudConnect::config]; $Failed
  ]
  
readCloudConnectConfiguration[info_Association] := Module[{
   cloudbase = handleCloudBase[info["CloudBase"]],
    authtype = handleAuthType[info["AuthenticationMethod"]],
   authurl = handleAuthURL[info["AuthenticationURL"]],
    signatureurl = handleSigURL[info["SignatureURL"]]
   },
  processAuthInformation[authtype][cloudbase, authurl, signatureurl]
]

readCloudConnectConfiguration[other___] := Module[{},
	Message[CloudConnect::invcfg, {other}];
	$Failed
]

$CloudAuthConfigDirectories = {
	(*do in this order so userbasedir has precidence*)
	FileNameJoin[{$InstallationDirectory, "Configuration", "CloudObject","Authentication"}],
	FileNameJoin[{$BaseDirectory, "ApplicationData", "CloudObject","Authentication"}],
	FileNameJoin[{$UserBaseDirectory, "ApplicationData", "CloudObject","Authentication"}]
}

$CloudAuthConfigFiles = Join @@ Map[
	FileNames["*.m", #]&,
	$CloudAuthConfigDirectories
]

If[And[!$CloudEvaluation, $CloudAuthConfigFiles =!= {}],
	Map[
		(connectionLog["Reading CloudConnect configuration file `1`", {#}];
		readCloudConnectConfiguration[Import[#]])&, 
		$CloudAuthConfigFiles
	]
]

handleDirectory[Automatic] := handleDirectory[FileNameJoin[{$UserBaseDirectory, "ApplicationData", "CloudObject","Authentication"}]]
handleDirectory[dir_String] := If[!DirectoryQ[dir],
	Check[CreateDirectory[dir], False],
	dir
]

Options[CloudObject`Utilities`AddAuthenticationInformation] = {
	"AuthenticationMethod" -> Automatic, 
	"AuthenticationURL" -> Automatic,
	"SignatureURL" -> Automatic,
	"ExportDirectory" -> Automatic
};

defaultName[base_]:="Automatic-" <> ToString[Hash[base]]

automaticAuthURL[base_String] := Module[{auth = If[StringMatchQ[base, __ ~~ "/"], base, StringJoin[base, "/"]]},
  StringJoin[auth, "oauth/access_token"]
]
  
automaticSignatureURL[base_String] := Module[{sig = StringReplace[base, "https://www." -> "http://"]},
  If[StringMatchQ[sig, __ ~~ "/"], StringDrop[sig, -1], sig]
]
  
CloudObject`Utilities`AddAuthenticationInformation[URL[base_String], opts:OptionsPattern[]] := CloudObject`Utilities`AddAuthenticationInformation[base, opts]
CloudObject`Utilities`AddAuthenticationInformation[base_String, opts:OptionsPattern[]] := With[{name=defaultName[base]},
	CloudObject`Utilities`AddAuthenticationInformation[name, base, opts]
]	
CloudObject`Utilities`AddAuthenticationInformation[name_String, base_String,OptionsPattern[]] := Module[{a,
	method = OptionValue["AuthenticationMethod"],
	auth = handleAuthURL[OptionValue["AuthenticationURL"], base],
	sig = handleSigURL[OptionValue["SignatureURL"], base],
	dir = handleDirectory[OptionValue["ExportDirectory"]]
},
	If[FreeQ[{method, auth, sig, dir}, False],
		a=Association["CloudBase" -> base, "AuthenticationMethod" -> method, "AuthenticationURL" -> auth, "SignatureURL" -> sig];
		a=Check[readCloudConnectConfiguration[a]; a, $Failed];
		If[a=!=$Failed,
			Check[Export[FileNameJoin[{dir, name<>".m"}], a]; True, $Failed],
			Message[CloudConnect::invcfg]; $Failed
		],
		Message[CloudConnect::invcfg]; $Failed]
]

Options[CloudObject`Utilities`RemoveAuthenticationInformation] = {
	"ExportDirectory" -> Automatic
}

CloudObject`Utilities`RemoveAuthenticationInformation[All, OptionsPattern[]] := Module[{files, dir = handleDirectory[OptionValue["ExportDirectory"]]},
	files = FileNames["*.m", dir];
	DeleteFile/@files;
]

CloudObject`Utilities`RemoveAuthenticationInformation[name_String, OptionsPattern[]] := Module[{dir = handleDirectory[OptionValue["ExportDirectory"]]},
	If[FileExistsQ[#], DeleteFile[#], $Failed
	]&[FileNameJoin[{dir, name<>".m"}]]
]

Options[CloudObject`Utilities`AuthenticationInformation] = {
	"ExportDirectory" -> Automatic
}

CloudObject`Utilities`AuthenticationInformation[OptionsPattern[]] := Module[{dir = handleDirectory[OptionValue["ExportDirectory"]]},
	Association[Map[Module[{data=Import[#]},
		If[AssociationQ[data], FileBaseName[#] -> data]]&, FileNames["*.m",dir]]
]]

CloudObject`Utilities`AuthenticationInformation[name_String, OptionsPattern[]] := Module[{dir = handleDirectory[OptionValue["ExportDirectory"]]},
	If[FileExistsQ[#], Import[#], $Failed
	]&[FileNameJoin[{dir, name<>".m"}]]
]

(*initial authentication*)
makeSubString[{username_String,password_String},{non_,time_}] := ExternalService`EncodeString[
	StringJoin["oauth_consumer_key=", getConsumerKey[], "&oauth_nonce=", non, "&oauth_signature_method=", $sigMethod,
		"&oauth_timestamp=", ToString[time], "&oauth_version=", $oauthVersion,
		"&x_auth_mode=client_auth&x_auth_password=", ExternalService`EncodeString[password],
		"&x_auth_username=", ExternalService`EncodeString[username]]]
(*subsequent requests*)
makeSubString[{non_String, time_}] := ExternalService`EncodeString[
	StringJoin["oauth_consumer_key=" ,getConsumerKey[], "&oauth_nonce=", non, "&oauth_signature_method=", $sigMethod,
		"&oauth_timestamp=", ToString[time], "&oauth_token=", getAccessToken[],
		"&oauth_version=", $oauthVersion
	]]
	
$oauthParameters = {
	"oauth_consumer_key" :> getConsumerKey[],
	"oauth_signature_method" :> $sigMethod,
	"oauth_token" :> getAccessToken[],
	"oauth_version" :> $oauthVersion
};

(* Sort needs some modification to conform to OAuth spec, in which case we're using the following: 			*)
(* 1) Sort by Byte character; padding is used here to avoid shorter strings from always taking precedence	*)
(* 2) if the keys are the same, sort by value; if those are the same then it doesn't really matter the order*)
oauthSigningSorter[_[key1_String, value1_], _[key2_String, value2_]] :=
Module[{maxLen = Max[StringLength /@ {key1, value1, key2, value2}], res},
	res = PadRight[ToCharacterCode[#], maxLen] & /@ {key1, key2};
	If[TrueQ[Equal @@ res],
		OrderedQ[PadRight[ToCharacterCode[#], maxLen] & /@ {value1, value2}],
		OrderedQ[res]
	]
]
(* URLQueryEncode returns "+" for spaces in parameters, which should be "%20" instead *)
oAuthQueryEncode[args_] := StringReplace[URLQueryEncode[args], "+" -> "%20"]
(* take any parameters from the URL or Body and combine them with default OAuth parameters, sorted by key *)
makeSubString[parameters:{_Rule..}] := With[{sorted = Sort[Join[parameters, $oauthParameters], oauthSigningSorter]},
	URLEncode[
		oAuthQueryEncode[sorted]
	]
]
(*initial authentication*)
makeSignatureBase[{username_String, password_String}, {non_, time_}] := StringJoin[
	"POST", "&",
	ExternalService`EncodeString[$authUrl],"&",
	makeSubString[{username, password}, {non, time}]
]
(*subsequent requests*)
makeSignatureBase[{non_, time_}, url_, method_, ctype_,body_, params_] := With[{version = getOAuthSignatureVersion[$CloudBase]},
	Switch[
		version,
		"1.0a", makeSignatureBaseV2[{non, time}, url, method, ctype, body, params],
		_, makeSignatureBaseV1[{non, time}, url, method]
	]
]

getOAuthSignatureVersion[server_] := If[
	MissingQ[$OAuthSignatureVersion],
	requestAndSetOAuthSignatureVersion[server],
	$OAuthSignatureVersion
]

requestAndSetOAuthSignatureVersion[server_] := Module[{status, content},
	Quiet[
        {status, content} = fetchOAuthVersion[$oauthVersionURL];
        If[status =!= 200, {status, content} = fetchOAuthVersion[$legacyOAuthVersionURL]]
    ];
	If[StringQ[content], content = StringTrim[content]];
	Switch[{status, content},
		{200, "1.0a"}, SetOAuthSignatureVersionStatus[server, "1.0a"],
		_, SetOAuthSignatureVersionStatus[server, None]
	]
]

$oauthVersionURL := URLBuild[{$CloudBase,"OAuthVersion"}]

$legacyOAuthVersionURL := URLBuild[{$CloudBase,"app","OAuthVersion"}]

fetchOAuthVersion[url_] := URLFetch[url, {"StatusCode", "Content"}, "FollowRedirects" -> False]

makeSignatureBaseV1[{non_, time_}, url_String, method_String] := StringJoin[
	method, "&", ExternalService`EncodeString[$signatureURL], "&", makeSubString[{non, time}]
]

makeSignatureBaseV2[{non_, time_}, url_String, method_String, ctype_, body_, params_] := Module[{baseurl, parameters},
	{baseurl, parameters} = partitionURLAndParameters[url];
	parameters = Join[parameters, {"oauth_nonce" -> non, "oauth_timestamp" -> ToString[time]}];
	
	(*special handling to scan body for parameters if this is a request with a url-encoded form body *)
	If[StringQ[ctype] && StringStartsQ[ctype, "application/x-www-form-urlencoded"] && body =!= "",
			parameters = Join[parameters, getBodyParameters[body]]
	];
	parameters = Join[parameters, getParameters[params]];
	StringJoin[
		method, "&",
		ExternalService`EncodeString[baseurl], "&",
		makeSubString[parameters]
	]
]

getParameters[params:{__Rule}] := params
getParameters[___] := {}

getBodyParameters[body:{_Integer..}] := getBodyParameters[FromCharacterCode[body]]
getBodyParameters[str_String] := URLQueryDecode[str]
getBodyParameters[___] := {}

partitionURLAndParameters[url_String] := Module[{urldata = URLParse[url], baseurl, parameters},
	parameters = urldata["Query"];
	urldata["Query"] = {};
	baseurl = URLBuild[urldata];
	{baseurl, parameters}
]

(*initial authentication*)
makeSignature[{username_String, password_String}, {non_, time_}] :=
    With[{sigBase =  makeSignatureBase[{username, password}, {non, time}]},
		(*Don't log base because it includes pw*)
        CloudHMAC[sigBase, "Consumer"]
    ]

(*subsequent requests*)
makeSignature[{non_, time_}, url_String, method_String, ctype_, body_, params_] := 
    With[{sigBase = makeSignatureBase[{non, time}, url, method, ctype, body, params]},
    	connectionLog["Signature base: ", sigBase];
        CloudHMAC[sigBase, "Access"]
    ]

(*initial authentication*)
makeOAuthHeader[{username_String, password_String}] := With[{non=nonce[], time=unixtime[]},
StringJoin["OAuth realm=\"", $authUrl, "\", oauth_consumer_key=\"", getConsumerKey[],
	 "\", oauth_nonce=\"", non, "\", oauth_timestamp=\"", ToString[time],
	 "\", oauth_signature_method=\"", $sigMethod, "\", oauth_version=\"", $oauthVersion,
	 "\", oauth_signature=\"", makeSignature[{username, password}, {non, time}], "\""]
]
(*subsequent requests*)
makeOAuthHeader[url_String, method_String, ctype_, body_, params_] := With[{non=nonce[], time=unixtime[]},
StringJoin["OAuth realm=\"", $signatureURL, "\", oauth_consumer_key=\"", getConsumerKey[],
	"\", oauth_token=\"", getAccessToken[],
	 "\", oauth_nonce=\"", non, "\", oauth_timestamp=\"", ToString[time],
	 "\", oauth_signature_method=\"", $sigMethod, "\", oauth_version=\"", $oauthVersion,
	 "\", oauth_signature=\"", ExternalService`EncodeString[makeSignature[{non, time}, url, method, ctype, body, params]], "\""]
]
(*legacy fallback for things that call this directly*)
makeOAuthHeader[url_String, method_String] := makeOAuthHeader[url, method, "None", None]

(*legacy fallback for things that call this directly*)
makeOAuthHeader[url_String, method_String] := makeOAuthHeader[url, method, "None", None, None]
makeOAuthHeader[url_String, method_String, ctype_, body_] := makeOAuthHeader[url, method, ctype, body, None]


makeGeneralOAuthHeader[url_String, method_String, ctype_, body_, params_] := With[{non=nonce[], time=unixtime[]},
StringJoin["OAuth oauth_consumer_key=\"", getConsumerKey[],
	"\", oauth_token=\"", getAccessToken[],
	 "\", oauth_nonce=\"", non, "\", oauth_timestamp=\"", ToString[time],
	 "\", oauth_signature_method=\"", $sigMethod, "\", oauth_version=\"", $oauthVersion,
	 "\", oauth_signature=\"", ExternalService`EncodeString[makeSignature[{non, time}, url, method, ctype, body, params]], "\""]
]

Internal`SignWolframHTTPRequest[request:HTTPRequest[url_String, contents_Association]] /; $CloudConnected := 
 Module[{assoc = contents, method = request["Method"], ctype = request["ContentType"], body = request["Body"], params = request["Query"]}, 
 	assoc["Headers"] = If[KeyExistsQ[assoc, "Headers"],
 		Switch[assoc["Headers"],
 			_List, Join[assoc["Headers"], {"Authorization" -> makeGeneralOAuthHeader[url, method, ctype, body, params]}],
 			_Association, Join[assoc["Headers"], <|"Authorization" -> makeGeneralOAuthHeader[url, method, ctype, body, params]|>],
 			_, <|"Authorization" -> makeGeneralOAuthHeader[url, method, ctype, body, params]|>],
 		<|"Authorization" -> makeGeneralOAuthHeader[url, method, ctype, body, params]|>
 		];
  HTTPRequest[url, assoc]
]

If[Not[$CloudEvaluation],
	Unprotect[$WolframID, $WolframUUID, $CloudUserID, $CloudUserUUID, $RegisteredUserName, $UserURLBase, 
		$RequesterWolframID, $RequesterWolframUUID, $CloudVersion, $CloudVersionNumber, $CloudWolframEngineVersionNumber];
	$CloudUserID := CloudObject`Internal`CloudUserIDStatus[$CloudBase];
	$WolframID := CloudObject`Internal`WolframIDStatus[$CloudBase];
	$RequesterWolframID := $WolframID;
	$CloudUserUUID := CloudObject`Internal`CloudUserUUIDStatus[$CloudBase];
	$WolframUUID := CloudObject`Internal`WolframUUIDStatus[$CloudBase];
	$RequesterWolframUUID := $WolframUUID;
	$RegisteredUserName := CloudObject`Internal`RegisteredUserNameStatus[$CloudBase];
	$UserURLBase := CloudObject`Internal`UserURLBaseStatus[$CloudBase];
	$CloudVersion := If[$CloudConnected, CloudObject`Internal`CloudVersionStatus[$CloudBase], None];
    $CloudVersionNumber := If[$CloudConnected, CloudObject`Internal`CloudVersionNumberStatus[$CloudBase], None];
	$CloudWolframEngineVersionNumber := CloudObject`Internal`CloudWolframEngineVersionNumberStatus[$CloudBase];
	Protect[$WolframID, $WolframUUID, $CloudUserID, $CloudUserUUID, $RegisteredUserName, $UserURLBase, 
		$RequesterWolframID, $RequesterWolframUUID, $CloudVersion, $CloudVersionNumber, $CloudWolframEngineVersionNumber];
]

Unprotect[$OAuthSignatureVersion];
$OAuthSignatureVersion := CloudObject`Internal`OAuthSignatureVersionStatus[$CloudBase];
Protect[$OAuthSignatureVersion];


(*currently not used, but will be used to later for when $Wolfram* symbls are set only when pointed @ a wolfram server...*)
wolframServerQ[base_String] := StringMatchQ[base, __~~"wolframcloud.com/"]
wolframServerQ[__] := False

setCloudUserID[id:(_String|None)] := CompoundExpression[SetCloudUserIDStatus[$CloudBase, id], SetWolframIDStatus[$CloudBase, id]]
setCloudUserUUID[uuid:(_String|None)] := CompoundExpression[SetCloudUserUUIDStatus[$CloudBase, uuid], SetWolframUUIDStatus[$CloudBase, uuid]]
setCloudConnected[value:True|False] := SetCloudConnectStatus[$CloudBase, value]
setRegisteredUserName[name:(_String)] := SetRegisteredUserNameStatus[$CloudBase, name]
setUserURLBase[name:(_String | None)] := SetUserURLBaseStatus[$CloudBase, name]
SetCloudVersion[versionNumber:(_String | None), releaseDate_String] := 
	With[{cloudVersion = StringJoin[versionNumber, " ", cloudVersionStringFromDate[releaseDate]]},
		SetCloudVersionStatus[$CloudBase, cloudVersion]
	]
SetCloudVersionNumber[name:(_String | None)] := SetCloudVersionNumberStatus[$CloudBase, name]
SetCloudWolframEngineVersionNumber[number:(_Real | None)] := (
	SetCloudWolframEngineVersionNumberStatus[$CloudBase, number];
	(* after setting $CloudWolframEngineVersion check if it's running an older version of the Wolfram Engine(and if so, issue a warning) *)
	If[number < $VersionNumber,
		Block[{$NumberMarks = False},
			Message[CloudConnect::clver, InputForm@number]
		]
	]
)

check401AndIssueMessage[content_] := Which[
	StringQ[content] && StringMatchQ[content, "OAuth Verification Failed"~~__],
	Message[CloudConnect::oauth],

	StringQ[content] && SameQ[content, "{\"error\":\"Incorrect username or password\"}"],
	Message[CloudConnect::creds],

	StringQ[content] && StringMatchQ[content, "{\"error\":\"OAuth Verification (2) Failed: Timestamp is out of sequence."~~__],
	Message[CloudConnect::badts],
	
	StringQ[content] && StringMatchQ[content, "{\"error\":\"Consumer Key Invalid\"}"],
	Message[CloudConnect::apkey],

	True,(*TODO: have generic handler here*)
	Message[CloudConnect::creds]
]

FEConnectFail[status_Integer] /; UnsameQ[$hasFailed, True] := If[TrueQ[$Notebooks],
	Set[$hasFailed, True];
	If[Not @ TrueQ @ NotebookTools`$CloudPublishConnect,
		connectionLog["Connection Failed", status];
		MathLink`CallFrontEnd @ FrontEnd`WolframCloud`ConnectionFailed[status],
		canceledCloudConnect[]
	]
]

fcc[arg_] := FromCharacterCode[arg]

getContentsFrom200[content_] := Quiet[
	Check[
		(*TODO: handle other errors here*)
		importFromJSON[content],
		Message[CloudConnect::bdrsp]; $Failed,
		{Import::fmterr}],
	{Import::fmterr}
]
 			
handleServerResponse[{status_, content_}]:= Switch[status,
	412,
	Message[CloudConnect::pcond]; FEConnectFail[status];
	connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed,
	
	408,
	Message[CloudConnect::tout]; FEConnectFail[status];
	connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed,	
	
	405,
	Message[CloudConnect::bdmtd]; FEConnectFail[status];
	connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed,
	
	404,
	Message[CloudConnect::nfnd]; FEConnectFail[status];
	connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed,
	
	403,
	Message[CloudConnect::fbdn]; FEConnectFail[status];
	connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed,

 	401,
 	check401AndIssueMessage[content]; FEConnectFail[status];
    connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed,

 	400,
 	Message[CloudConnect::badts]; FEConnectFail[status];
    connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed,
 	
 	500,
 	Message[CloudConnect::iserr]; FEConnectFail[status];
    connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed, 	
 	
 	503,
  	Message[CloudConnect::unav]; FEConnectFail[status];
    connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed, 	
 	
 	504,
 	Message[CloudConnect::gwto]; FEConnectFail[status];
    connectionLog["Authentication failed: `1`", {status, content}];
 	$Failed,

 	code_ /; 400 < code < 500,(*client error*)
 	Message[CloudConnect::cerr, status]; FEConnectFail[status];
    connectionLog["Authentication failed: `1`", {status, content}];
    $Failed,
    
    code_ /; code > 500,(*server error*)
    Message[CloudConnect::serr, status]; FEConnectFail[status];
    connectionLog["Authentication failed: `1`", {status, content}];,
 	$Failed,
 	
 	_,(*all other*)
 	Message[CloudConnect::uerr, status]; FEConnectFail[status];
    connectionLog["Authentication failed: `1`", {status, content}];,
 	$Failed
]

authenticateWithServer[{username_String, password_String},other_] := Catch[
 Module[{status, contentData, content},
 	status = Check[URLFetch[$authUrl, {"StatusCode", "ContentData"}, "Method" -> "POST",
 "Headers" -> {"Authorization" -> makeOAuthHeader[{username, password}]},
 "Parameters" -> {"x_auth_mode" -> "client_auth",
   "x_auth_password" -> password, "x_auth_username" -> username},
 "DisplayProxyDialog" -> False, "VerifyPeer" -> False], FEConnectFail[1600]; Throw[$Failed, $tag],
 	{MessageName[Utilities`URLTools`FetchURL, "conopen"], 
      MessageName[Utilities`URLTools`FetchURL, "contime"], 
      MessageName[Utilities`URLTools`FetchURL, "erropts"], 
      MessageName[Utilities`URLTools`FetchURL, "httperr"], 
      MessageName[Utilities`URLTools`FetchURL, "nofile"], 
      MessageName[Utilities`URLTools`FetchURL, "nolib"], 
      MessageName[URLFetch, "invhttp"],
      MessageName[General, "offline"], 
      MessageName[General, "nffil"]}];
 If[MatchQ[status, {_, _}],
 	{status, contentData} = status, 
 	FEConnectFail[1600]; Throw[$Failed,$tag]
 ];
 content = fcc[contentData];
 If[status === 200,
 	content=getContentsFrom200[content];
 	If[Not[MatchQ[content, {_Rule..}]], Return[$Failed]];
    connectionLog["Authentication response content: `1`", content];
    setAuthentication[
    	username,
    	Lookup[content, "uuid"],
 		StringJoin[{Lookup[content, "firstname"]," ",Lookup[content, "lastname"]}],
 		Lookup[content, "oauth_token_secret"],
 		Lookup[content, "oauth_token"]
 	],
	handleServerResponse[{status, content}]
 	]],
     $tag]

doAuthenticatedURLFetch[func_, url_String, param_, opts___?OptionQ] :=  If[$CloudConnected,
	enforcedAuthenticatedURLFetch[func, url, param, opts],
	lazyAuthenticatedURLFetch[func, url, param, opts]
]
doAuthenticatedURLFetch[func_, url_String, file_, param_, opts___?OptionQ]:=  If[$CloudConnected,
	enforcedAuthenticatedURLFetch[func, url,file ,param, opts],
	lazyAuthenticatedURLFetch[func, url,file, param, opts]
]
(* lazyAuthenticatedURLFetch will make an unauthenticatd URLFetch call, and retry the call after calling CloudConnect if it returns a 401 *)
lazyAuthenticatedURLFetch[func_, url_String, param_List, opts___?OptionQ] := Catch[
	With[{
		method = Quiet[OptionValue[URLFetch, {opts}, "Method"], {OptionValue::nodef}],
		options = Sequence @@ FilterRules[{opts}, Except["Method"]]
		},
		Module[{res, status},
			res = func[url, Prepend[param, "StatusCode"], "Method" -> method, options, "DisplayProxyDialog" -> False];
			If[ListQ[res], status = First[res]; res = Rest[res], Throw[$Failed, $tag]];
			If[status === 401, 
				With[{con =CloudConnect[]},
					If[con === $CloudUserID,
						enforcedAuthenticatedURLFetch[func, url, param, opts],
						HTTPError[401]
					]
				],
				res
			]
		]
	], $tag
]
lazyAuthenticatedURLFetch[func_, url_String,file_, param_List, opts___?OptionQ] := Catch[
	With[{
		method = Quiet[OptionValue[URLFetch, {opts}, "Method"], {OptionValue::nodef}],
		options = Sequence @@ FilterRules[{opts}, Except["Method"]]
		},
		Module[{res, status},
			res = func[url, file,Prepend[param, "StatusCode"], "Method" -> method, options, "DisplayProxyDialog" -> False];
			If[ListQ[res], status = First[res]; res = Rest[res], Throw[$Failed, $tag]];
			If[status === 401, 
				With[{con =CloudConnect[]},
					If[con === $CloudUserID,
						enforcedAuthenticatedURLFetch[func, url, param, opts],
						HTTPError[401]
					]
				],
				res
			]
		]
	], $tag
]
(*fall-back for URLFetchAsynchronous, etc*)
lazyAuthenticatedURLFetch[func_, args__] := func[args]

connectedCloudQ[cloud_] := Which[
	cloud === $connectedCloud, True,
	extractURLDomain[cloud] === $connectedCloud, True,
	True, False	
]

getContentTypeFromHeaders[headers_List] := Replace["Content-Type", headers]

enforcedAuthenticatedURLFetch[func_, url_String, param__, opts___?OptionQ] := Catch[
    If[Not[connectedCloudQ[$CloudBase]], CloudConnect[]];
    (* Do not remove the quiet, it will send messages if users are using deprecated options in URLFetch *)
    With[{method = Quiet[OptionValue[URLFetch, {opts}, "Method"], {OptionValue::nodef}], 
        headers = Quiet[OptionValue[URLFetch, {opts}, "Headers"], {OptionValue::nodef}],
        body = Quiet[OptionValue[URLFetch, {opts}, "Body"], {OptionValue::nodef}],
        params = Quiet[OptionValue[URLFetch, {opts}, "Parameters"], {OptionValue::nodef}],
        options = Sequence @@ FilterRules[{opts}, Except["Method"|"Headers"]]},
        With[{auth = makeOAuthHeader[url, method, getContentTypeFromHeaders[headers], body, params]},
            connectionLog["Headers: ", Join[headers, {"Authorization" -> auth}]];
            connectionLog[ method, url];
            connectionLog["Options: ", options];
            Check[
                func[url, param, "Method"->method,
                    "Headers" ->Join[headers, {"Authorization" -> auth}],
                    options],
                Throw[$Failed, $tag]]]],
    $tag]
enforcedAuthenticatedURLFetch[func_, url_String,file_, param__, opts___?OptionQ] := Catch[
    If[Not[connectedCloudQ[$CloudBase]], CloudConnect[]];
    (* Do not remove the quiet, it will send messages if users are using deprecated options in URLFetch *)
    With[{method = Quiet[OptionValue[URLFetch, {opts}, "Method"], {OptionValue::nodef}], 
        headers = Quiet[OptionValue[URLFetch, {opts}, "Headers"], {OptionValue::nodef}],
        body = Quiet[OptionValue[URLFetch, {opts}, "Body"], {OptionValue::nodef}],
        params = Quiet[OptionValue[URLFetch, {opts}, "Parameters"], {OptionValue::nodef}],
        options = Sequence @@ FilterRules[{opts}, Except["Method"|"Headers"]]},
        With[{auth = makeOAuthHeader[url, method, getContentTypeFromHeaders[headers], body, params]},
            connectionLog["Headers: ", Join[headers, {"Authorization" -> auth}]];
            connectionLog[method, url];
            connectionLog["Options: ", options];
            Check[
                func[url,file, param, "Method"->method,
                    "Headers" ->Join[headers, {"Authorization" -> auth}],
                    options],
                Throw[$Failed, $tag]]]],
    $tag]
authenticatedURLFetch[url_String, elements:Except[_Rule], opts___?OptionQ] :=
    doAuthenticatedURLFetch[URLFetch, url, elements, opts]
    
authenticatedURLFetch[url_String, opts___?OptionQ] :=
    doAuthenticatedURLFetch[URLFetch, url, "Content", opts]

authenticatedURLFetchAsynchronous[url_String, callback_, opts___?OptionQ] :=
    doAuthenticatedURLFetch[URLFetchAsynchronous, url, callback, opts]

authenticatedURLSave[url_String, file_:Automatic, content_:"Content", opts___?OptionQ] :=
    doAuthenticatedURLFetch[URLSave, url, file, content, opts]

authenticatedURLSaveAsynchronous[url_String, file_:Automatic,callback_:Identity, opts___?OptionQ] :=
    doAuthenticatedURLFetch[URLSaveAsynchronous, url, file,callback, opts]

authenticatedQ[]:=TrueQ[$CloudConnected]

CloudObject`Internal`SetAuthentication[arg_String] := Catch[
	Module[{res = extractAuthInfo[arg], authVersion},
		authVersion = getValueFromAuthInfo[res, "Version"];
		processAuthInfo[authVersion][res]
	],
 $tag]
(*SetAuthentication takes a compressed string*)
CloudObject`Internal`SetAuthentication[___] := $Failed

extractAuthInfo[arg_String] := Quiet[
	Check[
		Uncompress[arg],
		Throw[$Failed, $tag]
	]
]
extractAuthInfo[_] := Throw[$Failed, $tag]

getValueFromAuthInfo[auth_?AssociationQ, key_] := Lookup[
	auth,
	key,
	Throw[$Failed, $tag]
]
getValueFromAuthInfo[___] := Throw[$Failed, $tag]

processAuthInfo[1] = Function[{authInfo},
	Module[{id, uuid, name, secret, token, cloudbase},
		id = getValueFromAuthInfo[authInfo, "WolframID"];
		uuid = getValueFromAuthInfo[authInfo, "WolframUUID"];
		name = getValueFromAuthInfo[authInfo, "RegisteredUserName"];
		secret = getValueFromAuthInfo[authInfo, "AccessSecret"];
		token = getValueFromAuthInfo[authInfo, "AccessToken"];
		cloudbase = getValueFromAuthInfo[authInfo, "CloudBase"];
		Set[$CloudBase, cloudbase];
		setAuthentication[id, uuid, name, secret, token]		
	]
]
processAuthInfo[2] = Function[{authInfo},
	Module[{userURLBase = getValueFromAuthInfo[authInfo, "UserURLBase"]},
		processAuthInfo[1][authInfo];
		setUserURLBase[$CloudBase, userURLBase]
	]
]
processAuthInfo[3] = processAuthInfo[1]
	
processAuthInfo[___] := Throw[$Failed, $tag]

setAuthentication[username_String, uuid_String, userdisplayname_String, accesssecret_String, accesstoken_String] := If[TrueQ[$UseLibraryStorage],
	setCloudUserID[username]; setCloudUserUUID[uuid];
	setRegisteredUserName[userdisplayname];setAccessData[accesstoken, accesssecret];
	setCloudConnected[True]]
setAuthentication[userURLBase_String, username_String, uuid_String, userdisplayname_String, accesssecret_String, accesstoken_String] := If[
	TrueQ[setAuthentication[username, uuid, userdisplayname, accesssecret, accesstoken]],
	setUserURLBase[$CloudBase, userURLBase]
]
setAuthentication[___] := $Failed

CloudObject`Internal`GetAuthentication[version_:3] := Catch[
	makeCredsInfo[version],
	$tag
]

$AuthCredKeysV1 = {"WolframID", "WolframUUID", "RegisteredUserName", "AccessSecret", "AccessToken"};

makeCredsInfo[version_] := Module[{auth = getAuthentication[version]},
	If[And[auth =!= $Failed, Length[auth] == 5],
		auth = AssociationThread[$AuthCredKeysV1, auth];
		Set[auth["CloudBase"], $CloudBase];
		Set[auth["Version"], version];
		{Compress[auth]},
		Throw[$Failed, $tag]
	]
]
(*will add other credential format versions here as they're needed*)
makeCredsInfo[___] := Throw[$Failed, $tag]

getAuthentication[] := getAuthentication[3]

getAuthentication[1] := If[ TrueQ[$CloudConnected],
	{$CloudUserID, $CloudUserUUID, $RegisteredUserName, getAccessSecret[], getAccessToken[]},
	$Failed
](*legacy version*)
getAuthentication[2] := Append[getAuthentication[1], $UserURLBase]
getAuthentication[3] := getAuthentication[1](*current version*)
getAuthentication[___] := $Failed

loadLibCloudObj[] :=
    Block[{path},
		path=FindLibrary["WolframAuthenticate"];
        If[path === $Failed,
        	path = FileNameJoin[{DirectoryName[DirectoryName[$InputFileName]],
        		"LibraryResources", $SystemID, "WolframAuthenticate"}]
        ];
        AuthenticateLibraryFile = path;
        (
        	getConsumerKey = LibraryFunctionLoad[path, "get_consumer_key", {}, "UTF8String"];
            setAccessDataRaw = LibraryFunctionLoad[path, "set_access_data", {"UTF8String", "UTF8String"}, Integer]; (* setAccessData now is WL function that calls this *)
            getAccessToken = LibraryFunctionLoad[path, "get_access_key", {}, "UTF8String"];
            getAccessSecret = LibraryFunctionLoad[path, "get_access_secret", {}, "UTF8String"];(*TODO: remove this?*)
            CloudHMAC = LibraryFunctionLoad[path,"cloud_object_oauth_hmac",{"UTF8String","UTF8String"},"UTF8String"];
			True /; SameQ[LibraryFunction, Sequence @@ (Head /@ {getConsumerKey, setAccessDataRaw, getAccessToken, getAccessSecret, CloudHMAC})]
        ) /; (path =!= $Failed)
    ]

loadLibCloudObj[___] := $Failed

$UseLibraryStorage = If[$CloudEvaluation === True, False, UnsameQ[loadLibCloudObj[], $Failed]]

setAccessData[token_, secret_] := setAccessDataRaw[token, parameterEncode[secret]]

(* implements https://oauth.net/core/1.0/#encoding_parameters *)
parameterEncode[text_] := StringReplace[URLEncode[text], "%7E" -> "~"]

$storageKey:=Internal`HouseKeep[$credsDir, {
	"machine_ID" -> $MachineID,
	"version" -> $Version,
	"system_ID" -> $SystemID,
	"user_name" -> $UserName
}]

initencrypt[] := Symbol["NumberTheory`AESDump`RijndaelDecryption"][]

encrypt[args___]:=With[{ef = (initencrypt[]; Symbol["NumberTheory`AESDump`Private`rijndaelEncryption"])},
	ef[args]
]

decrypt[args___]:= With[{df = (initencrypt[]; Symbol["NumberTheory`AESDump`RijndaelDecryption"])},
	Block[{DataPaclets`SocialMediaDataDump`Private`flagQ = True, NumberTheory`AESDump`Private`flagQ = True},
	df[args]
	]
]

makeCredentialsChain[] := StringJoin[
	"cloudbase=", $CloudBase,
	"token=", getAccessToken[],
	"secret=", getAccessSecret[],
	"username=", $RegisteredUserName,
	"uuid=", $CloudUserUUID,
	"wolframid=", $CloudUserID
]

addToKeyChain[keychain_String,cloudbase_String] := Module[{chain = makeCredentialsChain[]},
	StringJoin[
		Riffle[
			Prepend[
				DeleteCases[
					StringSplit[keychain, "cloudbase="],
					x_String /; StringMatchQ[x, cloudbase~~__]
				],
			chain],
		"cloudbase="]
	]
]

removeFromKeyChain[keychain_String, cloudbase_String] := Module[{},
	StringJoin[
		Riffle[
			DeleteCases[
					StringSplit[keychain, "cloudbase="],
					x_String /; StringMatchQ[x, cloudbase~~__]
				],
		"cloudbase="]
	]
]

storeCredentials[] := storeCredentials[$credsDir, $credsFile, $storageKey]
storeCredentials[directory_String, filename_String, key_String] /; authenticatedQ[] := Catch[
	Block[{$KeyChain = encrypt[addToKeyChain[getKeyChain[directory, filename, key], $CloudBase], key(*,"CiphertextFormat" -> "ByteList"*)]},
	storeKeyChain[{directory, filename}, $KeyChain]	
	], $tag]

storeCredentials[___] := $Failed["NotAuthenticated"]

storeKeyChain[{directory_, filename_}, keychain_] := Block[{$KeyChain = keychain},
With[{
	CreateDirectorymessages := {CreateDirectory::filex, CreateDirectory::privv},
	Savemessages := {Save::wtype, Save::argm, Save::argmu, General::stream, Save::sym, General::privv, General::noopen, DumpSave::bsnosym},
	file = FileNameJoin[{directory, filename}]
},
	If[Not[DirectoryQ[directory]],
		Quiet[Check[
			CreateDirectory[directory],
			Throw[$Failed["NoMakeDir"], $tag],
			CreateDirectorymessages],
		CreateDirectorymessages]
	];
	Quiet[Check[
		DumpSave[file, $KeyChain],
		Throw[$Failed["NoDump"], $tag],
		Savemessages],
	Savemessages];
	True
]]

getKeyChain[] := getKeyChain[$credsDir, $credsFile, $storageKey]
getKeyChain[directory_String, filename_String, key_String] := ReplaceAll[Catch[
	If[TrueQ[$CloudDebug], Identity, Quiet][
	Block[{$KeyChain = ""}, With[
	{Getmessages := {Get::enkey, Get::notencode, General::privv, General::noopen, DumpGet::bgnew}, file = FileNameJoin[{directory, filename}]},
	If[Not[DirectoryQ[directory]], Return[""]];
	Quiet[Check[Get[file], Return[""], Getmessages], Getmessages];
	If[Not[MatchQ[$KeyChain, {_Integer..}]], Throw[$Failed["Bytes"], $tag]];
	$KeyChain = decrypt[$KeyChain, key];
	If[Not[StringQ[$KeyChain]], $KeyChain = ExportString[$KeyChain, "Byte"]];
	$KeyChain
]]], $tag], _$Failed->""]

getCredentials[] := getCredentials[$credsDir, $credsFile, $storageKey]
getCredentials[directory_String, filename_String, key_String] := Catch[
	If[TrueQ[$CloudDebug], Identity, Quiet][
	Block[{$KeyChain},
		$KeyChain = getKeyChain[directory, filename, key];
		If[!StringQ[$KeyChain], connectionLog["$KeyChain not a string"];Throw[$Failed["NotString"], $tag]];
		$KeyChain = StringSplit[$KeyChain, "cloudbase="];
		$KeyChain = Cases[$KeyChain, x_String /; StringMatchQ[x, $CloudBase~~__]];
		If[!MatchQ[$KeyChain, {_String}], connectionLog["$KeyChain lacks CloudBase"];Throw[$Failed["NoCloudBase"], $tag], $KeyChain = First[$KeyChain]];
		$KeyChain = Rest[StringSplit[$KeyChain, {"token=", "secret=", "username=", "uuid=", "wolframid=", "displayname="}]];
		If[MatchQ[$KeyChain, {_String, __String}],
		setAuthentication[Sequence @@ Reverse[$KeyChain]],
		connectionLog["$KeyChain lacks credentail pair"]; Throw[$Failed["NotStringPair"], $tag]
	]
]], $tag]
getCredentials[__] := $Failed["BadParameters"]

$hasCredsFile := FileExistsQ[FileNameJoin[{$credsDir, $credsFile}]]

CheckAbort[
If[Not[ValueQ[$CloudConnected]],
	Unprotect[$CloudConnected];
	If[$CloudEvaluation, 
		Set[$CloudConnected, True],
		SetDelayed[$CloudConnected, CloudObject`Internal`CloudConnectStatus[$CloudBase]]
	];
	Protect[$CloudConnected]
],
AbortProtect[(*cleanup all login info if $CloudConnected is aborted durinig init*)
	CloudDisconnect[]
]]

(*
SetAttributes[{encrypt,decrypt,storeCredentials,
	getCredentials, $storageKey, setAuthentication, authenticateWithServer,
	authenticate,iCloudConnect, flushCredentials}, Locked];
*)
SetAttributes[CloudConnect, ReadProtected];
Protect[CloudConnect];
SetAttributes[CloudDisconnect, ReadProtected];
Protect[CloudDisconnect];
Protect[$CloudConnected];

End[]

EndPackage[]
