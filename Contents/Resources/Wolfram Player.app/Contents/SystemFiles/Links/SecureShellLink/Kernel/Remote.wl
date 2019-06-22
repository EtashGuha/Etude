(* ::Package:: *)

(* ::Chapter:: *)
(*Remote.wl*)


(* ::Subsection:: *)
(*Initialization*)


BeginPackage["SecureShellLink`"]

System`RemoteConnect
System`RemoteConnectionObject
System`RemoteRun
System`RemoteRunProcess
System`$SSHAuthentication
System`RemoteFile
RemoteCopyFile

Begin["`Private`"]

If[!StringQ[$libraryPath],
	$libraryPath = FindLibrary["libSecureShellLink"];
	If[$libraryPath === $Failed,
		Message[LibraryFunction::load, "SecureShellLink"];
	,
		If[LibraryLoad[$libraryPath] === $Failed,
			Message[LibraryFunction::load, $libraryPath]];
	];
];

Unprotect[{RemoteConnect, RemoteConnectionObject, RemoteRun, RemoteRunProcess, RemoteFile, $SSHAuthentication}];

sshEvaluate = LibraryFunctionLoad[$libraryPath, "SSHEvaluate", {Integer, UTF8String}, RawArray];
sshShellWrite = LibraryFunctionLoad[$libraryPath, "SSHShellWrite", {Integer, UTF8String}, Integer];
sshShellRead = LibraryFunctionLoad[$libraryPath, "SSHShellRead", {Integer, Boolean}, RawArray];
sshShellReadStderr = LibraryFunctionLoad[$libraryPath, "SSHShellReadStderr", {Integer, Boolean}, RawArray];
sshClose = LibraryFunctionLoad[$libraryPath, "SSHClose", {Integer}, Void];
sshStderr = LibraryFunctionLoad[$libraryPath, "SSHGetStderr", {Integer}, RawArray];
sshReturnCode = LibraryFunctionLoad[$libraryPath, "SSHGetReturnCode", {Integer}, Integer];
sshPasswordAuthenticate = LibraryFunctionLoad[$libraryPath, "SSHPasswordAuthenticate", {Integer, UTF8String, UTF8String}, Integer];
sshPasswordAuthenticateSaveKey = Quiet[LibraryFunctionLoad[$libraryPath, "SSHPasswordAuthenticate", {Integer, UTF8String, UTF8String, UTF8String}, Integer], LibraryFunction::overload];
sshKeyPairAuthenticate = LibraryFunctionLoad[$libraryPath, "SSHKeyPairAuthenticate", {Integer, UTF8String, UTF8String}, Integer];
SSHAuthenticateKeyFile = LibraryFunctionLoad[$libraryPath, "SSHAuthenticateKeyFile", {Integer, UTF8String, UTF8String, UTF8String}, Integer];
SSHAuthenticateKeyData = LibraryFunctionLoad[$libraryPath, "SSHAuthenticateKeyData", {Integer, UTF8String, UTF8String, UTF8String}, Integer];
sshInit = LibraryFunctionLoad[$libraryPath, "SSHInit", {UTF8String}, Integer];
sshShellInit = LibraryFunctionLoad[$libraryPath, "SSHShellInit", {Integer}, Integer];
copySFTP = LibraryFunctionLoad[$libraryPath, "copySFTP", {Integer, UTF8String, Boolean, UTF8String, Boolean}, Integer];
copySCP = LibraryFunctionLoad[$libraryPath, "copySCP", {Integer, UTF8String, Boolean, UTF8String, Boolean}, Integer];
sshApplyConfig = LibraryFunctionLoad[$libraryPath, "SSHApplyConfig", {UTF8String}, RawArray];

(* ::Subsection:: *)
(*Parsing and Helper Functions*)

$SSHAuthentication := <||>;

(*Gets the user part of user@host*)
getUser[hostname_String] :=First[StringCases[hostname,(x__~~"@")->x], ""]
getUser[IPAddress[hostname_String]] := First[StringCases[hostname,(x__~~"@")->x], ""]
getUser[URL[hostname_String, opts___]] := First[StringCases[hostname,(x__~~"@")->x], ""]

(*Gets the host part of user@host*)
getHost[hostname_String] := Last[StringSplit[hostname, "@"], ""]
getHost[IPAddress[hostname_String]] := Last[StringSplit[hostname, "@"], ""]
getHost[URL[hostname_String, opts___]] :=Last[StringSplit[hostname, "@"], ""]

$keyDirectory = FileNameJoin[{$UserBaseDirectory,"ApplicationData","SecureShellLink","Keys"}]

(*Checks if file is short enough to be a drive name*)
mightBeDriveName[path_String] := StringLength[path] <= 1;

(*Gets the file part of user@host:file*)
getFile[file_String] := Block[{split = StringSplit[file, ":"]},
	If[Length[split] >= 2,
		If[!mightBeDriveName[First@split], split = Drop[split, 1]]
	];
	StringRiffle[split, ":"]
]

(*Drops the file part of user@host:file *)
dropFile[file_String] := Block[{split = StringSplit[file, ":"]},
	If[Length[split] <  2, Return[""]];
	If[mightBeDriveName[First@split], Return[""]];
	First@split
]

$userNameCache = <||>;
cacheName[host_String, user_String] := AppendTo[$userNameCache, host->user]
getCachedName[host_String] := With[{user = $userNameCache[host]}, If[StringQ[user], user, ""]]

(* Picking the right username and password from what is available. *)
resolveUserPass[userhost_, userpass_String, pass_String, auths_Association] := Block[{username, password},
	username = getUser[userhost];
	password = pass;
	If[username === "",
		username = userpass;
	,
		If[password === "",
			password = userpass;
		]
	];
	If[username === "" && StringQ[auths["Username"]], username = auths["Username"]];
	If[password === "" && StringQ[auths["Password"]], password = auths["Password"]];
	If[username === "", username = getCachedName[getHost[userhost]]];
	{username, password}
]

checkSSHConfig[user_String, hostname_String] := Block[{joined, hostFinal, userFinal},
	joined = sshApplyConfig[hostname];
	If[!MatchQ[joined, _NumericArray], Return[{user, hostname}]];
	joined = FromCharacterCode[Normal[joined]];
	hostFinal = getHost[joined];
	userFinal = getUser[joined];
	If[user =!= "", userFinal = user]; (* Prioritize username from function *)
	If[userFinal === "" && StringQ[$UserName], userFinal = $UserName]; (* Default $UserName *)
	If[hostFinal === "", hostFinal = hostname]; (* Prioritize hostname from ssh config *)
	{userFinal, hostFinal}
]


(* ::Subsection:: *)
(*RemoteFile*)


Options[RemoteFile] = {Authentication :> $SSHAuthentication};

RemoteFile/: CopyFile[file_, HoldPattern[url_RemoteFile], opts:OptionsPattern[]] :=
                 SecureShellLink`RemoteCopyFile[file, url, opts]
RemoteFile/: CopyFile[HoldPattern[url_RemoteFile], file_, opts:OptionsPattern[]] :=
                 SecureShellLink`RemoteCopyFile[url, file, opts]

(* ::Subsection:: *)
(*RemoteConnectionObject*)


RemoteConnectionObject::invl = "RemoteConnectionObject is not an active object.";

RemoteConnectionObject[assoc_Association][key_] := assoc[key];
RemoteConnectionObject[assoc_Association]["Properties"] := Keys[assoc];

RemoteConnectionObject/: DeleteObject[ro : RemoteConnectionObject[assoc_Association]] := Block[{index = $RemoteUUIDTable[assoc["UUID"]]},
	If[!NumberQ[index] || index < 0, Message[DeleteObject::nim, ro];Return[$Failed]];
	Close[assoc["StandardOutput"]];
	Close[assoc["StandardInput"]];
	Close[assoc["StandardError"]];
	sshClose[index];
	KeyDropFrom[$RemoteUUIDTable, assoc["UUID"]];
]

$RemoteUUIDTable = <||>;

(*summary box for SocketObject*)
System`RemoteConnectionObject /: MakeBoxes[sock:System`RemoteConnectionObject[assoc_Association]/;KeyExistsQ[assoc["UUID"]]@$RemoteUUIDTable,StandardForm|TraditionalForm] :=
	BoxForm`ArrangeSummaryBox[
		(*first argument is the head to use*)
		System`RemoteConnectionObject,
		(*second argument is the expression*)
		System`RemoteConnectionObject[assoc],
		(*third argument is the icon to use*)
		$RemoteConnectionIcon,
		(*the next argument is the always visisble properties*)
		KeyValueMap[
			BoxForm`SummaryItem[{ToString[#1]<>": ",#2}]&,
			KeyTake[{"Host"}]@assoc
		],
		(*the next argument is the optional items that come down when the plus button is pressed*)
		KeyValueMap[
			BoxForm`SummaryItem[{ToString[#1]<>": ",#2}]&,
			Join[
				KeyTake[{"Username", "UUID"}]@assoc,
				First/@KeyTake[{"StandardInput", "StandardOutput"}]@assoc
			]
		],
		(*lastly,the display form we want to display this with*)
		StandardForm,
		(*making it interpretable allows users to take the displayed object and query things about it directly without having to save it in a variable*)
		"Interpretable"->True
	];

RemoteConnectionObject/: WriteLine[RemoteConnectionObject[assoc_Association]/;KeyExistsQ[assoc["UUID"]]@$RemoteUUIDTable, args___] := WriteLine[assoc["StandardInput"], args];
RemoteConnectionObject/: WriteString[RemoteConnectionObject[assoc_Association]/;KeyExistsQ[assoc["UUID"]]@$RemoteUUIDTable, args___] := WriteString[assoc["StandardInput"], args];
RemoteConnectionObject/: Write[RemoteConnectionObject[assoc_Association]/;KeyExistsQ[assoc["UUID"]]@$RemoteUUIDTable, args___] := Write[assoc["StandardInput"], args];
RemoteConnectionObject/: Read[RemoteConnectionObject[assoc_Association]/;KeyExistsQ[assoc["UUID"]]@$RemoteUUIDTable, args___] := Read[assoc["StandardOutput"], args];
RemoteConnectionObject/: ReadString[RemoteConnectionObject[assoc_Association]/;KeyExistsQ[assoc["UUID"]]@$RemoteUUIDTable, args___] := ReadString[assoc["StandardOutput"], args];
RemoteConnectionObject/: ReadLine[RemoteConnectionObject[assoc_Association]/;KeyExistsQ[assoc["UUID"]]@$RemoteUUIDTable, args___] := ReadLine[assoc["StandardOutput"], args];


(* ::Subsection:: *)
(*Connected Streams*)


If[MemberQ[$InputStreamMethods, "RemoteConnection"],
	RemoveInputStreamMethod["RemoteConnection"]
];
DefineInputStreamMethod["RemoteConnection",{
	"ConstructorFunction"->(remoteInputCreate[#1]&),
	"CloseFunction"->(remoteStreamClose[#1]&),
	"ReadFunction"->(remoteStreamRead[#1,#2]&),
	"WaitForInputFunction"->(remoteStreamWait[#1]&),
	"SeekableQFunction"->({False,#1}&),
	"EndOfFileQFunction"->(remoteStreamEOF[#1]&)
}];

If[MemberQ[$OutputStreamMethods, "RemoteConnection"],
	RemoveOutputStreamMethod["RemoteConnection"]
];
DefineOutputStreamMethod["RemoteConnection",{
	"ConstructorFunction"->(remoteOutputCreate[#1]&),
	"CloseFunction"->(remoteStreamClose[#1]&),
	"WriteFunction"->(remoteStreamWrite[#1,#2]&),
	"FlushFunction"->(remoteStreamFlush[#1]&)
}];

$inBuffers = <||>
$dataThisRead = <||>;
$outBuffers = <||>;

remoteOutputCreate[b_] := (
	$outBuffers[b] = "";
	{True, b}
);

remoteInputCreate[b_] := (
	$inBuffers[b] = {};
	$dataThisRead[b] = False;
	{True, b}
);


remoteStreamFlush[b_] := (
	sshShellWrite[SSHOutputStreamID[b],$outBuffers[b]];
	$outBuffers[b] = "";
	{True,b}
);

remoteStreamWrite[b_,data_] := (
	$outBuffers[b] = $outBuffers[b] <> FromCharacterCode[data];
	{Length[data],b}
);

remoteStreamRead[b_,num_] := Block[{data={}, toRead = num, dataLen, newData, newLen, buf},
	dataLen = Length[$inBuffers[b]];
	If[dataLen > 0,
		If[dataLen <= num,
			data = $inBuffers[b];
			$inBuffers[b] = {};
			toRead -= dataLen
			,
			data = Take[$inBuffers[b], num];
			$inBuffers[b] = Drop[$inBuffers[b], num];
			toRead = 0;
			Return[{data,b}]
		]
	];
	If[stderrQ[b],
		newData = sshShellReadStderr[SSHInputStreamID[b], False];
		,
		newData = sshShellRead[SSHInputStreamID[b], False];
	]
	If[!MatchQ[newData, _NumericArray | _List], Return[$Failed]];
	newData = Normal[newData];
	newLen = Length[newData];
	If[newLen > toRead,
		$inBuffers[b] = Drop[newData, toRead];
		newData = Take[newData, toRead];
	];
	data=Join[data, newData];
	If[Length[data]>0,$dataThisRead[b]=True];
	{data, b}
];

remoteStreamClose[b_] := {True, b}

remoteStreamWait[b_] := Block[{ready=False, newData},
	If[Length[$inBuffers[b]] > 0,
		Return[{True, b}];
	];
	If[stderrQ[b],
		newData = sshShellReadStderr[SSHInputStreamID[b], True];
		,
		newData = sshShellRead[SSHInputStreamID[b], True];
	]
	If[!MatchQ[newData, _NumericArray | _List], Return[{False,b}]];
	$inBuffers[b] = Normal[newData];
	If[Length[$inBuffers[b]] > 0, ready = True];
	{ready, b}
];

remoteStreamEOF[b_] := Block[{eof=False, newData},
	If[Length[$inBuffers[b]] > 0,
		Return[{False, b}];
	];
	If[stderrQ[b],
		newData = sshShellReadStderr[SSHInputStreamID[b], False];
		,
		newData = sshShellRead[SSHInputStreamID[b], False];
	]
	If[!MatchQ[newData, _NumericArray | _List], Return[{True,b}]];
	$inBuffers[b] = Normal[newData];
	If[Length[$inBuffers[b]] == 0 && $dataThisRead[b], $dataThisRead[b]=False; eof = True];
	{eof, b}
];

stderrQ[str_String] := StringStartsQ[str, "stderr-"];
SSHOutputStreamID[str_String] := $RemoteUUIDTable[StringDrop[str, 6]];
SSHInputStreamID[str_String] := $RemoteUUIDTable[StringDrop[str, 7]]

If[TrueQ[$CloudEvaluation] && !TrueQ[Lookup[CloudSystem`KernelInitialize`$ConfigurationProperties, "AllowRemoteRunFunctionality"]],
   (* Dummy functions explaining not available on cloud *)
    RemoteConnect[___] := (Message[RemoteConnect::cloudf, HoldForm@RemoteConnect]; $Failed);
    RemoteRun[___] := (Message[RemoteRun::cloudf, HoldForm@RemoteRun]; $Failed);
    RemoteRunProcess[___] := (Message[RemoteRunProcess::cloudf, HoldForm@RemoteRunProcess]; $Failed);
,



	RemoteConnect::noAuth = "Username or password not provided.";
	RemoteConnect::invAuth = "Username and password combination invalid.";
	RemoteConnect::init = "Unable to start SSH connection.";
	RemoteConnect::cnct = "Unable to establish ssh session with host computer.";
	RemoteConnect::addr = "Cannot connect to hostname `1`.";
	RemoteConnect::nokey = "The private key file `1` could not be found.";
	Macros`SetArgumentCount[RemoteConnect, 1, 3];

	Options[RemoteConnect] := {Authentication :> $SSHAuthentication, RemoteAuthorizationCaching->False};

	RemoteConnect[userhost : _String | _IPAddress | _URL, userpass_String:"", pass_String:"", opts : OptionsPattern[]] :=
	Block[{auths, username = "", password = "", result, index, hostname, authenticated = -1, cacheAuth, uuid,
			stdinStream, stdoutStream, stdError, triedUser, rememberMe, pem, pemPassword, tryPemWithPass, tryPemWithAll},
		auths = OptionValue[Authentication];
		cacheAuth = TrueQ@OptionValue[RemoteAuthorizationCaching];
		{username, password} = resolveUserPass[userhost, userpass, pass, auths];
		hostname = getHost[userhost];
		{username, hostname} = checkSSHConfig[username, hostname];
		index = sshInit[hostname];
		If[index < 0,
			If[index === -3,
				Message[RemoteConnect::addr, hostname],
				Message[RemoteConnect::init]
			];
			Return[$Failed]
		];

		(*Try SSH Keys*)
		pem = Lookup[auths,"SSHKey"];
		If[!MissingQ[pem],
			pemPassword = Lookup[auths,"SSHKeyPassword", ""];
			tryPemWithPass[singlePem_] := Function[singlePass,
					Block[{passData = singlePass},
						If[authenticated > 0, Return[]];
						If[passData === None, passData = ""];
						(*Handle File[] wrapper on passwords*)
						If[MatchQ[passData, File[_String, ___]],
							If[FileExistsQ[First@passData],
								passData = StringTrim[ReadString[passData]];
							];
						];
						If[StringQ[passData],
					Which[
						MatchQ[singlePem, _File],
									authenticated = SSHAuthenticateKeyFile[index, username, singlePem[[1]], passData],
						StringQ[singlePem],
									authenticated = SSHAuthenticateKeyData[index, username, singlePem, passData]
							]
					]
				]
			];
			tryPemWithAll = Function[singlePem,
				Which[
					ListQ[pemPassword],
						tryPemWithPass[singlePem]/@pemPassword,
					MatchQ[pemPassword, _String | File[_String,___]],
						tryPemWithPass[singlePem]@pemPassword
				]
			];
			Which[
				ListQ[pem],
					tryPemWithAll/@pem,
				MatchQ[pem, _String|_File],
					tryPemWithAll@pem
			]
		];

		If[authenticated <= 0 && username =!= "",
			triedUser = username;
			authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
		];
		If[authenticated <= 0 && (username === "" || password === ""),
			result = loginDialog[username, hostname];
			If[MatchQ[result, {_String, _String, _}],
				{username, password, rememberMe} = result,
				Message[RemoteConnect::noAuth];
				Return[$Failed];
			];
			cacheAuth = rememberMe || cacheAuth
		];
		If[authenticated <= 0 && (username =!= triedUser),
			authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
		];
		If[authenticated <= 0,
			If[cacheAuth,
				authenticated = sshPasswordAuthenticateSaveKey[index, username, password, $keyDirectory]
			,
				authenticated = sshPasswordAuthenticate[index, username, password]
			];
		];
		If[authenticated <= 0,
			Message[RemoteConnect::invAuth];
			Return[$Failed];
		];
		cacheName[hostname, username];

		If[sshShellInit[index] <= 0,
			Message[RemoteConnect::cnct];
			Return[$Failed];
		];

		uuid = CreateUUID[];
		$RemoteUUIDTable[uuid] = index;
		stdoutStream = OpenRead["stdout-" <> uuid, Method->"RemoteConnection", BinaryFormat->True,AppendCheck->True];
		stderrStream = OpenRead["stderr-" <> uuid, Method->"RemoteConnection", BinaryFormat->True,AppendCheck->True];
		stdinStream = OpenWrite["stdin-" <> uuid, Method->"RemoteConnection", BinaryFormat->True];
		RemoteConnectionObject[<|"Host" -> hostname, "Username" -> username, "UUID"->uuid, "StandardOutput"->stdoutStream, "StandardInput"->stdinStream, "StandardError"->stderrStream|>]
	];


	RemoteRunProcess::noAuth = "Username or password not provided.";
	RemoteRunProcess::invAuth = "Username and password combination invalid.";
	RemoteRunProcess::init = "Unable to start SSH connection.";
	RemoteRunProcess::cnct = "Unable to connect to remote machine.";
	RemoteRunProcess::addr = "Cannot connect to hostname `1`.";
	RemoteRunProcess::invl = "RemoteConnectionObject is invalid.";
	RemoteRunProcess::pipe = "Command contains redirection. Input ignored.";
	Macros`SetArgumentCount[RemoteRunProcess, 2, 4];

	Options[RemoteRunProcess] := {Authentication :> $SSHAuthentication, RemoteAuthorizationCaching->False};

	RemoteRunProcess[host : _RemoteConnectionObject | _String | _IPAddress | _URL, commands_, opts:OptionsPattern[]] :=
		RemoteRunProcess[host, commands, All, "", opts];

	RemoteRunProcess[host : _RemoteConnectionObject | _String | _IPAddress | _URL, commands_,
			outputType : "StandardOutput" | "StandardError" | "ExitCode" | All, input_:"", opts:OptionsPattern[]] :=
	Block[{result, joinedCommands, stringInput},
		If[ListQ[commands],
			joinedCommands = StringRiffle[ToString/@commands, " "];
			,
			joinedCommands = ToString[commands];
		];
		stringInput = ToString[input];
		result = RemoteRunProcess[host, joinedCommands, All, stringInput, opts];
		If[outputType === All, Return[result]];
		If[AssociationQ[result], result = result[outputType]];
		If[StringQ[result] || IntegerQ[result], result, $Failed]
	];

	RemoteRunProcess[userhost : _String | _IPAddress | _URL, command_String, All, input_String, opts : OptionsPattern[]] :=
	Block[{hostname, auths, username, password, result, stdout, stderr, eCode, index = -1,
			authenticated = 0, cacheAuth, triedUser, commandInput=command, rememberMe, pem, pemPassword, tryPemWithPass, tryPemWithAll},
		auths = OptionValue[Authentication];
		cacheAuth = TrueQ@OptionValue[RemoteAuthorizationCaching];
		{username, password} = resolveUserPass[userhost, "", "", auths];
		hostname = getHost[userhost];
		{username, hostname} = checkSSHConfig[username, hostname];

		(*Verify ssh and initial connection*)
		index = sshInit[hostname];
		If[index < 0,
			If[index === -3,
				Message[RemoteRunProcess::addr, hostname],
				Message[RemoteRunProcess::init]
			];
			Return[$Failed]
		];

		(*Try SSH Keys*)
		pem = Lookup[auths,"SSHKey"];
		If[!MissingQ[pem],
			pemPassword = Lookup[auths,"SSHKeyPassword", ""];
			tryPemWithPass[singlePem_] := Function[singlePass,
					Block[{passData = singlePass},
						If[authenticated > 0, Return[]];
						If[passData === None, passData = ""];
						(*Handle File[] wrapper on passwords*)
						If[MatchQ[passData, File[_String, ___]],
							If[FileExistsQ[First@passData],
								passData = StringTrim[ReadString[passData]];
							];
						];
						If[StringQ[passData],
						Which[
							MatchQ[singlePem, _File],
									authenticated = SSHAuthenticateKeyFile[index, username, singlePem[[1]], passData],
							StringQ[singlePem],
									authenticated = SSHAuthenticateKeyData[index, username, singlePem, passData]
							]
						]
					]
				];
			tryPemWithAll = Function[singlePem,
				Which[
					ListQ[pemPassword],
						tryPemWithPass[singlePem]/@pemPassword,
					MatchQ[pemPassword, _String | File[_String,___]],
						tryPemWithPass[singlePem]@pemPassword
				]
			];
			Which[
				ListQ[pem],
					tryPemWithAll/@pem,
				MatchQ[pem, _String|_File],
					tryPemWithAll@pem
			]
		];

		(*Try default SSH Key*)
		If[authenticated <= 0 && username =!= "",
			triedUser = username;
			authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
		];

		(*Login dialog if needed*)
		If[authenticated <= 0 && (username === "" || password === ""),
			result = loginDialog[username, hostname];
			If[MatchQ[result, {_String, _String, _}],
				{username, password, rememberMe} = result,
				Message[RemoteRunProcess::noAuth];
				Return[$Failed];
			];
			cacheAuth = rememberMe || cacheAuth
		];

		(* With username provided, try default ssh key again. *)
		If[authenticated <= 0 && (username =!= triedUser),
			authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
		];

		(* Try password authentication. *)
		If[authenticated <= 0,
			If[cacheAuth,
				authenticated = sshPasswordAuthenticateSaveKey[index, username, password, $keyDirectory]
			,
				authenticated = sshPasswordAuthenticate[index, username, password]
			]
		];
		If[authenticated <= 0,
			Message[RemoteRunProcess::invAuth];
			Return[$Failed];
		];
		cacheName[hostname, username];

		If[sshShellInit[index] <= 0,
			Message[RemoteRunProcess::cnct];
			Return[$Failed];
		];

		If[input =!= "",
			If[StringContainsQ[command, "<"] || StringContainsQ[command, "|"],
				Message[RemoteRunProcess::pipe]
				,
				commandInput = StringJoin[command," <<'EOF_MARKER'\n",input,"\nEOF_MARKER"]
			]
		];

		stdout = sshEvaluate[index, commandInput];
		If[!MatchQ[stdout, _NumericArray | _List], Message[RemoteRunProcess::invl];Return[$Failed]];
		stdout = FromCharacterCode[Normal[stdout]];

		stderr = sshStderr[index];
		If[!MatchQ[stderr, _NumericArray | _List], Message[RemoteRunProcess::invl];Return[$Failed]];
		stderr = FromCharacterCode[Normal[stderr]];

		eCode = sshReturnCode[index];
		If[!IntegerQ[eCode] || eCode < 0, eCode = Missing[]];
		sshClose[index];
		<|"ExitCode"->eCode, "StandardOutput"->stdout, "StandardError"->stderr|>
	];

	RemoteRunProcess[ro : RemoteConnectionObject[param_], command_String, All, input_String, opts : OptionsPattern[]] := 
	Block[{result, index, stdout, stderr, eCode, commandInput=command},
		index = $RemoteUUIDTable[param["UUID"]];
		If[! NumberQ[index],
			Message[RemoteRunProcess::invl];
			Return[$Failed]
		];

		If[input =!= "",
			If[StringContainsQ[command, "<"] || StringContainsQ[command, "|"],
				Message[RemoteRunProcess::pipe]
				,
				commandInput = StringJoin[command," <<'EOF_MARKER'\n",input,"\nEOF_MARKER"]
			]
		];

		stdout = sshEvaluate[index, commandInput];
		If[!MatchQ[stdout, _NumericArray | _List], Message[RemoteRunProcess::invl];Return[$Failed]];
		stdout = FromCharacterCode[Normal[stdout]];

		stderr = sshStderr[index];
		If[!MatchQ[stderr, _NumericArray | _List], Message[RemoteRunProcess::invl];Return[$Failed]];
		stderr = FromCharacterCode[Normal[stderr]];

		eCode = sshReturnCode[index];
		If[!IntegerQ[eCode] || eCode < 0, eCode = Missing[]];
		<|"ExitCode"->eCode, "StandardOutput"->stdout, "StandardError"->stderr|>
	];



	RemoteRun::noAuth = "Username or password not provided.";
	RemoteRun::invAuth = "Username and password combination invalid.";
	RemoteRun::init = "Unable to start SSH connection.";
	RemoteRun::cnct = "Unable to connect to remote machine.";
	RemoteRun::addr = "Cannot connect to hostname `1`.";
	RemoteRun::invl = "RemoteConnectionObject is invalid.";
	Macros`SetArgumentCount[RemoteRun, 2];

	Options[RemoteRun] := {Authentication :> $SSHAuthentication, RemoteAuthorizationCaching->False};
	RemoteRun[userhost:_String|_IPAddress|_URL, command_String, opts : OptionsPattern[]] := 
	  Block[{hostname, username, password, auths, result, index, authenticated = -1,
			cacheAuth, triedUser, eCode, rememberMe, pem, pemPassword, tryPemWithPass, tryPemWithAll},
		auths = OptionValue[Authentication];
		cacheAuth = TrueQ@OptionValue[RemoteAuthorizationCaching];
		{username, password} = resolveUserPass[userhost, "", "", auths];
		hostname = getHost[userhost];
		{username, hostname} = checkSSHConfig[username, hostname];
		index = sshInit[hostname];
		If[index < 0,
			If[index === -3,
				Message[RemoteRun::addr, hostname],
				Message[RemoteRun::init]
			];
			Return[$Failed]
		];

		(*Try SSH Keys*)
		pem = Lookup[auths,"SSHKey"];
		If[!MissingQ[pem],
			pemPassword = Lookup[auths,"SSHKeyPassword", ""];

			tryPemWithPass[singlePem_] := Function[singlePass,
					Block[{passData = singlePass},
						If[authenticated > 0, Return[]];
						If[passData === None, passData = ""];
						(*Handle File[] wrapper on passwords*)
						If[MatchQ[passData, File[_String, ___]],
							If[FileExistsQ[First@passData],
								passData = StringTrim[ReadString[passData]];
							];
						];
						If[StringQ[passData],
						Which[
							MatchQ[singlePem, _File],
									authenticated = SSHAuthenticateKeyFile[index, username, singlePem[[1]], passData],
							StringQ[singlePem],
									authenticated = SSHAuthenticateKeyData[index, username, singlePem, passData]
							]
						]
					]
				];
			tryPemWithAll = Function[singlePem,
				Which[
					ListQ[pemPassword],
						tryPemWithPass[singlePem]/@pemPassword,
					MatchQ[pemPassword, _String | File[_String,___]],
						tryPemWithPass[singlePem]@pemPassword
				]
			];
			Which[
				ListQ[pem],
					tryPemWithAll/@pem,
				MatchQ[pem, _String|_File],
					tryPemWithAll@pem
			]
		];

		If[authenticated <= 0 && username =!= "",
			triedUser=username;
			authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
		];
		If[authenticated <= 0 && (username === "" || password === ""),
			result = loginDialog[username, hostname];
			If[MatchQ[result, {_String, _String, _}],
				{username, password, rememberMe} = result,
				Message[RemoteRun::noAuth];
				Return[$Failed]
			];
			cacheAuth = rememberMe || cacheAuth
		];
		If[authenticated <= 0 && (username =!= triedUser),
			authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory]
		];
		If[authenticated <= 0,
			If[cacheAuth,
				authenticated = sshPasswordAuthenticateSaveKey[index, username, password, $keyDirectory]
			,
				authenticated = sshPasswordAuthenticate[index, username, password]
			]
		];
		If[authenticated <= 0,
			Message[RemoteRun::invAuth];
			Return[$Failed]
		];
		cacheName[hostname, username];

		If[(sshShellInit[index]) <= 0,
			Message[RemoteRun::cnct];
			Return[$Failed]
		];

		sshEvaluate[index, command];
		eCode = sshReturnCode[index];
		If[!IntegerQ[eCode] || eCode < 0, eCode = $Failed];
		sshClose[index];
		eCode
	];

	RemoteRun[ro : RemoteConnectionObject[param_], command_String,
		opts : OptionsPattern[]] := Block[{result, index, eCode},
		index = $RemoteUUIDTable[param["UUID"]];
		If[!NumberQ[index],
			Message[RemoteRun::invl];
			Return[$Failed]
		];
		sshEvaluate[index, command];
		eCode = sshReturnCode[index];
		If[!IntegerQ[eCode] || eCode < 0, eCode = $Failed];
		eCode
	];

	Unprotect[URL];

    RemoteCopyFileOut[parsed_Association, file_, opts:OptionsPattern[]] := RemoteCopyFile[parsed, file, opts];
    RemoteCopyFileInto[file_, parsed_Association, opts:OptionsPattern[]] := RemoteCopyFile[file, parsed, opts];

	Options[RemoteCopyFile] := {Authentication :> $SSHAuthentication, OverwriteTarget->True};

	RemoteCopyFile[source_, RemoteFile[destURL_String | URL[destURL_String, innerURLOpts:OptionsPattern[]]
					, remoteOpts:OptionsPattern[]]|URL[destURL_String, outerURLOpts:OptionsPattern[]], opts:OptionsPattern[]] := Block[
					{components, combinedOptions},
		components = URLParse[destURL];
		combinedOptions = Replace[Association[innerURLOpts, remoteOpts, outerURLOpts, opts], Association -> Sequence, {1}, Heads -> True];
		RemoteCopyFile[source, components, combinedOptions]
	];

	RemoteCopyFile[source_, parsed_Association, opts:OptionsPattern[]] := Block[
			{components, sourceFile, destFile, result=$Failed, usingTemp=False},
		sourceFile = source;
		If[MatchQ[source, File[_String]], sourceFile = First[source]];
		If[!MatchQ[source, _String|File[_String]],
			sourceFile = CreateFile[];
			usingTemp = True;
			result=CopyFile[source,sourceFile, OverwriteTarget->True];
			If[!StringQ[result], Return[result]]
			,
			sourceFile = ExpandFileName[sourceFile]
		];

		components = Replace[parsed, None->"", 1];
		If[FailureQ[components], Return[$Failed]];
		destFile = FileNameJoin[components["Path"], OperatingSystem->"MacOSX"];
		If[StringStartsQ[destFile, "/~/"], destFile = StringDrop[destFile,3]];
		result = Switch[components["Scheme"],
			"sftp",
			RemoteSFTP[components["User"], components["Domain"], False, sourceFile, True, destFile, opts],
			"scp",
			RemoteSCP[components["User"], components["Domain"], False, sourceFile, True, destFile, opts],
			True,
			Message[URL::noscheme];$Failed
		];
		If[usingTemp,
			DeleteFile[sourceFile];
		];
		If[!FailureQ[result],
			result = URL[URLBuild[parsed]]
		];
		result
	];

	RemoteCopyFile[RemoteFile[sourceURL_String | URL[sourceURL_String, innerURLOpts:OptionsPattern[]], remoteOpts:OptionsPattern[]]
				| URL[sourceURL_String, outerURLOpts:OptionsPattern[]], dest_, opts:OptionsPattern[]] := Block[
		{components, combinedOptions},
		components = URLParse[sourceURL];
		combinedOptions = Replace[Association[innerURLOpts, remoteOpts, outerURLOpts, opts], Association -> Sequence, {1}, Heads -> True];
		RemoteCopyFile[components, dest, combinedOptions]
	];

	RemoteCopyFile[parsed_Association, dest_, opts:OptionsPattern[]] := Block[
			{components, destFile, sourceFile, result=$Failed, usingTemp = False},
		destFile = dest;
		If[MatchQ[dest, File[_String]], destFile = First[dest]];
		If[!MatchQ[dest, _String|File[_String]],
			destFile = CreateFile[];
			usingTemp = True;
		,
			destFile = ExpandFileName[destFile]
		];

		components = Replace[parsed, None->"", 1];
		sourceFile = FileNameJoin[components["Path"], OperatingSystem->"MacOSX"];
		If[StringStartsQ[sourceFile, "/~/"], sourceFile = StringDrop[sourceFile,3]];
		If[FailureQ[components], Return[$Failed]];
		result = Switch[components["Scheme"],
			"sftp",
			RemoteSFTP[components["User"], components["Domain"], True, sourceFile, False, destFile, opts],
			"scp",
			RemoteSCP[components["User"], components["Domain"], True, sourceFile, False, destFile, opts],
			True,
			Message[URL::noscheme];$Failed
		];
		If[!StringQ[result],
			DeleteFile[destFile];
			Return[$Failed]
		];
		If[usingTemp,
			result = CopyFile[destFile, dest, FilterRules[{opts}, Options[CopyFile]]];
			DeleteFile[destFile];
		];
		result
	];

	URL::noscheme = "URI scheme is not supported.";
	URL::noAuth = "Username or password not provided.";
	URL::invAuth = "Username and password combination invalid.";
	URL::init = "Unable to start SSH connection.";
	URL::addr = "Cannot connect to hostname `1`.";
	URL::openl = "Cannot open local file: `1`.";
	URL::openr = "Cannot open remote file: `1`.";

	Protect[URL];

]; (* End bracket of section defined as not available in cloud. *)

Options[RemoteSFTP] := {Authentication :> $SSHAuthentication, OverwriteTarget->True, RemoteAuthorizationCaching->False};

RemoteSFTP[userpass_String, domain_String, sourceIsRemote : True|False, source_String,
			destIsRemote : True|False, dest_String, opts : OptionsPattern[]] :=  Block[
	{hostname, auths, username="", password="", result,copyStatus, index, authenticated = -1,
			cacheAuth, triedUser, split, pem, pemPassword, tryPemWithPass, tryPemWithAll},

	split = StringSplit[userpass, ":"];
	If[Length[split] >= 2,
		username = split[[1]];
		password = split[[2]]
		,
		username = userpass
	];
	auths = OptionValue[Authentication];
	cacheAuth = TrueQ@OptionValue[RemoteAuthorizationCaching];
	{username, password} = resolveUserPass[domain, username, password, auths];
	hostname = domain;
	{username, hostname} = checkSSHConfig[username, hostname];
	index = sshInit[hostname];
	If[index < 0,
		If[index === -3,
			Message[URL::addr, hostname],
			Message[URL::init]
		];
		Return[$Failed];
	];

	(*Try SSH Keys*)
	pem = Lookup[auths,"SSHKey"];
	If[!MissingQ[pem],
		pemPassword = Lookup[auths,"SSHKeyPassword", ""];
		tryPemWithPass[singlePem_] := Function[singlePass,
				Block[{passData = singlePass},
					If[authenticated > 0, Return[]];
					If[passData === None, passData = ""];
					(*Handle File[] wrapper on passwords*)
					If[MatchQ[passData, File[_String, ___]],
						If[FileExistsQ[First@passData],
							passData = StringTrim[ReadString[passData]];
						];
					];
					If[StringQ[passData],
					Which[
						MatchQ[singlePem, _File],
								authenticated = SSHAuthenticateKeyFile[index, username, singlePem[[1]], passData],
						StringQ[singlePem],
								authenticated = SSHAuthenticateKeyData[index, username, singlePem, passData]
						]
					]
				]
			];
		tryPemWithAll = Function[singlePem,
			Which[
				ListQ[pemPassword],
					tryPemWithPass[singlePem]/@pemPassword,
				MatchQ[pemPassword, _String | File[_String,___]],
					tryPemWithPass[singlePem]@pemPassword
			]
		];
		Which[
			ListQ[pem],
				tryPemWithAll/@pem,
			MatchQ[pem, _String|_File],
				tryPemWithAll@pem
		]
	];
	
	If[authenticated <= 0 && username =!= "",
		triedUser = username;
		authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
	];
	If[authenticated <= 0 && (username === "" || password === ""),
		result = loginDialog[username, hostname];
		If[MatchQ[result, {_String, _String, _}],
			{username, password, rememberMe} = result,
			Message[URL::noAuth];
			Return[$Failed];
		];
		cacheAuth = rememberMe || cacheAuth
	];
	If[authenticated <= 0 && (username =!= triedUser),
		authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
	];
	If[authenticated <= 0,
		If[cacheAuth,
			authenticated = sshPasswordAuthenticateSaveKey[index, username, password, $keyDirectory]
		,
			authenticated = sshPasswordAuthenticate[index, username, password]
		]
	];
	If[authenticated <= 0,
		Message[URL::invAuth];
		Return[$Failed]
	];
	cacheName[hostname, username];
	
	copyStatus = copySFTP[index, source, sourceIsRemote, dest, destIsRemote];
	If[copyStatus === -1 || copyStatus === -2, Message[URL::init]];
	If[copyStatus === -3, Message[URL::openr, source]];
	If[copyStatus === -4, Message[URL::openl, source]];
	If[copyStatus === -5, Message[URL::openr, dest]];
	If[copyStatus === -6, Message[URL::openl, dest]];
	If[copyStatus < 0, Return[$Failed]];
	sshClose[index];
	dest
]

Options[RemoteSCP] := {Authentication :> $SSHAuthentication, OverwriteTarget->True, RemoteAuthorizationCaching->False};

RemoteSCP[userpass_String, domain_String, sourceIsRemote : True|False, source_String,
		destIsRemote : True|False, dest_String, opts : OptionsPattern[]] := Block[
		{hostname, auths, username="", password="", result, copyStatus, index,
				authenticated = -1, cacheAuth, triedUser, split, pem, pemPassword, tryPemWithPass, tryPemWithAll},
	split = StringSplit[userpass, ":"];
	If[Length[split] >= 2,
		username = split[[1]];
		password = split[[2]]
		,
		username = userpass
	];
	auths = OptionValue[Authentication];
	cacheAuth = TrueQ@OptionValue[RemoteAuthorizationCaching];
	{username, password} = resolveUserPass[domain, username, password, auths];
	hostname = domain;
	{username, hostname} = checkSSHConfig[username, hostname];
	index = sshInit[hostname];
	If[index < 0,
		If[index === -3,
			Message[URL::addr, hostname],
			Message[URL::init]
		];
		Return[$Failed];
	];

	(*Try SSH Keys*)
	pem = Lookup[auths,"SSHKey"];
	If[!MissingQ[pem],
		pemPassword = Lookup[auths,"SSHKeyPassword", ""];
		tryPemWithPass[singlePem_] := Function[singlePass,
				Block[{passData = singlePass},
					If[authenticated > 0, Return[]];
					If[passData === None, passData = ""];
					(*Handle File[] wrapper on passwords*)
					If[MatchQ[passData, File[_String, ___]],
						If[FileExistsQ[First@passData],
							passData = StringTrim[ReadString[passData]];
						];
					];
					If[StringQ[passData],
					Which[
						MatchQ[singlePem, _File],
								authenticated = SSHAuthenticateKeyFile[index, username, singlePem[[1]], passData],
						StringQ[singlePem],
								authenticated = SSHAuthenticateKeyData[index, username, singlePem, passData]
						]
					]
				]
			];
		tryPemWithAll = Function[singlePem,
			Which[
				ListQ[pemPassword],
					tryPemWithPass[singlePem]/@pemPassword,
				MatchQ[pemPassword, _String | File[_String,___]],
					tryPemWithPass[singlePem]@pemPassword
			]
		];
		Which[
			ListQ[pem],
				tryPemWithAll/@pem,
			MatchQ[pem, _String|_File],
				tryPemWithAll@pem
		]
	];
	If[authenticated <= 0 && username =!= "",
		triedUser = username;
		authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
	];
	If[authenticated <= 0 && (username === "" || password === ""),
		result = loginDialog[username, hostname];
		If[MatchQ[result, {_String, _String, _}],
			{username, password, rememberMe} = result,
			Message[URL::noAuth];
			Return[$Failed];
		];
		cacheAuth = rememberMe || cacheAuth
	];
	If[authenticated <= 0 && (username =!= triedUser),
		authenticated = sshKeyPairAuthenticate[index, username, $keyDirectory];
	];
	If[authenticated <= 0,
		If[cacheAuth,
			authenticated = sshPasswordAuthenticateSaveKey[index, username, password, $keyDirectory]
		,
			authenticated = sshPasswordAuthenticate[index, username, password]
		]
	];
	If[authenticated <= 0,
		Message[URL::invAuth];
		Return[$Failed]
	];

	cacheName[hostname, username];
	
	copyStatus = copySCP[index, source, sourceIsRemote, dest, destIsRemote];
	If[copyStatus === -1 || copyStatus === -2, Message[URL::init]];
	If[copyStatus === -3, Message[URL::openr, source]];
	If[copyStatus === -4, Message[URL::openl, source]];
	If[copyStatus === -5, Message[URL::openr, dest]];
	If[copyStatus === -6, Message[URL::openl, dest]];
	If[copyStatus < 0, Return[$Failed]];
	sshClose[index];
	dest
]


(* ::Subsection:: *)
(*RemoteConnection Icon*)


$RemoteConnectionIcon=Graphics[{Thickness[0.03333333333333333], {FaceForm[{RGBColor[0.28600000000000003`, 0.28600000000000003`, 0.28600000000000003`], Opacity[1.]}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{10.810500000000001`, 8.728699999999998}, {10.800500000000001`, 8.037699999999997}, {10.360500000000002`, 8.888699999999998}, {10.3535, 9.415699999999998}}, {{13.2985, 10.305699999999998`}, {14.0195, 10.015699999999999`}, {13.217500000000001`, 9.915699999999998}, {12.576500000000001`, 10.103699999999998`}}, {{11.177500000000002`, 8.058699999999998}, {11.187500000000002`, 8.745699999999998}, {11.6415, 9.418699999999998}, {11.604500000000002`, 8.876699999999998}}, {{10.996500000000001`, 9.151699999999998}, {10.3825, 10.074699999999998`}, {10.995500000000002`, 10.943699999999998`}, {11.634500000000001`, 10.098699999999997`}}, {{10.697500000000002`, 11.179699999999997`}, {10.078500000000002`, 10.301699999999999`}, {8.989500000000001, 10.677699999999998`}, {9.695500000000001, 11.594699999999998`}}, {{9.982500000000002, 8.955699999999998}, {8.868500000000001, 8.360699999999998}, {9.0645, 9.645699999999998}, {9.970500000000001, 9.835699999999997}}, {{8.6615, 10.378699999999998`}, {9.424500000000002, 10.115699999999999`}, {8.8945, 10.003699999999998`}, {7.972500000000002, 10.153699999999997`}}, {{8.572500000000002, 11.191699999999997`}, {9.047500000000001, 11.384699999999999`}, {8.557500000000001, 10.749699999999997`}, {7.8915000000000015`, 10.531699999999997`}}, {{9.063500000000001, 12.893699999999997`}, {8.966500000000002, 13.872699999999998`}, {9.392500000000002, 13.291699999999999`}, {9.421500000000002, 12.4287}}, {{8.803500000000001, 12.608699999999999`}, {9.334500000000002, 11.916699999999999`}, {8.5165, 11.583699999999997`}, {7.732500000000002, 12.411699999999998`}}, {{9.707500000000001, 13.5047}, {9.278500000000001, 14.089699999999997`}, {10.1005, 13.735699999999998`}, {10.469500000000002`, 13.1877}}, {{10.7985, 11.553699999999997`}, {9.823500000000001, 11.957699999999999`}, {9.7865, 13.048699999999997`}, {10.7855, 12.632699999999998`}}, {{11.0585, 14.900699999999997`}, {11.5995, 13.892699999999998`}, {10.9915, 13.092699999999997`}, {10.4495, 13.897699999999997`}}, {{12.227500000000001`, 13.087699999999998`}, {12.213500000000002`, 11.865699999999999`}, {11.1765, 11.528699999999997`}, {11.1635, 12.634699999999999`}}, {{11.5715, 13.226699999999997`}, {11.9425, 13.714699999999999`}, {12.732500000000002`, 14.032699999999998`}, {12.328500000000002`, 13.549699999999998`}}, {{12.604500000000002`, 12.316699999999997`}, {12.614500000000001`, 13.298699999999998`}, {13.117500000000001`, 13.8997}, {13.0355, 12.907699999999998`}}, {{13.5545, 11.088699999999998`}, {14.134500000000001`, 10.383699999999997`}, {13.450500000000002`, 10.658699999999998`}, {12.957500000000001`, 11.313699999999997`}}, {{13.608500000000001`, 11.479699999999998`}, {12.7125, 11.817699999999999`}, {13.299500000000002`, 12.623699999999998`}, {14.3815, 12.410699999999999`}}, {{12.332500000000001`, 11.499699999999997`}, {12.992500000000001`, 10.619699999999998`}, {11.9395, 10.324699999999998`}, {11.3045, 11.165699999999998`}}, {{13.040500000000002`, 8.355699999999999}, {11.986500000000001`, 8.933699999999998}, {12.0485, 9.856699999999998}, {13.0165, 9.573699999999999}}, CompressedData["
1:eJxTTMoPSmViYGAQAWIQnQYCbnoOWhJTr3B2KDscPLXQddtlbYdHZlIHohvU
HDbPfb/8mLaew52tLTUXgjUd5gG53te1HMqrluo4/9F0yN5TMllCRdthWfgp
oyMTdR0kWcL4dC+pO7wCGWCt4+DxsEpknbqaw7cniQuvheg7sDdOde72UXHw
+dwXXOKi4+CU8PSC0m8Fhwfck1c2bdR1eFdjbxoXpeiwiHEPq9AXTQfrLSfK
9t2XdvCTE8vyBdqvtwnooDQFB5BzNu9Uc7ivwtY4lVna4WzHpXsPXis7RFoC
NcQpOrDbzg6d763s0OUItOCQgsPbk4ed1q6Uc5CNSrG+r6/iUPfbquCchqLD
XqDzWb6pOvjfkq5JLJJ2iO0/9FVjj7oD0HUq0+UVHVRu/6zLuqPl8Ptb6YM5
F+Uctn/+e6XipJbDOwtX9yJJZQcA1/2Ndw==
"]}]}, {FaceForm[{RGBColor[0.608, 0.608, 0.608], Opacity[1.]}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{24.680699999999998`, 16.515900000000002`}, {20.9837, 16.515900000000002`}, {20.9837, 14.358900000000002`}, {24.680699999999998`, 14.358900000000002`}, {24.680699999999998`, 11.688900000000004`}, {29.914699999999996`, 15.438900000000002`}, {24.680699999999998`, 19.184900000000003`}}}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{20.983499999999996`, 6.4854}, {26.217499999999994`, 2.7374}, {26.217499999999994`, 5.4074}, {29.915499999999994`, 5.4074}, {29.915499999999994`, 7.5644}, {26.217499999999994`, 7.5644}, {26.217499999999994`, 10.2344}}}]}, {FaceForm[{RGBColor[0.28600000000000003`, 0.28600000000000003`, 0.28600000000000003`], Opacity[1.]}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {1, 3, 3}}}, CompressedData["
1:eJxTTMoPSmVmYGBgBGJhIGYC4pR3UU52L/Qcfp1+d/Kw0397sR6vVyxftDH4
E+p+WxWcY3fgEBKxPxYjj8GHqZ8UXKIy/b8EnK/k+DH5zF1xOD9ITizL97OY
wzaHpkfHZ/yG8/98K30wR/A7nL8m815hV98nuP5Lex6LyJ58Czcfxoe5H8bn
e6A7YUGFPpxfVijN++CtPtw8GB9mH4wPcw9MP8y96OEDAMtfnGs=
"]], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}}}, {{{20.467999999999996`, 5.387}, {1.0319999999999965`, 5.387}, {1.0319999999999965`, 17.619}, {20.467999999999996`, 17.619}}, {{20.084, 19.215}, {1.734, 19.215}, {1.169, 19.215}, {0.5, 18.881999999999998`}, {0.5, 18.317000000000004`}, {0.5, 4.51}, {0.5, 3.945}, {1.169, 3.259}, {1.734, 3.259}, {20.084, 3.259}, {20.648, 3.259}, {20.999, 3.945}, {20.999, 4.51}, {20.999, 18.317000000000004`}, {20.999, 18.879}, {20.646, 19.215}, {20.084, 19.215}}}]}}, AspectRatio -> Automatic, ImageSize -> {30., 21.}, PlotRange -> {{0., 30.}, {0., 20.1667}}];


(* ::Subsection:: *)
(*Login Dialog Settings*)


$RememberMeFontSize = 12;
$RememberMeTextColor = $DefaultColor;

$SubheaderColor = RGBColor[13/15, 1/15, 0];
$SubheaderFontSize = 18;
$BaseFontFamily = "Source Sans Pro";
$InputFieldHeaderFontSize = 14;
$DefaultColor = RGBColor[100/255, 100/255, 100/255];
$InputFieldFontSize = 14;
$InputTextColor = RGBColor[.2, .2, .2];

$ButtonFontSize = 12;
$CancelButtonTextColor = $DefaultColor;
$SignInButtonTextColor = RGBColor[1, 1, 1];
$SignInButtonDisabledTextColor = RGBColor[.8, .8, .8];

$DialogBackgroundColor = RGBColor[1, 1, 1];
spacr[vertspc_] := Spacer[{30, vertspc}];
$DialogMargin = 41;
$DialogWidth = 490;

signdefault = FrontEnd`FileName[{"Dialogs", "CloudLogin", "v2.0"}, "SigninButton-Default.9.png"];
signhover = FrontEnd`FileName[{"Dialogs", "CloudLogin", "v2.0"}, "SigninButton-Hover.9.png"];
signpressed = FrontEnd`FileName[{"Dialogs", "CloudLogin", "v2.0"}, "SigninButton-Pressed.9.png"];
canceldefault = FrontEnd`FileName[{"Dialogs", "CloudLogin", "v2.0"}, "CancelButton-Default.9.png"];
cancelhover = FrontEnd`FileName[{"Dialogs", "CloudLogin", "v2.0"}, "CancelButton-Hover.9.png"];
cancelpressed = FrontEnd`FileName[{"Dialogs", "CloudLogin", "v2.0"}, "CancelButton-Pressed.9.png"];

checkBoxOn = Dynamic[RawBoxes[FEPrivate`ImportImage[FrontEnd`ToFileName[{"Dialogs", "CloudLogin", "v2.0"}, "CheckboxOn.png"]]], ImageSizeCache -> {16., {5., 11.}}];
checkBoxOff = Dynamic[RawBoxes[FEPrivate`ImportImage[FrontEnd`ToFileName[{"Dialogs", "CloudLogin", "v2.0"}, "CheckboxOff.png"]]], ImageSizeCache -> {16., {5., 11.}}];

chkbox[dyn_Dynamic] := Toggler[dyn,
		{
			True -> checkBoxOn,
			False -> checkBoxOff
		}
]

button[label_, "hyperlink", events_] := MouseAppearance[button[label, events], "LinkHand"];

button[label_, events_] := EventHandler[
	label,
	events,
	PassEventsDown -> True
]

styledtxt[txt_, "rememberme"] := styleBaseFont[txt, FontSize -> $RememberMeFontSize, FontColor -> $RememberMeTextColor, LineBreakWithin -> False];

styleButton[txt_, colr_] := 
 styleBaseFont[txt, 
  FontSize -> $ButtonFontSize, LineBreakWithin -> False, 
  FontColor -> colr]

Attributes[enabledColorToggle] = {HoldAll};
enabledColorToggle[label_, enabledColr_, disabledColr_, 
  btnenabledfunction_] := 
 PaneSelector[{True -> styleButton[label, enabledColr], 
   False -> styleButton[label, disabledColr]}, btnenabledfunction]

Attributes[loginbtn] = {HoldRest};

loginbtn[label_, appearance_List, function_, 
  opts : OptionsPattern[]] := 
 Button[label, function, Appearance -> appearance, 
  FrameMargins -> {{10, 10}, {0, 0}}, 
  ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]], 
  Alignment -> {Center, Center}, opts]

loginbtn["SignInButton", btnfunction_, btnenabledfunction_] := 
 loginbtn[enabledColorToggle[
   "Sign In", $SignInButtonTextColor, $SignInButtonDisabledTextColor, 
   btnenabledfunction], {"Default" -> signdefault, 
   "Hover" -> signhover, "Pressed" -> signpressed, 
   "ButtonType" -> "Default"}, btnfunction, 
  Enabled -> btnenabledfunction]

loginbtn["CancelButton", btnfunction_] := 
 loginbtn[styleButton[
   "Cancel", $CancelButtonTextColor], {"Default" -> canceldefault, 
   "Hover" -> cancelhover, "Pressed" -> cancelpressed, 
   "ButtonType" -> "Cancel"}, btnfunction]

signinbtns[Dynamic[logincreds_], btnenabledfunction_] := Grid[{{
 	loginbtn["SignInButton", DialogReturn[logincreds], btnenabledfunction], 
    loginbtn["CancelButton", DialogReturn[MathLink`CallFrontEnd@FrontEnd`WolframCloud`ConnectionCancelled[]; logincreds = $Canceled]]
}}]

inputField[header_, dyn : Dynamic[symbol_], 
  opts : OptionsPattern[]] :=
 Grid[{{header},
    {InputField[dyn, String, ContinuousAction -> True, 
     ImageSize -> {Full, Automatic}, 
     BaseStyle -> {FontFamily -> 
        $BaseFontFamily, 
       FontWeight -> "Regular", 
       FontColor -> $InputTextColor, 
       FontSize -> $InputFieldFontSize}, 
     opts]}}, Alignment -> Left]

styleBaseFont[txt_, opts___] := 
 Style[txt, FontFamily -> $BaseFontFamily, 
  FontWeight -> "Regular", LineSpacing -> {1, 0}, 
  AutoSpacing -> False, opts]

styleText[txt_, opts___] := 
 styleBaseFont[txt, 
  FontSize -> $InputFieldHeaderFontSize,
  FontColor -> $DefaultColor, 
  LineBreakWithin -> False, opts]

styleSubHeader[txt_] := 
 styleBaseFont[txt, 
  FontSize -> $SubheaderFontSize, 
  FontColor -> $SubheaderColor]


(* ::Subsection:: *)
(*Login Dialog*)


loginDialog[username_String, hostname_String] := 
 With[{boxid = "username", pwdboxid = " passwd"},
   Block[{$loginCredentials},
    Module[{ uname = username, 
      pwd = "", rememberme = False, ctrlCol},
     If[Developer`UseFrontEnd[CurrentValue["UserInteractionEnabled"]],
       ctrlCol = 
       Column[{styleSubHeader[
          "Log in to " <> hostname], spacr[26],
         inputField[
          styleText["Username"], 
          Dynamic[uname], BoxID -> boxid], 
         spacr[28], 
         inputField[
          styleText["Password"], 
          Dynamic[pwd], BoxID -> pwdboxid, 
          FieldMasked -> True], spacr[28], 
         DynamicWrapper[
          Grid[{{
				Grid[{{
						chkbox[Dynamic[rememberme]],
						button[styledtxt["Remember Me", "rememberme"],
							"MouseClicked" :> (
								rememberme = !rememberme
							)
						]
					}},
					Alignment -> {Automatic, Center}
				]
				},
			  {Grid[{{signinbtns[
                 Dynamic[$loginCredentials],
                 Dynamic[uname =!= ""]
              ]}}, 
              ItemSize -> {{Scaled[0.58`], Scaled[0.42`]}, Automatic},
               Spacings -> {0, 0}, 
              Alignment -> {{Center}, Automatic}]}}, 
           Alignment -> {Left, Automatic}, 
           Spacings -> {Automatic, 1}], 
          $loginCredentials = {uname, pwd, rememberme}
          ]}, 
        Alignment -> Left, Spacings -> {0, 0}, 
        ItemSize -> {Automatic, Automatic}]; 
      DialogInput[
       ExpressionCell[
        Framed[Column[{spacr[34], 
           Pane[ctrlCol, {Full, All}, 
            FrameMargins -> {{$DialogMargin, 
               $DialogMargin}, {0, 0}}], 
           spacr[26]}, Spacings -> {0, 0}], 
         ImageSize -> {Full, Full}, FrameMargins -> 0, 
         ImageMargins -> {{0, 0}, {-3, -1}}, FrameStyle -> None], 
        CellMargins -> {{-1, -5}, {0, -2}}, CellFrameMargins -> 0], 
       Background -> $DialogBackgroundColor, 
       CellContext -> Notebook, DynamicUpdating -> True, 
       DynamicEvaluationTimeout -> 100.`, 
       WindowTitle :> "Remote Login",
       WindowSize -> {$DialogWidth, All}, 
       Modal -> True, 
       NotebookDynamicExpression :> 
        Refresh[FrontEnd`MoveCursorToInputField[EvaluationNotebook[], 
          If[uname === "", boxid, pwdboxid]], 
         None]]; $loginCredentials,
      Return[$Canceled]]]]] /; BoxForm`sufficientVersionQ[11]


Protect[{RemoteConnect, RemoteConnectionObject, RemoteRun, RemoteRunProcess, RemoteFile}];
End[];
EndPackage[];
