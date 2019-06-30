Package["HTTPHandling`"]

(*necessary for async communication with the http server*)
PackageImport["MQTTLink`"]

PackageExport[WebServer]
PackageExport[StartWebServer]
PackageExport[$WebServers]
PackageExport[$MQTTBroker]

WebServer::procFail = "The `1` server failed to start with error code `2` and stderr output:\n`3`.";
StartWebServer::serverErr = "The server encountered an unexpected error and could not handle the request";


SetAttributes[{fixData, fixExpr}, SequenceHold]

(*this is for tracking the broker process and list of servers *)

$DefaultHandler := HTTPErrorResponse[404]
If[! ValueQ[$WebServers], $WebServers = <||>];
If[! ValueQ[$MQTTBroker], $MQTTBroker = None];

fixData[data_] /; MatchQ[Lookup[data, "MultipartElements"], {__}] := Append[
	<|data|>,
	"MultipartElements" -> MapAt[
		With[
			{filename = Lookup[#, "FileName"]},
			If[
				StringQ[filename], 
				Append[#, "ContentType" :> FromCharacterCode[BinaryReadList[filename], "Unicode"]], 
				#
			]
		] &,
		Apply[Join, Map[Thread, Lookup[data, "MultipartElements"]]],
		{All, 2}
	]
]

fixData[data_]:=data

fixExpr[router:{(_Rule|_RuleDelayed)...}|_?AssociationQ|_Rule|_RuleDelayed] := URLDispatcher[router]
fixExpr[expr_] := expr

(ws:WebServer[_?AssociationQ])["AsyncHandler"]:=Function[{asyncTask,client,msg},
	Block[{stderr,exitcode,pubRes},
		With[
			{
				(*import the data as json, converting it from a ByteArray to a string and then to JSON*)
				json =  Association[ImportString[FromCharacterCode[Normal[msg["Data"]]], "JSON"]],
				httpServerProcess = ws["Meta"]["Server"]
			},
			(
				(*before we publish back out on the broker, check to make sure it is still running*)
				If[ProcessStatus[$MQTTBroker["Process"]]=!="Running",
					(*THEN*)
					(*it's stopped for some reason, issue a message and don't try to publish*)
					(
						stderr = ReadString @ ProcessConnection[$MQTTBroker["Process"], "StandardError"];
						exitcode = ProcessInformation[$MQTTBroker["Process"], "ExitCode"];
						Message[WebServer::procFail, "MQTT broker", exitcode, stderr];
						Return[$Failed];
					)
				];
				(*also check the http server*)
				If[ProcessStatus[httpServerProcess]=!="Running",
					(*THEN*)
					(*it's stopped for some reason, issue a message and don't try to publish*)
					(
						stderr = ReadString @ ProcessConnection[httpServerProcess, "StandardError"];
						exitcode = ProcessInformation[httpServerProcess, "ExitCode"];
						Message[WebServer::procFail, "HTTP", exitcode, stderr];
						Return[$Failed];
					)
				];
				(*just send back the response on the topic specified by the Connection*)
				pubRes = TopicPublish[
					client, 
					json["Connection"],
					JSONTools`ToJSON[
						Replace[
							System`GenerateHTTPResponse @@ {fixExpr[ws["Handler"]], fixData[json]}, {
								res_HTTPResponse :> res,
								e_ :> HTTPResponse[
									"GenerateHTTPResponse must return an HTTPResponse: " <> ToString[e, InputForm], 
									<|"StatusCode" -> 500|>
								]
							}
						][{"StatusCode","Body","Headers","Cookies"}],
						"AllowAllSymbols" -> True, 
						"Compact" -> True,
						"ASCIIOnly" -> True
					]
				];
				
				If[!AssociationQ[pubRes],
					(*THEN*)
					(*sending the message failed for some reason*)
					(
						Message[StartWebServer::serverErr];
					)
				];
			)
		]
	]
];


StartWebServer[expr_:$DefaultHandler] := 
	StartWebServer[expr, <||>]

StartWebServer[expr_, port_Integer] := 
	StartWebServer[expr, <|"Port" -> port|>]

StartWebServer[expr_, rule:{RepeatedNull[_Rule|_RuleDelayed]}|_Rule|_RuleDelayed] := 
	StartWebServer[expr, <|rule|>]

StartWebServer[expr_, meta_?AssociationQ] := Block[
	{res},
	With[
		{ws = WebServer[meta]},
		(
			res = ws["StartServer"];
			If[ res =!= $Failed,
				(*THEN*)
				(*add the handler and return the result*)
				res = ws["RegisterHandler", Unevaluated[expr]],
				(*ELSE*)
				(*failed, so return the $Failed*)
				$Failed
			]
		)
	]
]

WebServer[port_Integer:7000] := 
	WebServer[<|"Port" -> port|>]

WebServer[meta_?AssociationQ]["Port"] := 
	Lookup[meta, "Port", 7000]

(ws:WebServer[_?AssociationQ])["Meta"] :=
	Lookup[$WebServers, ws["Port"], <||>]

(ws:WebServer[_?AssociationQ])["AbsoluteURL", path_:"/"] :=
	"http://localhost:" <> ToString[ws["Port"]] <> path

(ws:WebServer[_?AssociationQ])["Running"] := TrueQ@And[
	ProcessStatus[$MQTTBroker["Process"]]==="Running",
	ProcessStatus[ws["Meta"]["Server"]]==="Running",
	ws["Meta"]["Connection"]["Connected"]
]
		


(ws:WebServer[_?AssociationQ])["Handler", func_:Identity] := 
	Replace[
		ws["Meta"], {
			meta_ /; KeyExistsQ[meta, "Handler"] :> Extract[meta, "Handler", func],
			_ :> func @@ {$DefaultHandler}
		}
	]

(ws:WebServer[_?AssociationQ])["RegisterHandler", expr_] := (
	$WebServers[ws["Port"],"Handler"] := expr;
	ws
)

(ws:WebServer[_?AssociationQ])["StopServer"] := (
	If[ ws["Running"],
		If[ws["Meta"]["Connection"]["Connected"],
			DisconnectClient[ws["Meta"]["Connection"]];
		];
		KillProcess[ws["Meta"]["Server"]];
		$WebServers[ws["Port"]] = <|
			ws["Handler", Function[a, "Handler" :> a, {HoldFirst}]]
		|>;
	];
	ws
	)




(ws:WebServer[_?AssociationQ])["StartHTTPServer"] := Block[
	{process, stderr, exitcode},
	(*start up the server process*)
	process = StartProcess[{
		First @ FileNames[
			"server*",
			FileNameJoin[{
				PacletManager`PacletResource["HTTPHandling", "Server"],
				$SystemID
			}]
		],
		"-port=" <> ToString[ws["Port"]],
		"-debug=" <> "false"
	}];
	(*wait for a line - this is just to synchronize with the process before checking if it dies*)
	ReadLine[process];
	If[ProcessStatus[process] =!= "Running",
		(*THEN*)
		(*starting the process failed*)
		(
			stderr = ReadString @ ProcessConnection[process, "StandardError"];
			exitcode = ProcessInformation[process, "ExitCode"];
			Message[WebServer::procFail, "HTTP", exitcode, stderr];
			$Failed
		),
		(*ELSE*)
		(*the process started properly, mark it as true*)
		(
			process
		)
	]
];

(*starting a connection is essentially making a MQTT client and subscribing on the GenerateHTTPResponse topic*)
(ws:WebServer[_?AssociationQ])["StartConnection"] := Module[{client},
	(
	
		client = CreateClient["localhost",1883,"ClientDisconnectedFunction":>(Message[StartWebServer::serverErr]&)];
		If[MatchQ[client,_MQTTClient],
			(*THEN*)
			(*it worked, we have a valid client so subscribe it on the relevant topic*)
			(
				TopicSubscribe[client,"GenerateHTTPResponse","MessageReceivedFunction":>ws["AsyncHandler"]];
				client
			),
			(*ELSE*)
			(*the client failed to be created, *)
			(
				$Failed
			)
		]
	)
];


WebServer::running = "A server is already running on port ``."

(ws:WebServer[_?AssociationQ])["StartServer", force_:False] := Block[
	{port = ws["Port"], serverProcess, connection},
	If[ws["Running"] && !force,
		Message[WebServer::running, port];
		Return[ws]
	];
	(*clear this entry in $WebServers*)
	$WebServers[port] = <||>;
	(*make sure that a MQTT broker is running on port 1883*)
	Quiet[$MQTTBroker = StartBroker[1883],{StartBroker::running}];
	(*connect to the MQTT broker*)
	connection = ws["StartConnection"];
	(*start the web server binary*)
	serverProcess = ws["StartHTTPServer"];
	(*now check the two processes and the connection client*)
	If[Not[MemberQ[{connection,serverProcess,$MQTTBroker},$Failed]],
		(*THEN*)
		(*it worked - put these into the association*)
		(
			$WebServers[port,"Server"] = serverProcess;
			$WebServers[port,"Connection"] = connection;
			ws
		),
		(*ELSE*)
		(*one of them failed, they will have generated their own messages, just return $Failed*)
		$Failed
	]
]

Unprotect[{KillProcess, SystemOpen}]

WebServer /: SystemOpen[ws:WebServer[_?AssociationQ]] := SystemOpen[ws["AbsoluteURL"]]
WebServer /: KillProcess[ws:WebServer[_?AssociationQ]] := ws["StopServer"]
WebServer /: RunProcess[ws:WebServer[_?AssociationQ]] := ws["StartServer"]

Protect[{KillProcess, SystemOpen}]

WebServer /: MakeBoxes[ws:WebServer[_?AssociationQ], StandardForm] := 
	BoxForm`ArrangeSummaryBox[
		WebServer, 
		ws, 
		Dynamic[Button[
			Graphics[{
				If[TrueQ[ws["Running"]], Darker[Green], Red],
				EdgeForm[Directive[Thick, White]],
				Disk[], 
				Text[
					Style["\[WolframLanguageLogo]", Directive[22, White, Bold]], 
					Scaled[{.53, .48}]
				]}, 
				ImageSize -> Dynamic[{
					Automatic, 
					(3.2 * CurrentValue["FontCapHeight"]) / AbsoluteCurrentValue[Magnification]
				}],
				Background -> None
			],
			If[TrueQ[ws["Running"]], KillProcess[ws], RunProcess[ws]],
			Appearance -> "Frameless"
		]], {
			BoxForm`SummaryItem[{"URL: ", Hyperlink[ws["AbsoluteURL"]]}],
			BoxForm`SummaryItem[{"Running: ", Dynamic[TrueQ[ws["Running"]]]}]
		}, {
			BoxForm`SummaryItem[{"Handler: ", Dynamic[Short[ws["Handler", HoldForm]]]}]
		}, 
		StandardForm
	]


