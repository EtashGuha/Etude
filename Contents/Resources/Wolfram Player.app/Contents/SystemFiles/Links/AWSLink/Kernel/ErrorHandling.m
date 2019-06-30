(*******************************************************************************

Error Handling Logic

(first to be loaded by init.m - package macros go there)

*******************************************************************************)

Package["AWSLink`"]

PackageImport["JLink`"]
PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
General::"awslinkunk" = "One or more Java error happend.";
General::"awslinkliberr" = "Java function `1` failed. Error from AWS Java SDK: \"`2`\".";
General::"jlinkliberr" = "JLink call to `1` with args `2` failed.";
General::"awslinkuneval" = "Java function `1` with args `2` did not evaluate.";
General::"awsarginv" := "Invalid resource or arguments types passed to `1`.";
General::"awsunvalid" = "The `1`[...] provided is an unvalid resource, please try `2`Connect[`2`Client[...], `1`[...]] to re-connect to the service";
General::"awsnoprop" = "`1` is not an available property of an `2` object";
General::"awsunsupported" = "`1` is not (yet) a supported option";
General::"awsoptmismatch" = "Options `1` and `2` are uncompatible.";
General::"awswrongopt" = "Options `1` does not accepts value `2`.";
General::"awsnofield" = "Unexpected: field `2` provided for setting object `1`.";
General::"awsnoclient" = "Unable to perform the request: no `1`Client[...] provided.\
 Try to set the \"Client\" option value of the function or the session $AWSDefaultClients[\"`2`\"]\
 to a valid `1`Client[...].";
General::"awsbrokenclient" = "The `1` connection is broken, please use AWSClientConnect[\"`2`\"] to re-connect to AWS";
General::"wrgclient" = "Unable to perform the request: the value provided is not\
 a valid `1`Client[...] or Automatic";
General::"awswrgtemplate" = "Unable to import content from object `1`, missing or illspecified template: `2`";
General::"awsremoved" = "The task has been removed - properties are not accessible";
General::"awsLogger" = "Unable to write logs to `1` (priviledges issue ?), `2`"

PackageExport["$AWSDebugLogs"]
$AWSDebugLogs = None;

InitLogger[] :=
	If[ $AWSDebugLogs =!= None,
		If[ MatchQ[$AWSDebugLogs, (_String|_File)] && FileExistsQ[$AWSDebugLogs],
			If[ !DirectoryQ[$AWSDebugLogs],
				Block[
					{temp = OpenAppend[AbsoluteFileName[$AWSDebugLogs]]},
					If[FailureQ[temp],
						$AWSDebugLogs = Null;
						InitLogger[],
						Close[temp];
						True
					]
				],
				$AWSDebugLogs = createFile[
					FileNameJoin[
						{
							$AWSDebugLogs,
							"AWSLinkLogFile"<>
							StringReplace[
								DateString["ISODateTime"],
								":"->"-"]<>
							".txt"
						}
					]
				];
				InitLogger[]
			],
			$AWSDebugLogs = createFile[$AWSDebugLogs];
			InitLogger[]
		],
		True
	];

createFile[fileName_] := Block[
	{trial},
	trial = Quiet[CreateFile[fileName, CreateIntermediateDirectories -> True]];
	If[FailureQ[trial],
		trial=CreateFile[];
		If[!FailureQ[trial],
			Message[AWSGetInformation::"awsLogger", fileName, "using: "<>trial<>" instead"];,
			trial = None;
			Message[AWSGetInformation::"awsLogger"," temporary file", " no logs !"]
		];
	trial
	]
];

logThatAWSError[errorMsg_String]:=
	If[
		$AWSDebugLogs =!= None && FileExistsQ[$AWSDebugLogs] && Not@DirectoryQ[$AWSDebugLogs],
		PutAppend[StringJoin[DateString[], "\n", errorMsg], AbsoluteFileName[$AWSDebugLogs]]
	];

errorParser[message_String] := (
	logThatAWSError[message];
	First[
		StringCases[
			message,
			str : Shortest[StartOfString ~~ __ ~~ "\n\tat"] :> StringReplace[str, {"\n\tat" -> "", "\n" -> "", "\t" -> ""}],
			1
		],
		message
	]
)

msgParser[in : Hold[Message[MessageName[sym_, name_], args___], reported_]] :=
	Block[
		{msgTemplate},
		msgTemplate = ReleaseHold[HoldPattern[MessageName[sym, name]] /. Messages[sym]];
		StringJoin[
			If[reported, "reported message:\n", "\t\tshut up message:\n"],
			If[ StringQ[msgTemplate],
				TemplateApply[
					msgTemplate,
					(*an now an ugly workaround because TempateApply arguments evaluation is not good*)
					Map[
						ToString[Unevaluated[#], InputForm]&,
						Map[Unevaluated, Unevaluated[{args}]]
					]
				],
				StringJoin["\t","\t",ToString[in]]
			]
		]
	];

PackageScope["AWSMessageLogger"]
SetAttributes[AWSMessageLogger, HoldFirst]
AWSMessageLogger[exp_] :=
	(
	InitLogger[];
	Internal`HandlerBlock[
		{"Message", logThatAWSError[msgParser[#]]&},
		exp
	]
	);

$exceptionHandler =
	Function[
		{symbol, exception, message},
		ThrowFailure["awslinkliberr", symbol, errorParser[message]]
	];

PackageScope["safeLibraryInvoke"]
SetAttributes[safeLibraryInvoke, HoldFirst];
Options[safeLibraryInvoke] = {"Function" -> Missing[], "Arguments" -> Missing[]}
safeLibraryInvoke[exp_, OptionsPattern[]] := 
	AWSMessageLogger @ Block[
		{res, $JavaExceptionHandler = $exceptionHandler},
		
		res = Check[
			exp,
			If[ MissingQ[OptionValue["Function"]],
				ThrowFailure["awslinkunk"],
				ThrowFailure["jlinkliberr", OptionValue["Function"], OptionValue["Arguments"]]
			],
			{
				Java::"argx", Java::"argx0", Java::"argx1", Java::"argxs",
				Java::"argxs0", Java::"argxs1", Java::"excptn", Java::"flds",
				Java::"fldx", Java::"fldxs", Java::"init", Java::"nofld",
				Java::"nofld$", Java::"nohndlr", Java::"nometh", Java::"nometh$",
				Java::"obj", Java::"pexcptn", Java::"setfield", Java::"usage",
				JavaNew::"argx", JavaNew::"argx0", JavaNew::"argx1", JavaNew::"fail",
				JavaNew::"intf", JavaNew::"invcls", JavaNew::"usage",
				LoadJavaClass::"ambig", LoadJavaClass::"ambigctor",
				LoadJavaClass::"fail", LoadJavaClass::"usage", JavaObject::"bad",
				JavaObject::"usage"
			}
		];
		
		If[ FailureQ[res],
			logThatAWSError["safeLibrary returned a Failure without catching error check message above"];
			ThrowFailure["awslinkunk"],
			res
		]
	];

safeLibraryInvoke[func_, args___] := safeLibraryInvoke[func[args], "Function" -> func, "Arguments" -> {args}];

(*Messages[Java], Messages[JavaNew], Messages[LoadJavaClass], Messages[JavaObject]*)

PackageScope["FailInOtherCases"]
FailInOtherCases[sym_Symbol] :=
	SetDelayed[
		sym[___],
		Failure["awsarginv", <|"MessageTemplate" :> General::"awsarginv", "MessageParameters" -> {sym}|>]
	];
