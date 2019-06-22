
BeginPackage["CompileUtilities`Debug`Logger`"]

CreateLogger;
LoggerClass;
Logger;
LoggerQ;
LoggerRecord;
$Loggers

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]


(*
 Probably should filter to find active loggers.
*)

If[ !ListQ[$Loggers], $Loggers = {}]


levelColor := levelColor = <|
	"CRITICAL" -> Red,
	"ERROR" -> Darker[Red],
	"WARNING" -> Orange,
	"INFO" -> Darker[Blue],
	"DEBUG" -> Blue,
	"TRACE" -> GrayLevel[0.8],
	"TEMPORARY" -> Black,
	"NOTSET" -> Black
|>

levelId = <|
	"NOTSET" -> 1000,
	"CRITICAL" -> 50,
	"ERROR" -> 40,
	"WARNING" -> 30,
	"INFO" -> 20,
	"DEBUG" -> 10,
	"TRACE" -> 5,
	"TEMPORARY" -> 1
|>


RegisterCallback["DeclareCompileUtilitiesClass", Function[{st},
LoggerClass = DeclareClass[
	Logger,
	<|
		"critical" -> (
			If[levelId[Self["level"]] <= levelId["CRITICAL"],
				Self["_record", "CRITICAL", ##]
			]&
		),
		"error" -> (
			If[levelId[Self["level"]] <= levelId["ERROR"],
				Self["_record", "ERROR", ##]
			]&
		),
		"warning" -> (
			If[levelId[Self["level"]] <= levelId["WARNING"],
				Self["_record", "WARNING", ##]
			]&
		),
		"info" -> (
			If[levelId[Self["level"]] <= levelId["INFO"],
				Self["_record", "INFO", ##]
			]&
		),
		"debug" -> (
			If[levelId[Self["level"]] <= levelId["DEBUG"],
				Self["_record", "DEBUG", ##]
			]&
		),
		"trace" -> (
			If[levelId[Self["level"]] <= levelId["TRACE"],
				Self["_record", "TRACE", ##]
			]&
		),
		"temporary" -> (
			Self["_record", "TEMPORARY", ##]&
		),
		"_record" -> (
			With[{lr = LoggerRecord[Self, #1, {##2}]},
				If[ToUpperCase[#1] === "TEMPORARY",
					CompileTemporaryInformation[lr],
					Print[lr]
				]
			]&
		),
		"toBoxes" -> Function[{fmt},
			loggerToBoxes[Self, fmt]
		]
	|>,
	{
		"name",
		"level"
	},
	Predicate -> LoggerQ
]
]]


CreateLogger[name_, level_:"NOTSET"] :=
	Module[ {obj},
		obj = CreateObject[
			Logger,
			<|
				"name" -> name,
				"level" -> ToUpperCase[level]
			|>
		];
		AppendTo[$Loggers, obj];
		obj
	]

(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

loggerIcon[color_] := Graphics[Text[
  Style["LOG", color, Bold, 
   0.9*CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];   

loggerToBoxes[logger_, fmt_] :=
	With[{level = logger["level"]},
		BoxForm`ArrangeSummaryBox[
			Logger,
			logger,
	  		loggerIcon[levelColor[level]],
	  		{
	  			BoxForm`SummaryItem[{"name: ", logger["name"]}],
	  			BoxForm`SummaryItem[{"level: ", level}]
	  		},
			{BoxForm`SummaryItem[{getLevelControlList[logger]}]}, 
	  		fmt,
			"Interpretable" -> False
	  	]
	]
  	
(**************************************************)
(**************************************************)


loggerRecordIcon[level_] := Graphics[Text[
  Style["Log\nRec", levelColor[level],  Bold, 
   1.2*CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification], FontSize->7]], $FormatingGraphicsOptions];   
     
LoggerRecord /: MakeBoxes[rec:(LoggerRecord[logger_, level_, {msg__}]), fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"LoggerRecord",
		rec,
  		loggerRecordIcon[level],
  		{
  			BoxForm`SummaryItem[{"Level: ", level}],
  		    BoxForm`SummaryItem[{msg}]
  		},
  		{BoxForm`SummaryItem[{getLevelControlList[logger]}]}, 
  		fmt,
		"Interpretable" -> False
  	]

getLevelControlList[logger_] :=
	Module[{levels},
		levels = Keys[levelColor];
		ActionMenu[ "Set Level", Map[ With[ {level = #},
			   level :> logger["setLevel", #]]&, levels],
			   Appearance -> PopupMenu]
	]


(**************************************************)

End[]

EndPackage[]
