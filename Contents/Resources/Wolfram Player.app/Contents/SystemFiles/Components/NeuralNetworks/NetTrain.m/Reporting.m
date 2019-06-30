Package["NeuralNetworks`"]


(* Note: this is substituted into the definition of NetTrain using a MacroEvaluate *)
$progressReportingSetup := Quoted[

	$startTime = $now;

	$doPlotting = $doPrinting = False; 
	$progCell = $progressType = None; $progressVar = 0;
	
	If[trainingProgressReporting =!= None,
		$reportingPeriodicCallback = processTrainingProgressReportingSpec[],
		$reportingPeriodicCallback = Hold;
	];
	
	$startTime = $now;
	
	ScopeVariable[
		$currentMetric, $metricFlipper, $lastMetricPlot, 
		$reportingObserver, $lastReportingObservation
	];

	$endTime = $startTime + Replace[$targetTime, None -> Infinity];	
];


RunInitializationCode[
	$defaultProgType = If[!$Notebooks, "Print", If[!$CloudOrPlayer, "Panel", None]];
	$allowedProgType := $allowedProgType = EnumT @ Flatten @ {If[$Notebooks, {"Panel", "ProgressIndicator"}, Nothing], "Print", Automatic, None};
];


processTrainingProgressReportingSpec[] := Block[
	{allowedProgInf, progRepSpec, progRepOpts, callbackFunc, secint, pffunc, method, numMetrics},

	progRepSpec = ToList[trainingProgressReporting];

	method = First @ progRepSpec;
	progRepSpec = Rest @ progRepSpec;
	progRepOpts = Sequence @@ Append[progRepSpec, $defTimerUnit -> 3 (* seconds *)];

	Switch[method,
	"Panel" /; $Notebooks,
		callbackFunc = makePeriodicFunction[TrainingProgressReporting, updateDynamicProgress, progRepOpts, "Interval" -> {0.33, "Seconds"}];
		numMetrics = Length[$measurementsInfo];
		$lastMetricPlotBoxes = "";
		(* there is always a single, total-loss metric *)
		$metricFlipper = Which[
		numMetrics == 1, 
			$currentMetric = 0; 
			None,
		numMetrics == 2,
			$currentMetric = 1;
			ToBoxes @ Grid[{{
				Button["", Appearance -> None],
				$measurementsInfo[[2, "LongName"]],
				Button["", Appearance -> None]
			}}, ItemSize -> {{3, 20, 3}, {2}}],
		True,
			$currentMetric = 1;
			First @ makeLeftRightClickerBoxes[
				$currentMetric,
				Map[Apply[measurementTooltip]] @ Map[{#, #LongName}&] @ $measurementsInfo[[2 ;;]]
			]
		];
		updatePlotBoxes[];
		secint = callbackFunc[[1, -1, 1, -1, -1, 3]];
		$doPlotting = True;
		$progCell = PrintTemporary @ makeTrainingBox[secint]
	,
	"ProgressIndicator" /; $Notebooks,
		callbackFunc = makePeriodicFunction[TrainingProgressReporting, Function[$progressVar = $progressFraction], progRepOpts, "Interval" -> {0.1, "Seconds"}];
		$progCell = PrintTemporary @ makeSimpleProgressBox[]
	,
	"Print",
		$doPrinting = True; 
		$printCount = 0;
		(* Sample output:
  %  round  batch   examples     inputs   learning       time       time      batch      round       test    current      round       test
        /3   /750  processed    /second       rate    elapsed       left       loss       loss       loss      error      error      error
 33      1    750      48000      11083   7.27*^-4         5s        11s   3.79*^-2   1.86*^-1   5.77*^-2      5.50%      5.50%      1.85%
		*)
		$colSizes = {3, colSize[maxTrainingRounds], colSize[$batchesPerRound], Max[colSize[batchSize * $batchesPerRound * maxTrainingRounds], 10], 10, 10, 10, 10};
		$colSizes = Join[$colSizes, Flatten @ Map[measurementColSizes] @ $measurementsInfo];
		If[$reportMemoryUsage, AppendTo[$colSizes, StringLength[MemoryUsageInfoString[]] + 2]];
		If[$reportTimings, AppendTo[$colSizes, (StringLength["666ms "] * Length[$timingAssociation]) + StringLength["mean: "] + 2]];
		callbackFunc = makePeriodicFunction[TrainingProgressReporting, progressPrintFunction, progRepOpts,  
			"Interval" -> {2, "Seconds"}];
		Print["Starting training."];
		Print["Optimization Method: ", ToString[$LastOptimizerMethod]];
		Print["Device: ", ContextToString @ $DefaultContext];
		Print["Batch Size: ", ToString[batchSize]];
		Print["Batches Per Round: ", ToString[$batchesPerRound]];
		Print["Training Examples: ", ToString[$trainingLength]];
	,
	File[_String], 
		file = First[method];
		fileExt = FileExtension[file];
		setupReportingInfo[];
		repFunc = toReportingFileFunction[ToUpperCase @ fileExt, file] @* Function[$lastReportingObservation = $reportingObserver[];];
		callbackFunc = makePeriodicFunction[TrainingProgressReporting, repFunc, progRepOpts, "Interval" -> {1, "Round"}]
	,
	_Function /; $Notebooks,
		$progCell = {};
		$progressVar := $progressFraction;
		AppendTo[$progCell, 
			PrintTemporary @ makeSimpleProgressBox[1.0]
		];
		pffunc = customProgressFunction @ wrapAbortChecker @ method;
		(* ensure we don't prevent PerformanceGoal being set manually *)
		If[FreeQ[If[SymbolQ[method], DownValues[method], method], PerformanceGoal], 
			pffunc = withPerfGoal[pffunc]
		];
		$customProg = Spacer[{5,5}]; $customProgError = None;
		AppendTo[$progCell,
			PrintTemporary @ Dynamic[$customProg, TrackedSymbols :> {$customProg}, ShrinkingDelay -> 5.0]
		];
		callbackFunc = makePeriodicFunction[TrainingProgressReporting, pffunc, progRepOpts,  
			"Interval" -> {1, "Rounds"}, "MinimumInterval" -> 0.05]
	,
	None, 
		callbackFunc = Hold
	,
	_,
		ThrowFailure["netinvopt", TrainingProgressReporting, TypeString[$allowedProgType]];
	];

	callbackFunc
];


updateDynamicProgress[] := (
	$lastReportingObservation = $reportingObserver[];
	updatePlotBoxes[];
);


PackageScope["$EnableRetinaSnapshots"]
$EnableRetinaSnapshots = False;

takePanelSnapshot[] := Which[
	!MatchQ[$progCell, HoldPattern @ _CellObject], 
		None,
	TrueQ @ $EnableRetinaSnapshots,
		Block[{img = Rasterize[$progCell, ImageResolution -> 144]}, Image[img, ImageSize -> ImageDimensions[img]/2]],
	True,
		Rasterize[$progCell]
];

withPerfGoal[f_][] := Block[{$PerformanceGoal = $ProgressReportingFunctionPerformanceGoal}, f[]];

NetTrain::badrepfile = "Could not open file `` specified as value for TrainingProgressReporting.";
checkOpenWrite[file_, final_:None] := ModuleScope[
	stream = Quiet @ Check[OpenWrite[file], $Failed];
	If[Head[stream] =!= OutputStream, ThrowFailure["badrepfile", file]];
	AppendTo[$CleanupQueue, If[final =!= None,
		Hold[WriteString[stream, final]; Close[stream]],
		Hold[Close[stream]]
	]];
	stream
];

toReportingFileFunction[fmt:"CSV" | "TSV", file_] := ModuleScope[
	stream = checkOpenWrite[file];
	sep = If[fmt === "CSV", ", ", "\t"];
	setupReportingFileFields[];
	WriteString[stream, StringRiffle[Keys @ $reportingInfo, sep], "\n"];
	Function @ WriteString[stream, 
		StringRiffle[Values @ $reportingInfo, sep],
		"\n"
	]
];

toReportingFileFunction["JSON", file_] := ModuleScope[
	stream = checkOpenWrite[file, "\n]"];
	WriteString[stream, "["];
	isFirst = True;
	Function @ WriteString[stream, 
		If[isFirst, isFirst = False; "\n\t", ",\n\t"],
		Developer`WriteRawJSONString[
			evalAssoc @ $reportingInfo,
			ConversionRules -> {None -> Null},
			"Compact" -> True
		]
	]
];

toReportingFileFunction["M" | "WL", file_] := ModuleScope[
	stream = checkOpenWrite[file, "\n}"];
	WriteString[stream, "{"];
	isFirst = True;
	Function @ WriteString[stream,
		If[isFirst, isFirst = False; "\n\t", ",\n\t"],
		ToString[evalAssoc @ $reportingInfo, InputForm, PageWidth -> Infinity]
	]
];

NetTrain::repfileext = "File `` specified as value for TrainingProgressReporting does not have one of the supported extensions ``.";
toReportingFileFunction[ext_, file_] := 
	ThrowFailure["repfileext", file, {"WL", "JSON", "CSV", "TSV"}];

setupReportingInfo[] := Scope[
	reportingKeys = Flatten @ {
		"ProgressFraction","Round", "Batch", "ExamplesProcessed", "ExamplesPerSecond", "LearningRate", "TimeElapsed", "TimeRemaining",
		If[$reportMemoryUsage, $lastMemoryUsageInfoString, Nothing]
	};
	measurements = Association @ Flatten @ Map[measurementRules] @ $measurementsInfo;
	$reportingInfo ^= Join[
		MapAt[Round, "TimeElapsed"] @ KeyTake[$livePropertyData, reportingKeys],
		measurements
	];
];

measurementRules[assoc_Association] /; assoc["TextFormat"] === None := Nothing
measurementRules[assoc_Association] := With[
	{
		perBatch = assoc["$PerBatch"],
		preFormat = applyPre @ assoc["PreFormat"],
		key = assoc["Key"]
	},
	{
		If[perBatch, StringJoin["Current", key] :> 
			preFormat @ Lookup[$lastReportingObservation, key, None], Nothing],

		StringJoin["Round", key] :> 
			preFormat @ Lookup[$roundMeasurements, key, None],

		If[$doValidation, StringJoin["Validation", key] :> 
			preFormat @ Lookup[$validationMeasurements, key, None], Nothing]
	}
]

evalAssoc[assoc_] := Association @ (Rule @@@ Normal[assoc]);

PackageExport["$ProgressReportingFunctionPerformanceGoal"]
$ProgressReportingFunctionPerformanceGoal = "Speed";
(* TODO: turn these into InternalOptions *)

PackageExport["$ProgressReportingFunctionChecking"]
$ProgressReportingFunctionChecking = True;

$pfchecker := If[TrueQ[$ProgressReportingFunctionChecking], EvaluateChecked, Identity];

(* TODO: how in theory could we give access to the instantaneous values of the metrics *)
customProgressFunction[func_][] := (
	Set[$customProg, Deploy @ $pfchecker @ formatResult @ func @ $livePropertyData];
)


formatResult[e_] := fmtRes[e];

fmtRes[l_List ? MachineArrayQ] := 
	Style[MatrixForm[
		l /. r_Real | r_Rational :> SciString[r, 6], 
		TableAlignments -> Right,
		TableDirections -> If[ArrayDepth[l] === 1, Row, Automatic]
	], FontSize -> 10];

$fmtd = 0;
fmtRes[l_List] := Block[{$fmtd = $fmtd + 1}, Which[
	VectorQ[l], If[$fmtd == 1, Multicolumn[fmtRes /@ l], Row[fmtRes /@ l, " "]],
	MatrixQ[l], Grid[Map[fmtRes, l, 2]],
	True, Column[fmtRes /@ l, Alignment -> Left]
]];

fmtRes[a_Association] := Grid[
	KeyValueMap[{Style[#1, "Label", 10], fmtRes[#2]}&, a], 
	Alignment -> Left, Dividers -> {False, Center}, 
	FrameStyle -> LightGray
];

fmtRes[e_] := e;

$printFormattingRules = {
	i_Integer :> IntegerString[i], 
	r_Real :> WLSciString[r], 
	ScientificForm[v_, _] :> WLSciString[v], 
	None -> " \[LongDash] "
};

printRow[items___] := Print @ RightAlignedRowString[$colSizes, {items} /. $printFormattingRules];

colSize[Infinity] := 6;
colSize[n_] := Max[2+base10Digits[n], 6];


measurementColSizes[assoc_Association] /; assoc["TextFormat"] === None := Nothing
measurementColSizes[assoc_Association] := {If[assoc["$PerBatch"], 10, Nothing], 10, If[$doValidation,10, Nothing]}

measurementHeadings[assoc_Association] /; assoc["TextFormat"] === None := Nothing
measurementHeadings[assoc_Association] := With[
	{
		shortName = assoc["ShortName"]
	},
	Transpose @ {
		If[assoc["$PerBatch"], {"current", shortName}, Nothing],
		{"round", shortName},
		If[$doValidation,{"test", shortName}, Nothing]
	}
]

measurementValues[assoc_Association] /; assoc["TextFormat"] === None := Nothing
measurementValues[assoc_Association] := With[
	{
		textFormat = Replace[assoc["TextFormat"], {errForm -> percentForm, f_ :> f /* lossForm}],
	 	preFormat = applyPre @ assoc["PreFormat"],
	 	perBatch = assoc["$PerBatch"],
	 	key = assoc["Key"]
	},
	{
		If[perBatch, textFormat @ preFormat @ Lookup[$lastReportingObservation, key, None], Nothing],
		textFormat @ preFormat @ Lookup[$roundMeasurements, key, None],
		If[$doValidation, textFormat @ preFormat @ Lookup[$validationMeasurements, key, None], Nothing]
	}
]

progressPrintFunction[] := Scope[
	elapsed = $timeElapsed;
	If[elapsed < 0.2, Return[]];
	$lastReportingObservation = $reportingObserver[];
	If[$measurementsInfo === {}, 
		top = bot = {},
		{top, bot} = MapThread[Join, Map[measurementHeadings] @ $measurementsInfo]];
	If[Mod[$printCount++, 50] == 0,
		printRow["%", "round", "batch", "examples", "inputs", "learning", "time", "time", 
			Sequence @@ top,
			If[$reportMemoryUsage, "memory", Nothing], 
			If[$reportTimings, "timings", Nothing]];

		printRow["", If[$noRoundLimit, "", {"/", maxTrainingRounds}], {"/", $batchesPerRound}, "processed", "/second", "rate", "elapsed", "left", 
			Sequence @@ bot,
			If[$reportMemoryUsage, "usage", Nothing], 
			If[$reportTimings, "Batch  Exec  Sync   Gen   Col   Rep   Funcs", Nothing]];

		$firstPrint ^= False; 
	];
	printRow[
		Round[$progressFraction * 100], 
		$round, $batch, batchSize * $absoluteBatch,
		Round[$batchRate * $updatesPerBatch * batchSize], MXNetLink`$LastGlobalLearningRate,
		TimeString @ Round @ elapsed,
		TimeString @ Round @ $timeRemaining,
		Sequence @@ Flatten @ Map[measurementValues] @ $measurementsInfo,
		If[$reportMemoryUsage, $lastMemoryUsageInfoString, Nothing],
		If[$reportTimings, "curr: " <> timingAssocStr[1] (* mean *), Nothing]
	];
	If[$reportTimings, 
		(* take another row to print out the max time values *)
		printRow[
			"", "", "", "", "", "", "", "", Sequence @@ Table["", Length @ top], If[$reportMemoryUsage, "", Nothing], 
			" max: " <> timingAssocStr[3] (* max *)
		]
	]
];


timingAssocStr[n_] := StringJoin @ Riffle[Map[shortTimeStr[#, n]&, Values @ $timingAssociation], " "];

shortTimeStr[_, _] := "     ";
shortTimeStr[t_List, n_] := StringPadLeft[StringJoin @ StringReplace[timeString[Part[t, n]], " " -> ""], 5];

grid[a___] := Grid[{a}, Alignment -> Left, Spacings -> {1, 0.5}];

row[a___] := Row[DeleteCases[Flatten[{a}], Null]];
$c = ", ";

makeTrainingBox[minint_] := (
	$showBatch = TrueQ[timeForRound > 3] || $batchesPerRound > 1000 || maxTrainingRounds <= 10;
	If[$showBatch === False && !ValueQ[timeForRound], 
		(* if BatchSize was fixed, then we don't know whether the time for a round is large or not,
		so keep measuring it, and if it gets large, start showing the batch *)
		$showBatch := If[TrueQ[$now - $roundStartTime > 3], Clear[$showBatch]; $showBatch = True, False]
	];
	$methodInfo = row[
		$LastOptimizerMethod, " optimizer", $c, 
		"batch size ", batchSize, $c, 
		ContextToString @ $DefaultContext
	];
	TrainingBox[{
		Hold[Item[ProgressIndicator[$progressFraction], Alignment -> Center], SpanFromLeft],
		"progress" :> row[
			Round[$progressFraction * 100], "% ", 
			"(round ", $round, If[!$noRoundLimit, {"/", maxTrainingRounds}],
			If[$showBatch, {$c, "batch ", $batch, "/", $batchesPerRound}],
			")"
		],
		"time" :> row[
			TimeString[Round[$timeElapsed]], " elapsed", $c,
			If[$timeElapsed > 1 && $absoluteBatch > 4, {TimeString[Round[$timeRemaining]], " left", $c}],
			rateForm[$batchRate * $updatesPerBatch * batchSize], " examples/s"
		],
		"method" -> $methodInfo,
		If[$LastOptimizerMethod =!= "ADAM", "learning rate" :> ScientificForm[MXNetLink`$LastGlobalLearningRate, 3]],
		"" :> fixedWidthRow[keystyle /@ {"current", "round", If[$doValidation, "validation", Nothing]}],
		Sequence @@ Map[measurementRow,	$measurementsInfo],
		If[$reportMemoryUsage,
		"memory" :> StringReplace[$lastMemoryUsageInfoString, " GPU" -> "\nGPU"]],
		If[$reportTimings,
		"timings" :> grid[
			{"batch:",		fmtSmallTimings[$batchTimings]},
			{"sync:",		fmtSmallTimings[$syncTimings]},
			{"gen:",		fmtSmallTimings[$generatorTimings]},
			{"exec:",		fmtSmallTimings[$executorTimings]},
			{"coll:",		fmtSmallTimings[$collectorTimings]},
			{"rep:",		fmtSmallTimings[$reportingTimings]},
			{"calls:",		fmtSmallTimings[$callbackTimings]}
		]],
		stackMetricPlots[$lastLossPlotBoxes, $lastMetricPlotBoxes, $metricFlipper],
		Hold[staticNiceButton["Stop", $manualStop = $shouldStop = True; $reasonTrainingStopped = "ManualStop"], SpanFromLeft]
		},
		minint
	]
);

fmtSmallTimings[t_] := Style[timingTuple[t], FontFamily -> "Source Code Pro"];

timingTuple[{last_, min_, max_, _, _}] := StringJoin[
	timeString @ last, " (", timeString @ min, "-", timeString @ max, ")"
]

timingTuple[_] := " \[LongDash] ";

timeString[t_] := Which[
	!NumberQ[t], " ? ",
	t < 0.001, {fmtReal2[t*1000000], "\[Micro]s"},
	t < 1.0, {fmtReal2[t*1000], "ms"},
	True, {fmtReal2[t], "s"}
];

fmtReal2[t_] := If[
	t < 10, ToString @ NumberForm[t, {2, 1}],
	IntegerString @ Round[t]
];

SetHoldAll[stackMetricPlots];
stackMetricPlots[p1_, p2_, flipper_] := With[
	{legend = If[$doValidation, $legend, ""],
	 flipper2 = If[flipper =!= None, flipper, ""]},
	Hold[
		RawBoxes @ GridBox[
			{{legend}, {"loss"}, {p1}, {flipper2}, {p2}}, 
			GridBoxSpacings -> {"Columns" -> {{0}}}
		],
		SpanFromLeft
	] /. {""} -> Nothing
];

$legend := $legend = makeLegend[];

fixedWidthRow[list_] := Row[Pane[#, 70] & /@ list];

measurementTooltip[assoc_Association, label_] := With[
	{		
		classAveraging = Lookup[assoc, "ClassAveraging", None],
		aggregation = Lookup[assoc, "Aggregation", None],
		source = Lookup[assoc, "Source", None]
	},
	Tooltip[label,
		Column[{
				Row[{"Measurement: ", assoc["Measurement"]}],
				If[classAveraging =!= None, Row[{"Class averaging: ", classAveraging}], Nothing],
				If[aggregation =!= None, Row[{"Aggregation: ", aggregation}], Nothing],
				If[source =!= None, Row[{"Source: ", source}], Nothing]
			}]
	]
]

measurementRow[assoc_Association] /; assoc["TextFormat"] === None := Nothing
measurementRow[assoc_Association] := With[
	{
		textFormat = Replace[assoc["TextFormat"], {errForm -> percentForm, f_ :> f /* lossForm}],
	 	preFormat = applyPre @ assoc["PreFormat"],
	 	perBatch = assoc["$PerBatch"],
	 	key = assoc["Key"]
	},
	measurementTooltip[assoc, assoc["ShortName"]] :> fixedWidthRow[{ 
		If[perBatch, textFormat @ preFormat @ Lookup[$lastReportingObservation, key, None], keystyle @ "n.a."],
		textFormat @ preFormat @ Lookup[$roundMeasurements, key, None],
	 	If[$doValidation,
	 		If[$computingValidation, "computing...", textFormat @ preFormat @ Lookup[$validationMeasurements, key, None]],
	 		Nothing
	 	]
	}]
];

applyPre[Identity] := Identity;

applyPre[f_][None] := None;
applyPre[f_][e_] := f[e];

lossForm[None] := " \[LongDash] ";
lossForm[str_String] := str;
lossForm[r_] := ScientificForm[r, 3];

errForm[None] := None;
errForm[e_] := percentForm[e];

rateForm[r_] := If[r > 100, Round[r], NumberForm[r, 2]];

percentForm[None] := " \[LongDash] ";
percentForm[str_String] := str;
percentForm[r_] := Which[
	r == 1.0,     "100%",
	r >= 0.999,    percentString[r*1000000, -5],
	r >= 0.99,    percentString[r*100000, -4],
	r >= 0.9,     percentString[r*10000, -3],
	r >= 0.1,      percentString[r*1000, -2],
	r >= 0.01,     percentString[r*10000, -3],
	r >= 0.001,    "0" <> percentString[r*100000, -4], (* <- this is where we hit 0.1% *)
	r == 0,        "0%",
	True, errtxt @ NumberForm[r*100, 3]]; errtxt[NumberForm[r*100, 3]
];

percentString[r_, pos_] := StringInsert[IntegerString @ Round[r], ".", pos] <> "%"

Clear[makeSimpleProgressBox]
makeSimpleProgressBox[ui_:Infinity] := Dynamic[
	Row[{
		ProgressIndicator[$progressVar],
		"   ", Pane[IntegerString[Round[$progressVar * 100]] <> "%", 40], Button["Stop", $manualStop = $shouldStop = True; $reasonTrainingStopped = "ManualStop"],
		"   ", If[$timeElapsed < 0.1, "", TimeString[Round[$timeRemaining]] <> " remaining"]
	}], 
	BaseStyle -> {FontFamily -> CurrentValue["PanelFontFamily"]}, 
	TrackedSymbols :> {$progressVar},
	UpdateInterval -> ui
];



PackageScope["TrainingBox"]

keystyle[x_] := Style[x, GrayLevel[0.4]];	

stripNull[e_] := DeleteCases[Flatten[e], Null];

TrainingBox[rows_, uinterval_] := Scope[
	grid = With[{rows2 = Hold @@@ stripNull[rows]}, Dynamic[
		Grid[
			rows2,
			Dividers -> {{False, Opacity[0.15]}, {}},
			ItemStyle -> {{GrayLevel[0.4], None}},
			ColumnWidths -> {7, 25},
			ColumnAlignments -> {Right, Left},
			ColumnSpacings -> {1.6, 2.5},
			RowSpacings -> 1.4, RowMinHeight -> 1.3,
			BaseStyle -> {FontSize -> 11}
		],
		TrackedSymbols :> {}, UpdateInterval -> uinterval,
		ContentPadding -> False
	] /. Hold -> List];
	titleItem = Item[
		Framed[
			Style["Training Progress", 12, "SuggestionsBarText"],
			FrameMargins -> {{10,5},{-4,2}},
			FrameStyle -> None
		],
		ItemSize -> {Automatic,1},
		Alignment -> {Left, Bottom},
		FrameStyle -> Opacity[0.1],
		Background -> GrayLevel[0.96],
		Frame -> {{False,False},{True, False}}
	];
	gridItem = Item[
		Framed[grid, FrameMargins -> {{10,10},{10,5}}, FrameStyle -> None],
		BaseStyle -> {FontWeight -> "Light", FontFamily -> CurrentValue["PanelFontFamily"], 
			NumberMarks -> False, 
			ScriptBaselineShifts -> {0, 0.5},
			ScriptMinSize -> 8, ScriptSizeMultipliers -> 0.5
		},
		Alignment -> Left	
	];
	Deploy[
		Style[Framed[
			Column[{
					titleItem,
					gridItem
				},
				ColumnWidths -> Automatic,
				ColumnAlignments -> Center,
				RowLines -> False,
				RowSpacings -> {3,1},
				StripOnInput -> True
			],
			Background -> White,
			FrameMargins -> {{0,0},{0,0}},
			FrameStyle -> LightGray,
			RoundingRadius -> 5
		], LineBreakWithin -> False]
	]
];
