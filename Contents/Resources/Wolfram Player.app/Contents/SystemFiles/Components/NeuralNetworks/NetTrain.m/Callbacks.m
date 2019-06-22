Package["NeuralNetworks`"]


(* parsing of TrainingProgressFunction option *)

parseTrainingProgressFunction[None] := Hold;
parseTrainingProgressFunction[f_] := 
	Replace[parsePF[f], l_List :> ApplyThrough[l]] /* handleStop

handleStop[e_List] := Scan[handleStop, e];
handleStop["StopTraining"] := ($manualStop = $shouldStop = True; $reasonTrainingStopped = "ManualStop";)
handleStop["AbortTraining"] := ($softAbort = $shouldStop = True; $reasonTrainingStopped = "Aborted";)
handleStop[_] := Null;

parsePF[list_List] := Map[parsePF, list];
parsePF[{f_, opts__Rule}] := makePeriodicFunction[TrainingProgressFunction, wrapAbortChecker @ f, opts];
parsePF[f_] := makePeriodicFunction[TrainingProgressFunction, wrapAbortChecker @ checkCallbackKeyUsage @ f];

wrapAbortChecker[f_] := Function[inarg, CheckAbort[f[inarg], $softAbort = $shouldStop = True; $reasonTrainingStopped = "Aborted"; $Aborted]];

NetTrain::invcfunckeys = "The provided function `` referenced the following non-existent key(s): ``. See the reference page for TrainingProgressFunction for the available keys.";
checkCallbackKeyUsage[f_Function] := Scope[
	h = Hold[f];
	h = Replace[h, Function[_] :> Hold, {2, Infinity}];
	slots = DeepCases[h, HoldPattern[Slot][s_String] :> s];
	extra = Complement[slots, $livePropertyKeys];
	If[extra =!= {}, ThrowFailure["invcfunckeys", f, extra]];
	f
];

checkCallbackKeyUsage[e_] := e;

Options[makePeriodicFunction] = {"Interval" -> 1, "MinimumInterval" -> 0.0, $defTimerUnit -> 2, "ForceZero" -> False};

NetTrain::invsubopts = "`` is not a valid suboption to ``."
makePeriodicFunction[optsym_, func_, opts:OptionsPattern[]] := Scope[
	{int, minint, defunit, forceZero} = UnsafeQuietCheck[
		OptionValue[{"Interval", "MinimumInterval", $defTimerUnit, "ForceZero"}],
		TestSetsEqual[
			Keys @ Flatten @ {opts}, {"Interval", "MinimumInterval"}, 
			ThrowFailure["invsubopts", #1, optsym]&
		]
	];
	int = parseInterval[int, defunit];
	minint = parseInterval[minint, 3];
	int[[3]] = Max[int[[3]], minint[[3]]];
	iMakePeriodicFunction[func, int]
];

iMakePeriodicFunction[f_List | f_Association, interval_] := 
	iMakePeriodicFunction[ApplyThrough[f], interval];

iMakePeriodicFunction[f_, interval_List] := Module[
	{next = {interval[[1]], interval[[2]], interval[[3]]} * If[TrueQ[forceZero], 0, 1], callback = f},
	(* ^ for periodic functions with a round interval or time interval specified, we don't
	want to run them after the very first batch, so we start their timer at an appropriate value.
	forcezero turns this off, which must be done for timers inside roundcollectors *)
	Function[
		If[And @@ Thread[#1 >= next],
			next = #1 + interval; callback[##2]
		]
	]
];


NetTrain::inveventspec = "`` is not a valid setting for ``. Setting should be either one of the stings ``, or a list of these.";
$allowedEvents = Sort @ {
	"TrainingStarted", "TrainingComplete", 
	"RoundStarted", "RoundComplete",
	"CheckpointStarted", "CheckpointComplete", 
	"NetValidationImprovement",
	"ValidationStarted", "ValidationComplete", 
	"TrainingStoppedEarly",
	"TrainingAborted", "WeightsDiverged"
};

makePeriodicFunction[optsym_, func_, "Event" -> All] := 
	makePeriodicFunction[optsym, func, "Event" -> $allowedEvents];

makePeriodicFunction[optsym_, func_, "Event" -> list_List] := 
	Scan[makePeriodicFunction[optsym, func, "Event" -> #]&, list];

makePeriodicFunction[optsym_, func_, "Event" -> ev_] := (
	If[!MatchQ[ev, Alternatives @@ $allowedEvents], ThrowFailure["inveventspec", "Event" -> ev, optsym, $allowedEvents]];
	KeyAppendTo[$eventHandlers, ev, func];
	Hold
);

handleEvent[event_] := (
	$lastEvent = event; 
	Through[Lookup[$eventHandlers, event, {}][$livePropertyData]];
)

NetTrain::invtint = "The value of the \"Interval\" suboption should be a positive quantity with a unit of \"Rounds\", \"Batches\", \"Percent\", \"Seconds\", \"Minutes\", or \"Hours\".";

parseInterval[unit: "Batch"|"Batches"|"Round"|"Rounds"|"Second"|"Seconds"|"Minute"|"Minutes"|"Hour"|"Hours"|"Percent", _] :=
	parseInterval[{1, unit}, Null];

parseInterval[
	HoldPattern @ Quantity[n_ ? Positive, unit_String | IndependentUnit[unit_String]] |
	{n_ ? Positive, unit_String}, _] := 
	Match[
		unit,
		"Batches"|"Batch" /; IntegerQ[n] :> {n, 0, 0},
		"Percent" :> {Ceiling[n/100. * $maxBatches], 0, 0},
		"Rounds"|"Round"|"Revolutions" :> If[
			IntegerQ[n], {1, n, 0},
			{Ceiling[n * $batchesPerRound], 0, 0}
		],
		"Seconds"|"Second" :> {1, 0, N[n]},
		"Minutes"|"Minute" :> {1, 0, 60 * N[n]},
		"Hours"|"Hour" :> {1, 0, 60 * 60 * N[n]},
		ThrowFailure["invtint"]
	];

parseInterval[n_ ? NonNegative, defaultInd_] := 
	ReplacePart[{0, 0, 0}, defaultInd -> N[n]]

parseInterval[_, _] := ThrowFailure["invtint"];

$none = Style["\[Dash]", Gray];

