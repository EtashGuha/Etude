Package["NeuralNetworks`"]

$stoppingCriterionSetup := Quoted[
	$itersSinceImprovement = 0;
	{$stoppingCriterion, $betterCriterion} = parseStoppingCriterion[trainingStoppingCriterion];
	If[$stoppingCriterion =!= None,
		$earlyStoppingPeriodicCallback = makePeriodicFunction[
			TrainingStoppingCriterion, checkEarlyStopping,
			If[$doValidation, validationInterval, {"Interval" -> 1}]
		];
	]
];

checkEarlyStopping[] := (
	$CoreLoopLogger["CheckingEarlyStopping"];
	If[$stoppingCriterion[$stoppingCriterionFunctionVars],
		$shouldStop = True;
		$reasonTrainingStopped = "StoppingCriterion";
		handleEvent["TrainingStoppedEarly"];
		$CoreLoopLogger["TrainingStoppedEarly"];
	];
);

(* parsing of TrainingStoppingCriterion option *)

NetTrain::invstopusage = "`` is not a valid set of arguments for TrainingStoppingCriterion.";
NetTrain::invstopnumopts = "Please specify only one of \"AbsoluteChange\", and \"RelativeChange\" as options for TrainingStoppingCriterion.";
NetTrain::invstopmetric = "`` is not an available criterion for TrainingStoppingCriterion, available criteria include ``.";
NetTrain::novalidation = "No validation set provided, defaulting to training set for stopping criterion.";
NetTrain::invtscres = "The function supplied to TrainingStoppingCriterion must return True or False. Check that you are using a valid criterion.";
NetTrain::invrelchange = "The value for \"RelativeChange\" must be a fraction between 1 and 0, but was ``.";
NetTrain::nonscalarcrit = "The specified TrainingStoppingCriterion is not a scalar. Please aggregate or class-average the criterion to ensure it is a scalar."

criterionSpecT = CustomType["plural[string] or plural[function]", If[MatchQ[#, _String | _Function], #, $Failed]&];
percentageT = CustomType["plural[percentage quantity]", If[MatchQ[#, HoldPattern @ Quantity[_ , "Percent"]], #, $Failed]&];
roundT = CustomType["plural[round quantity]", If[MatchQ[#, HoldPattern @ Quantity[_, "round" | "Round" | "rounds" | "Rounds"]], First @ #, $Failed]&];
nonNegativeRealT = CustomType["plural[non-negative real number]", If[NumberQ[#] && NonNegative[#], #, $Failed]&];

specs = {
	"Criterion" -> criterionSpecT,
	"Patience" -> Defaulting[EitherT[{NaturalT, roundT}], 0],
	"InitialPatience" -> Defaulting[EitherT[{NaturalT, roundT}], 0]
};
extraSpecs = {
	"AbsoluteChange" -> Defaulting[Nullable[nonNegativeRealT], None],
	"RelativeChange" -> Defaulting[Nullable[EitherT[{nonNegativeRealT, percentageT}]], None]
};
StoppingCriterionStringDefinionT = StructT[specs ~Join~ extraSpecs];
StoppingCriterionFuncDefinionT = StructT[specs];

$stoppingCriterionFunctionVars = Association["ValidationLoss" :> $validationLoss, "ValidationMeasurements" :> $validationMeasurements, "RoundLoss" :> $roundLoss, "RoundMeasurements" :> $roundMeasurements];
		
parseStoppingCriterion[None] := 
	{None, makeDefaultImprovedQ[]};

parseStoppingCriterion[crit_Function] := 
	parseStoppingCriterion[<|"Criterion" -> crit|>];

parseStoppingCriterion[crit_String] := 
	parseStoppingCriterion[<|"Criterion" -> crit|>];

parseStoppingCriterion[Automatic] := 
	{None, makeDefaultImprovedQ[]}

parseStoppingCriterion[spec:<|"Criterion" -> _String, opts___Rule|>] := ModuleScope[
	If[!$doValidation, Message[NetTrain::novalidation]];

	spec = CoerceUserSpec[#, StoppingCriterionStringDefinionT, "the built-in stopping criterion specification"] & @ spec;	

	{absChange, relChange, patience, initialPatience} = Lookup[spec, {"AbsoluteChange", "RelativeChange", "Patience", "InitialPatience"}];
	count = Count[{absChange, relChange}, Except[None]];
	If[count > 1, ThrowFailure["invstopnumopts"]];	
	If[count == 0, absChange = 0]; (*The default is an absolute decrement of 0*)

	If[$doValidation,
		$criterionAssociation := $validationMeasurements,
		$criterionAssociation := $roundMeasurements
	];

	criterion = spec["Criterion"];

	criterionValue := Lookup[$criterionAssociation, criterion,
		ThrowFailure["invstopmetric", criterion,  QuotedStringRow[Keys @ $criterionAssociation, " and "]]
	];
	
	direction = Lookup[$metricDirections, criterion, "Decreasing"];
	improvedQfunc = Switch[direction,
		"Increasing",
			$bestValue = -Infinity; Then[# > $bestValue, $bestValue = Max[#, $bestValue]]&,
		"Decreasing",
			$bestValue = Infinity; Then[# < $bestValue, $bestValue = Min[#, $bestValue]]&
	];
	(*^ see Validation.m for where this is used *)

	If[relChange =!= None && (relChange >= 1 || relChange <= 0), ThrowFailure["invrelchange", relChange]];

	checkerFunction = Which[
		absChange =!= None && direction == "Increasing",
			$best = -Infinity; Then[# <= $best + absChange, $best = Max[#, $best]]&,
		absChange =!= None && direction == "Decreasing",
			$best = Infinity; Then[# >= $best - absChange, $best = Min[#, $best]]&,
		relChange =!= None && direction == "Increasing",
			$best = -Infinity; Then[# <= $best * (1 + relChange), $best = Max[#, $best]]&,
		relChange =!= None && direction == "Decreasing",
			$best = Infinity; Then[# >= $best * (1 - relChange), $best = Min[#, $best]]&
	];

	checkScalar := Function[
		Switch[Dimensions[#],
			{},
		 		checkScalar = Identity; #,
		 	{1},
				checkScalar = First; First @ #,
			_,
				checkScalar = Identity; Message[NetTrain::nonscalarcrit]; #
		]
	];
	(* ^ checkScalar is a memoizing function that will only be called once per call to NetTrain.
	The reason for doing this is that we don't wan't (or need) to check for scalars multiple times.
	We only check the first time we check the stopping criterion since the result will never change.
	We'd rather not print multiple warning messages that say the same thing or use extra computation.
	To acomplish this, the first time checkSclar is called, it changes its own definition.
	*)

	func = checkerFunction @ checkScalar @ criterionValue &;

	stopQ = addPatience[func, initialPatience, patience];

	{stopQ, Function[improvedQfunc @ criterionValue]}
];

checkBoolean[arg_] := If[BooleanQ[arg], arg, Message[NetTrain::invtscres]; False];

parseStoppingCriterion[spec:<|"Criterion" -> _Function, opts___Rule|>] := ModuleScope[

	spec = CoerceUserSpec[#, StoppingCriterionFuncDefinionT, "the function stopping criterion specification"] & @ spec;	

	stopQ = addPatience[spec["Criterion"] /* checkBoolean, spec["InitialPatience"], spec["Patience"]];

	betterQ = makeDefaultImprovedQ[];

	{stopQ, betterQ}
];

addPatience[criterion_, 0, 0] := criterion;

addPatience[criterion_, initialPatience_, patience_] := ModuleScope[
	$roundsSinceImprovement = 0;
	Function @ Which[
		$round < initialPatience, False, (* if we're in grace period, don't stop, increase rounds *)
		!TrueQ[criterion @ #], $roundsSinceImprovement = 0; False, (* if criterion says don't stop, don't stop, reset patience *)
		$roundsSinceImprovement == patience, True, (* if patience has run out, stop *)
		True, $roundsSinceImprovement++; False (* lose patience, but don't stop *)
	]
];

	
parseStoppingCriterion[x_] := ThrowFailure["invstopusage", x];

makeDefaultImprovedQ[] /; !$doValidation := None;

(* TODO: Fix me, lowestValidationError doesn't exist any more *)
makeDefaultImprovedQ[] /; $doValidation := ModuleScope[
	$lowestValidationLoss = $lowestValidationError = Infinity;
	
	Function[
		If[KeyExistsQ[$validationMeasurements, "ErrorRate"],
			validationError = $validationMeasurements["ErrorRate"];
			If[validationError < $lowestValidationError || (validationError == $lowestValidationError && $validationLoss < $lowestValidationLoss),
				$lowestValidationLoss = $validationLoss;
				$lowestValidationError = validationError;
				True
			,
				False
			]
		,
			If[$validationLoss < $lowestValidationLoss,
				$lowestValidationLoss = $validationLoss;
				True
			,
				False
			]
		]
	]
];
