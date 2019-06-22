Package["NeuralNetworks`"]


PackageExport["NetTrain"]

Clear[NetTrain, NetTrainOptionHead]

Options[NetTrain] = Sort @ {
	BatchSize -> Automatic,
	MaxTrainingRounds -> Automatic,
	Method -> Automatic,
	TargetDevice -> "CPU",
	ValidationSet -> None,
	LearningRateMultipliers -> Automatic,
	TrainingProgressCheckpointing -> None,
	TrainingProgressReporting -> Automatic,
	TrainingProgressFunction -> None,
	PerformanceGoal -> Automatic,
	RandomSeeding -> Inherited,
	TimeGoal -> Automatic,
	LearningRate -> Automatic,
	LossFunction -> Automatic,
	TrainingStoppingCriterion -> None,
	TrainingProgressMeasurements -> Automatic,
	WorkingPrecision -> Automatic
};

Options[NetTrainOptionHead] = Join[
	Options[NetTrain], {
	"MaxTrainingBatches" -> Automatic,
	"InternalOptions" -> {}
}];

$shimq = True;
NetTrain /: SetOptions[NetTrain, args___] /; $shimq := Block[{$shimq},
	SetOptions[NetTrain, args];
	Quiet @ SetOptions[NetTrainOptionHead, args];
	Options[NetTrain]
];

RunInitializationCode[
	Format[NetTrainOptionHead, StandardForm] := NetTrain;
	Format[NetTrainOptionHead, OutputForm] := NetTrain;
];

PackageScope["makeCollector"]
SetHoldAll[makeCollector]

makeCollector[n_, body_] := Hold[n; Evaluate[$ncollectors++]; body];
(* ^ This function allows us to specify the ordering of different collectors. 
collectors with lower n$ values will be run before collectors with higher values.
The $ncollectors++ is there to make the sorting stable - that is ensure that collectors
with the same n are kept in the order that they were added.  *)

DefineMacro[addBatchCollector, addBatchCollector[bagSym_, body_, n_] := addCollector[bagSym, body, False, n]];
DefineMacro[addRoundCollector, addRoundCollector[bagSym_, body_, n_] := addCollector[bagSym, body, True, n]];

SetHoldAllComplete[addCollector];
addCollector[bagSymbol_, body_, roundq_, n_] := With[
	{collectorVar = If[roundq, $roundCollectors, $batchCollectors]}, 
	Quoted[
		bagSymbol = Bag[];
		JoinTo[collectorVar, makeCollector[n, BagPush[bagSymbol, body]]];
		(* see 344738 for why we can't use AppendTo *)
	]
];


NetTrain::invnet = "First argument to NetTrain should be a fully specified net.";
NetTrain::unspecloss = "Provided loss layer is not fully specified.";
NetTrain::netnoparams = "Net does not contain any trainable parameters: returning net unchanged."
NetTrain::netfrozen = "All trainable parameters have been frozen: returning net unchanged."
General::optpi = "The value of `` -> `` should be a positive machine-sized integer."

$now := N[SessionTime[]]; (* %KERNEL otherwise it is a bignum *)

$SymbolRule = Rule[_Symbol | "InternalOptions" | "MaxTrainingBatches", _] | RuleDelayed[_Symbol, _];
$NotOption = Except[$SymbolRule | {$SymbolRule..}];

NetTrain[net_, data_, opts:OptionsPattern[NetTrainOptionHead]] :=
	NetTrain[net, data, "TrainedNet", opts];

$LastAbortedLine = 0;
$LastAbortedTime = 0;
$TrainingCounter = 0;

(* for now used to forbid measurements that might affect training, which
is not a concern for NetMeasurements *)
$isInNetTrain = False;

SetAttributes[NetTrainBlock, HoldAllComplete];
NetTrainBlock[expr_] := WithLocalSettings[
	$CleanupQueue = {}; (* this has to be global, as the Scope needs to happen inside the WithLocalSettings *)
	, 
	CatchFailureAsMessage[NetTrain, CatchEncodeFailure @ expr]
	,
	AllowExceptions[
		ReleaseHold[$CleanupQueue];
		$CleanupQueue = {};
	]
] // Replace[{$Aborted :> Abort[], $SoftAborted :> $Aborted}];

NetTrain[net_, idata_, returnSpec:$NotOption, opts:OptionsPattern[NetTrainOptionHead]] := Timed @ NetTrainBlock @ Scope[

	$CleanupQueue ^= {};

	(****************************************************************)
	(* get options and do basic validation                          *)
	(****************************************************************)

	TimeLabel["Setup"];

	If[returnSpec === All, returnSpec = "ResultsObject"];	
	If[returnSpec === {}, Return[{}]];

	listOfSpecsQ = VectorQ[returnSpec, StringQ[#] || AssociationQ[#]&];

	$isInNetTrain = True;
	$roundCollectors = $batchCollectors = Hold[];
	$ncollectors = 1;
	$clock := {$absoluteBatch, $virtualRound, $timeElapsed};
	$CurrentExecutorBucketCount = 0;
	$lastEvent = None; $eventHandlers = <||>;
	$callbackTimings = $reportingTimings = $collectorTimings = $executorTimings = $generatorTimings = $batchTimings = $syncTimings = None;
	$seenPrecompHashes = {}; 
	$LastGeneratorData = None;

	origReturnSpec = returnSpec;
	(* convert aliases of old props to their new versions - see Properties.m *)
	returnSpec = Replace[returnSpec, $netTrainPropertyAliases, {0, 1}];

	If[!MatchQ[returnSpec, If[listOfSpecsQ, {$returnSpecP..}, $returnSpecP]],
		NetTrain::invntrspec = "Property specification `` is not All, a valid property, or a list of valid properties. Valid properties include all keys available to TrainingProgressFunction.";
		ThrowFailure["invntrspec", Short[origReturnSpec]];
	];

	$restArgs := Apply[Sequence,
		{Replace[returnSpec, "ResultsObject" -> All], opts} /. 
		(ValidationSet -> e:Except[_Scaled]) :> (ValidationSet -> Placeholder["vdata"])];

	UnpackOptions[
		method, randomSeeding, internalOptions, 
		batchSize, maxTrainingRounds, maxTrainingBatches, validationSet,
		lossFunction, learningRate, learningRateMultipliers, 
		trainingProgressCheckpointing, trainingProgressFunction, trainingProgressReporting, trainingStoppingCriterion, trainingProgressMeasurements,
		timeGoal, performanceGoal, targetDevice, workingPrecision
	];	

	$doValidation = validationSet =!= None;

	SetAutomatic[trainingProgressReporting, $defaultProgType];
	(* ^ see Reporting.m *)

	GeneralUtilities`$ComputeWithProgressEnabled = trainingProgressReporting =!= None;
	(* TODO: update me if ComputeWithProgress is ever moved to Developer` *)

	If[!MatchQ[randomSeeding, Automatic | Inherited | _Integer | _String],
		NetTrain::seeding = "Value of option RandomSeeding -> `1` is not Automatic, Inherited, an integer, or a string.";
		ThrowFailure["seeding", randomSeeding] 
	];

	(* see PerformanceGoal.m *)
	MacroEvaluate[$applyPerformanceGoal];


BlockRandom[	

	mxSeed = RandomInteger[2^31];
	MXSeedRandom[mxSeed];

	(* it is much more expensive to do things like get array dimensions if the array isn't
	packed, so we take a hit now in exchange for other things being cheaper later *)
	data = TryPack[idata];

	If[StringQ[net],
		net = resolveNamedNet[net];
		If[!ValidNetQ[net], ThrowFailure[]]
	];

	If[!ValidNetQ[net], ThrowFailure["invnet"]];
	
	userMeasurementPorts = GetUserMeasurementPorts[OutputNames[net], trainingProgressMeasurements];

	(* return early for some simple properties *)
	Switch[returnSpec,
		"TrainingNet", Return @ NetAttachLoss[net, lossFunction, userMeasurementPorts],
		"Properties", Return @ $userVisibleProperties,
		_, Null
	];	

	If[!ConcreteNetQ[net], 
		net2 = JITInferTrainNet[net, lossFunction, data, userMeasurementPorts];
		(* ^ normally, we would first construct the training graph (via ToNetTrainer), get the inputs
		of that training graph, and then use the list of inputs to validate and parse the training data 
		via ParseTrainingData. but if the net itself is missing e.g. input dimensions, we must look
		at the data and original net in order to fill those in, which JITInferTrainNet does. *)
		If[FailureQ[net2] || !ConcreteNetQ[net2], ThrowNotSpecifiedFailure[net, "train", "Defined"]];
		net = net2;
	];

	CheckForTuplePorts[Inputs[net], NetTrain];
	CheckForTuplePorts[Outputs[net], NetTrain];

	If[!PositiveMachineIntegerQ[batchSize] && batchSize =!= Automatic,
		ThrowFailure["optpi", BatchSize, batchSize];
	];

	$targetTime = None;
	maxTrainingRounds = Replace[maxTrainingRounds,
		q:DurationP :> ($targetTime = N @ QuantityMagnitude[q, "Seconds"]; Automatic)
	]; (* ^ legacy feature now taken over by TimeGoal, but we still support it *)

	If[!PositiveMachineIntegerQ[maxTrainingRounds] && maxTrainingRounds =!= Automatic,
		ThrowFailure["optpi", MaxTrainingRounds, maxTrainingRounds];
	];
	
	If[maxTrainingRounds === Automatic && IntegerQ[maxTrainingBatches],
		If[!IntegerQ[batchSize], Return @ $Failed];
		maxTrainingRounds = Ceiling[maxTrainingBatches * (batchSize / $trainingLength)];
	];

	Switch[timeGoal,
		Automatic|None,Null,
		DurationP, $targetTime = N @ QuantityMagnitude[timeGoal, "Seconds"],
		(_ ? NumericQ) ? Positive, $targetTime = N @ timeGoal,
		_, ThrowFailure["optpi", TimeGoal, timeGoal]
	];

	If[IntegerQ[maxTrainingRounds] && maxTrainingRounds < 5, 
		$EnablePreEncoding = False,
		$EnablePreEncoding = True
	];	

	SetAutomatic[workingPrecision,
		If[ContainsQ[performanceGoal, "TrainingSpeed"] && ContainsQ[targetDevice, "GPU"], "Mixed", "Real32"]];
	$DefaultDataTypeCode = ParseWorkingPrecision[workingPrecision];
	$DefaultContext = ParseMultiContext[targetDevice];
	$DefaultMixedPrecisionQ = If[workingPrecision === "Mixed", True, False];

	If[ListQ[$DefaultContext] && batchSize === Automatic, 
		NetTrain::nomdabs = "Cannot choose a BatchSize automatically when doing multi-GPU training. Please specify one manually with NetTrain[..., BatchSize -> n].";
		ThrowFailure["nomdabs"];
	];

	(****************************************************************)
	(* initialize the net and set up an NetTrainer                   *)
	(****************************************************************)
	
	(* if divergences happen in the initial timing phase, we'll ignore them *)
	$AbnormalValueCallback = repairArray;

	(* free up as much RAM as possible, but also don't wipe the net model cache *)
	ClearCache["Internal"];

	$lossFunction = lossFunction; (* for "TrainingNet" property *)
	$net = NetInitialize[net, "SamplingFunction" -> SymbolicRandomArray];
	If[FailureQ[$net], ThrowRawFailure[$net]];

	(* pick a method *)
	If[method === Automatic,
		method = "ADAM";
	];

	If[learningRate =!= Automatic,
		method = Insert[ToList[method], LearningRate -> learningRate, 2]
	];

	$noRoundLimit = False;
	If[$targetTime =!= None && maxTrainingRounds === Automatic,
		maxTrainingRounds = Infinity;
		$noRoundLimit = True;
		(* this is to avoid having to time the network, which can
		be pretty slow *)
	];

	(* see InternalOptions.m *)
	MacroEvaluate[$internalOptionsSetup];

	(* this is to make sure our first trainer doesn't have a batch size
	that exceeds the data length, so we can avoid reshaping the first trainer.
	mainly an optimizer for small example NetTrain usage *)
	$trainingLength = GuessDataLength[data];
	$validationLength = None;
	If[$trainingLength === 0, ThrowFailure["invtdata"]];

	(* for ValidationSet -> Scaled[..], pick a maxBatchSize that will allow 
	us to allocate an automatic ValidationSet of the right size. Only kicks 
	in for VERY small training data set sizes *)
	maxBatchSize = $trainingLength;
	If[MatchQ[validationSet, Scaled[n_ /; 0 <= n <= 1]] && batchSize === Automatic,
		maxBatchSize = Ceiling[First[validationSet] * $trainingLength];
	];

	If[!FreeQ[returnSpec, $executorProps], $lastPlan = Automatic]; 
	(* ^ the Automatic triggers the saving of plan by ToNetExecutor *)

	$maxBatches = $trainingLength; (* dummy value until fixed later *)
	setTrainer[] := (
		$trainer = ToNetTrainer[
			$net, Replace[r_Rule :> {r}] @ $lossFunction, 
			"BatchSize" -> batchSize, "MaxBatchSize" -> maxBatchSize,
			(* Cannot use more GPUs than the batch size *)
			"Context" -> If[ListQ[$DefaultContext],
				$DefaultContext[[;; UpTo[Min[maxBatchSize, If[IntegerQ[batchSize], batchSize, Infinity]]]]],
				$DefaultContext
			],
			"DataType" -> $DefaultDataTypeCode,
			"MixedPrecisionQ" -> $DefaultMixedPrecisionQ,
			"Optimizer" -> method, "LearningRateMultipliers" -> learningRateMultipliers,
			"TotalBatches" -> Hold[$maxBatches], "UpdatesPerBatch" -> $updatesPerBatch,
			"Measurements" -> trainingProgressMeasurements,
			"Validation" -> $doValidation, "ErrorHandler" -> NetTrainInputErrorHandler
		];
	);
	setTrainer[];

	If[$trainer === None,
		If[learningRateMultipliers =!= Automatic,
			Message[NetTrain::netfrozen],
			Message[NetTrain::netnoparams]
		];
		Return @ If[returnSpec === "TrainedNet", net, $Failed];
	];

	If[MatchQ[returnSpec, $executorProps], Return @ Replace[returnSpec, $returnSpecFunctions]];

	inputs = $trainer["Inputs"]; 
	batchSize = $trainer["BatchSize"];
	$measurementsInfo = $trainer["MeasurementsInfo"];

	(****************************************************************)
	(* create the training data generator                           *)
	(****************************************************************)

	(* this is an assoc containing misc info that gets passed to any user-defined
	generators. Set up by ParseTrainingData. *)
	$generatorInput = None;

	$LastGeneratorPermutation = None;

	(* compensate for the fact that the generator, if any, will itself add 1 to the batch numbers 
	due to the effect of prefetching. the first time it evals we want these to all be 1. *)
	$batch = $absoluteBatch = 0; $round = 1; $virtualRound = 0;

	(* handling and reporting of encoder errors *)
	MacroEvaluate[$inputErrorsSetup];

	(* normalize the training data, check it matches the inputs of the training net,
	create a generator, and get an exact length *)
	{$trainingGenerator, $trainingLength} = ParseTrainingData[data, inputs, batchSize, ConstructTrainingGenerator];
	originalTrainingLength = $trainingLength; (* <- if we take some for validation, we must know the original length *)

	(* if we guessed the training length wrong and the batch size is actaully bigger than
	the training length, we have to recreate the mxtrainer *)
	If[batchSize > $trainingLength,
		$maxBatches = $trainingLength;
		(* TODO: actually implement NetTrainerReshape and use here instead *)
		setTrainer[];
	];

	$batchesPerRound = Ceiling[$trainingLength / batchSize];

	$collectPerExampleLosses = !FreeQ[{origReturnSpec, trainingProgressFunction, trainingProgressReporting}, "ExampleLosses"|"FinalExampleLosses"|"ExampleLossHistories"];

	$examplePermutation = $LastGeneratorPermutation;
	If[$collectPerExampleLosses,
		(* we need to store the training permutation now because validation will change it *)
		NetTrain::noexloss = "\"ExampleLossLists\" property is not availabe when using the given form of training data.";
		If[$examplePermutation === None, ThrowFailure["noexloss"]];
	];

	(****************************************************************)
	(* process the ValidationSet option                             *)
	(****************************************************************)

	(* see Validation.m *)
	MacroEvaluate[$validationSetup];	

	(****************************************************************)
	(* pick a max training round by timing an update                *)
	(****************************************************************)

	If[maxTrainingRounds === Automatic, 
		{genTime, $nextDataBatch} = AbsoluteTiming @ $trainingGenerator[1];
		execTime = timeTrainer[];
		If[Head[$nextDataBatch] === Bucket, 
			execTime = Min[execTime, timeTrainer[]];
			(* its slow to make the first bucket, so time again *)
		];
		execTime = Which[ (* choose an repetition time adaptively *)
			execTime < 0.001, Min @ Table[timeTrainer[], 50], 
			execTime < 0.01, Min @ Table[timeTrainer[], 10],
			execTime < 0.05, Min[execTime, Table[timeTrainer[], 3]],
			execTime < 0.25, Min[execTime, timeTrainer[]],
			True, execTime
		];
		(* async data gen: we will be generating the next batch while
		training this batch, so we do Max rather than add them *)
		timeForBatch = Max[genTime, execTime, 0.0000001];
		timeForRound = timeForBatch * $batchesPerRound;

		maxTrainingRounds = If[$targetTime === None,
			Clip[Ceiling[20. / timeForRound], {10, 10000}],
			Ceiling[1.2 * $targetTime / timeForRound]
		];
	];

	$maxBatches = maxTrainingRounds * $batchesPerRound;
	If[IntegerQ[maxTrainingBatches], $maxBatches = maxTrainingBatches];

	(****************************************************************)
	(* parse specs that take intervals                              *)
	(****************************************************************)

	(* now that we know $maxBatches we can resolve the history specs (in case they wish to use
	"Interval" -> Quantity[..., "Percent"] *)
	returnSpec = Replace[returnSpec,
		histSpec:<|"Property" -> _, ___|> ? AssociationQ :> parseHistory[histSpec],
		{0, 1}
	];	

	$progressPeriodicCallback = parseTrainingProgressFunction[trainingProgressFunction];
	$checkpointPeriodicCallback = parseCheckpointSpec[trainingProgressCheckpointing];

	(****************************************************************)
	(* initialize various dynamic-scoped 'globals'                  *)
	(****************************************************************)

	If[$collectPerExampleLosses, 
		$perExampleCurrentLoss = CTable[Missing[], originalTrainingLength]];
	$batchRate = 0;
	$progressFraction = $timeRemaining = $timeElapsed = 0.;

	(* see Metrics.m *)
	MacroEvaluate[$metricStateSetup];

	(* TODO: eventually move all of the loss and validation history stuff into metric state 
	because it will no longer be a special case unto itself *)

	$bestValidationRound = None;
	$bestValidationFile = None;
	$lastValidationTime = None; $validationFileCount = 0;
	$CurrentExecutorBucketCount = 0;

	$checkpointFiles = Bag[];

	(****************************************************************)
	(* setup recording of losses and measurements                   *)
	(****************************************************************)

	(* sort the internal ones before the user ones *)
	$batchCollectors = Sort @ $batchCollectors;
	$roundCollectors = Sort @ $roundCollectors;

	$doMetrics = $measurementsInfo =!= {};

	(* setup trainingStoppingCriterion, see StoppingCriterion.m *)
	(* this has to be done after both validation setup and the above line because it relies on the $doValidation and $doMetrics globals *)
	(* this needs to happen after $metricStateSetup, because it needs to see what metric keys are available *)
	MacroEvaluate[$stoppingCriterionSetup];

	(****************************************************************)
	(* setup globals that give access to current net                *)
	(****************************************************************)

	(* see Properties.m *)
	MacroEvaluate[$netAccessVariablesSetup];

	(****************************************************************)
	(* time-related globals                                         *)
	(****************************************************************)	

	$timeElapsed := $now - $startTime;
	$meanBatchesPerSecond := $absoluteBatch / $timeElapsed;
	(* ^ during training use the $timeElapsed global *)

	If[$targetTime === None,
		$timeRemaining := (1.0 - $progressFraction) / (Max[$progressFraction, $MachineEpsilon] / $timeElapsed);
		$rawProgressFraction := N[$absoluteBatch / $maxBatches];
	,
		$timeRemaining := Round[Max[$targetTime - $timeElapsed, 0.]];
		$rawProgressFraction := Clip[N[$timeElapsed / $targetTime], {0, 1}];
	];

	(****************************************************************)
	(* setup internal periodic callbacks                            *)
	(****************************************************************)	

	$memoryUsagePeriodicCallback = If[$reportMemoryUsage || $MemoryUsageLogger =!= Hold,
		makePeriodicFunction[
			TrainingProgressReporting, Function[
				$memusage = MemoryUsageInfo[];
				$MemoryUsageLogger[$memusage];
				If[$reportMemoryUsage, $lastMemoryUsageInfoString = MemoryUsageInfoString[$memusage]]
			], 
			"Interval" -> $memoryReportingInterval,
			$defTimerUnit -> 3
		],
		Hold
	];

	$timingPeriodicCallback = If[$TimingLogger =!= Hold,
		makePeriodicFunction[
			"TimingLogging", Function[$TimingLogger[$timingAssociation]], 
			"Interval" -> 10,
			$defTimerUnit -> 3
		]
	];

	(****************************************************************)
	(* spin up the progress reporting window / print callback       *)
	(****************************************************************)

	(* see Reporting.m *)
	MacroEvaluate[$progressReportingSetup];

	$hardAbort = $softAbort = $shouldStop = $manualStop = False;

	(* we didn't want divergences to be handled before, but now we do *)
	$AbnormalValueCallback = throwNetDiverge;
	
	(****************************************************************
    Explanation of abort flags:
	
	* $shouldStop is always True by the end. it's just used to exit the training loops.

	* $manualStop says if either the Stop button was pressed or a callback returned "StopTraining".
	when this is true, a final validation round will not be performed.

	* $softAbort says if a callback function said to abort training, either through
	returning "AbortTraining" or through calling Abort[]. This returns $Aborted, rather than
	produce a top-level Abort[], so its a form of 'soft abort'.

	* abortedQ says whether either a hard abort (the user did CMD+.) or a soft abort 
	(callback said to abort) occurred. If the user did CMD+., the current net is returned the first
	time. Subsequent hard aborts within a short time window produce a top-level Abort. 
	A soft abort by contrast always causes $Aborted to be returned.
	*)

	(****************************************************************)
	(* main training loop                                           *)
	(****************************************************************)

	MXSeedRandom[mxSeed];
	TimeLabel["TrainingLoop"];

	handleEvent["TrainingStarted"];

	(* init first data batch *)
	$currIndex = $nextIndex = $batchperm[[1]];

	$nextDataBatch = $trainingGenerator[$nextIndex];
	$batch = $round = $absoluteBatch = 1;

	{$totalTrainingTime, $hardAbort} = AbsoluteTiming @ CheckAbort[
		(* see CoreLoop.m *)
		MacroEvaluate[$coreTrainingLoop];
		False
	, 	
		If[$progCell =!= None && FreeQ[returnSpec, "$TrainingProgressPanelImage"], 
			NotebookDelete[$progCell]; $progCell = None];
		True
	];

	(****************************************************************)
	(* post training                                                *)
	(****************************************************************)

	TimeLabel["Cleanup"];
	TrainerFinalize[$trainer];
	abortedQ = $hardAbort || $softAbort;

	$meanBatchesPerSecond = $absoluteBatch / $totalTrainingTime;
	(* ^ after training use the $totalTrainingTime global *)

	(* ensure that checkpointing and validation happens one final time (though
	it won't happen if it already happened for this batch, hence the -1) *)
	$futureClock = {$absoluteBatch-1, Infinity, Infinity};

	If[!abortedQ && !$manualStop,
		If[!$shouldStop, $batch = 0];
		(* if the last round actually completed, then reset $batch *)
		$validationPeriodicCallback[$futureClock];
		$checkpointPeriodicCallback[$futureClock];		
	];

	If[abortedQ,
		handleEvent["TrainingAborted"],
		handleEvent["TrainingComplete"]
	];
	
	If[$progCell =!= None && FreeQ[returnSpec, "$TrainingProgressPanelImage"], 
		NotebookDelete[$progCell]];
	
	If[$doPrinting, 
		If[abortedQ, 
			Print["Training aborted."], 
			Print["Training complete."]
		]
	];

	If[$AllowEarlyStoppingWithAbort && abortedQ && !$softAbort,
		If[$LastAbortedLine =!= $Line && SessionTime[] > $LastAbortedTime + 5.0, abortedQ = False];
		$LastAbortedLine ^= $Line;
		$LastAbortedTime ^= SessionTime[];
	];

,
	RandomSeeding -> randomSeeding
]; (* <- BlockRandom *)

	TimeLabel["CalculateResult"];
If[abortedQ, 
		If[$softAbort, $SoftAborted, $Aborted]
	,
		$finalNet = MemberQ[ToList @ returnSpec, "FinalNet"];
		Replace[returnSpec, $returnSpecFunctions, If[listOfSpecsQ, {1}, {0}]]
	]
];

DeclareArgumentCount[NetTrain, {1, 3}];

PackageScope["$LastDivergentArray"]

NetTrain::arrdiv = "Training was stopped early because one or more trainable parameters of the net diverged. `` To avoid divergence, ensure that the training data has been normalized to have zero mean and unit variance. You can also try specifying a lower learning rate, or use a different optimization method; the (possibly automatic) values used were ``, ``. Alternatively, you can use the \"GradientClipping\" option to Method to bound the magnitude of gradients during training."

throwNetDiverge[arr_] := Scope[
	$LastDivergentArray ^= arr;
	If[$doValidation, 
		extraInfo = "The net with the lowest validation loss will be returned.";
	,
		extraInfo = "As no ValidationSet was provided, the most recent net will be returned, which is likely to be unusable.";
	];
	$AbnormalValueCallback ^= repairArray;
	Message[NetTrain::arrdiv, extraInfo, Method -> $LastOptimizerMethod, LearningRate -> $LastInitialLearningRate];
	$manualStop ^= $shouldStop ^= True;
	$reasonTrainingStopped = "WeightsDiverged";
	handleEvent["WeightsDiverged"];
	repairArray[arr]
];

repairArray[arr_] := ConstantArray[0., arrayDimensions[arr]];

timeTrainer[] := 
	First @ AbsoluteTiming[
		TrainerUpdate[$trainer, $nextDataBatch];
		TrainerCurrentLoss[$trainer];
		NDArrayWaitForAll[];
	];


SetHoldFirst[CatchEncodeFailure];
CatchEncodeFailure[body_] := 
	Catch[body, EncodeFail, $Failed&];

(* aux functions *)


PackageScope["$NetTrainRandomDataSize"]

$NetTrainRandomDataSize = 512;


PackageExport["$AllowEarlyStoppingWithAbort"]

If[!FastValueQ[$AllowEarlyStoppingWithAbort], $AllowEarlyStoppingWithAbort = True];

NDArrayGetNoDiverge[nd_NDArray] := NDArrayGetNormal[nd];
	

SetHoldFirst[updateTimings];
updateTimings[var_Symbol, t_] := var = Replace[var, {
	{last_, min_, max_, total_, count_} :> {t, Min[min, t], Max[max, t], total + t, count + 1},
	_ :> {t, t, t, t, 1}
}];

