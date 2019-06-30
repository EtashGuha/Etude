Package["NeuralNetworks`"]


(* Note: this is substituted into the definition of NetTrain using a MacroEvaluate *)
$netAccessVariablesSetup := Quoted[
	$trainerArrays = $trainerGradients = Null;
	setupCurrentNetData[];
];

setupCurrentNetData[] := (
	$trainerArrays = KeyMap[FromNetPath, $trainer["Arrays"]];
	$trainerGradients = KeyMap[FromNetPath, $trainer["ArrayGradients"]];
);

$currentNet := TrainerCurrentNet[$trainer];	
$currentArrays := Map[NDArrayGet, $trainerArrays];
$currentArrayGradients := Map[NDArrayGet, $trainerGradients];
$currentArraysRMS := Map[NDArrayGetNormal /* RMSEnergy, $trainerArrays];
$currentArrayGradientsRMS := Map[NDArrayGetNormal /* RMSEnergy, $trainerGradients];		
$currentArraysFlat := toNumericArray[arrayFlatten @ Map[NDArrayGet, Values @ $trainerArrays]];
$currentArrayGradientsFlat := toNumericArray[arrayFlatten @ Map[NDArrayGet, Values @ $trainerGradients]];

getBestNet[] := (
	getFinalNet[]; (* in case we are asked for the best net AND THEN final net *)
	If[$doValidation && ($bestValidationFile =!= None), TrainerLoadCheckpoint[$trainer, $bestValidationFile]];
	$currentNet
);

getFinalNet[] := (
	If[TrueQ[$finalNet], $finalNet = $currentNet];
	$finalNet
);

(* ToTrainer will prepend a measurement assoc to the user-provided ones that 
corresponds to the total loss, hence the First below *)
$batchLoss := First @ $batchMeasurementsValues;
$roundLoss := First[$roundMeasurements, None];
$validationLoss := First[$validationMeasurements, None];
$batchLossList := BagContents[First @ $batchMeasurementsBags];
$roundLossList := BagContents[First @ $roundMeasurementsBags];
$validationLossList := BagContents[First @ $validationMeasurementsBags];

$thisBatchPermutation := $examplePermutation[[$currIndex]];

recordCurrentExampleLosses[] := Scope[
	pos = $thisBatchPermutation;
	$perExampleCurrentLoss[[pos]] = Total @ TrainerCurrentLoss[$trainer];
]

$timingAssociation := <|
	"Batch" -> $batchTimings,
	"Executor" -> $executorTimings,
	"Synchronize" -> $syncTimings,
	"Generator" -> $generatorTimings,
	"Collector" -> $collectorTimings,
	"Reporting" -> $reportingTimings,
	"Callbacks" -> $callbackTimings
|>;

toHumanTimingAssoc[{last_, min_, max_, total_, count_}] := 
	<|"Last" -> last, "Min" -> min, "Max" -> max, "Mean" -> N[total / count]|>;

toClosure[e_] := Block[{varsets = {}, e2, newvar},
	e2 = ReplaceAll[e, t_Symbol ? System`Private`HasImmediateValueQ :> RuleCondition[
		newvar = Unique[SymbolName[Unevaluated[t]]]; SetAttributes[newvar, Temporary];
		AppendTo[varsets, Hold[Set][newvar, t]];
		newvar
	]];
	ReleaseHold[varsets];
	e2
];

(* this object holds all the properties that callbacks are allowed to query during training.
it also provides the properties that are available after training via the third argument of 
NetTrain, though one or two of them are overridden *)

$livePropertyData = Association[

	(* nets *)
	"Net" :> $currentNet,
	"TrainingNet" :> $trainer["TrainingNet"],

	(* measurement-related properties *)
	"BatchMeasurements" :> $batchMeasurements,
	"RoundMeasurements" :> $roundMeasurements,
	"ValidationMeasurements" :> $validationMeasurements,
	"BatchMeasurementsLists" :> $batchMeasurementsLists,
	"RoundMeasurementsLists" :> $roundMeasurementsLists,
	"ValidationMeasurementsLists" :> $validationMeasurementsLists,
	"ExampleLosses" :> $perExampleCurrentLoss,

	(* these are derived from Round/Batch/ValidationMeasurements[Bag], so they are technically just sugar *)
	"BatchLoss" :> $batchLoss,
	"RoundLoss" :> $roundLoss,
	"ValidationLoss" :> $validationLoss, 
	"BatchLossList" :> $batchLossList, 
	"RoundLossList" :> $roundLossList,
	"ValidationLossList" :> $validationLossList,

	(* batchsize, batch, round info *)
	"TotalRounds" :> maxTrainingRounds,
	"TotalBatches" :> $maxBatches, 
	"Round" :> $round, 
	"Batch" :> $batch, 
	"AbsoluteBatch" :> $absoluteBatch,
	"BatchSize" :> batchSize,
	"BatchesPerRound" -> $batchesPerRound,
	"BatchPermutation" :> $examplePermutation,
	"BatchData" :> $LastGeneratorData,

	(* metadata about training run *)
	"ProgressFraction" :> $progressFraction,
	"ExamplesProcessed" :> batchSize * $absoluteBatch, 
	"TrainingExamples" :> $trainingLength,
	"ValidationExamples" :> $validationLength,
	"TimeRemaining" :> $timeRemaining, 
	"TimeElapsed" :> $timeElapsed,
	"Event" :> $lastEvent,
	"BatchesPerSecond" :> $batchRate, 
	"ExamplesPerSecond" :> $updatesPerBatch * $batchRate * batchSize,	
	"MeanBatchesPerSecond" :> $meanBatchesPerSecond,
	"MeanExamplesPerSecond" :> $meanBatchesPerSecond * $updatesPerBatch * batchSize,	
	"NetTrainInputForm" :> HoldForm[NetTrain][Placeholder["net"], Placeholder["data"], $restArgs],
	"BestValidationRound" :> $bestValidationRound,

	"SkippedTrainingData" :> <|
		"SkippedExampleIndices" -> BagContents[$skippedExamples],
		"SkippedBatchIndices" -> Keys @ $batchWasSkipped,
		"SkippedExamples" -> $skippedExampleCount
	|>,

	(* metadata about optimization *)
	"LearningRate" :> MXNetLink`$LastGlobalLearningRate,
	"InitialLearningRate" :> MXNetLink`$LastInitialLearningRate,
	"FinalLearningRate" :> MXNetLink`$LastGlobalLearningRate,
	"OptimizationMethod" :> MXNetLink`$LastOptimizerMethod,
	"CheckpointingFiles" :> BagContents[$checkpointFiles],
	"WeightsLearningRateMultipliers" :> $trainer["ArrayLearningRateMultipliers"],
	"TargetDevice" -> targetDevice,

	(* these are internal / hidden properties *)
	"$Trainer" :> $trainer,
	"$TrainingGenerator" :> {toClosure @ $trainingGenerator, $batchesPerRound},
	"$ValidationGenerator" :> {toClosure @ $validationGenerator, Ceiling[$validationLength / batchSize]},
	"$MemoryUsage" :> MemoryUsageInfo[],
	"$BucketCount" :> $CurrentExecutorBucketCount,
	"$GPUInformation" :> RateLimitedGPUInformation[],
	"$Timings" :> Map[toHumanTimingAssoc, $timingAssociation],
	"$MeasurementsInfo" :> $measurementsInfo,
	"$TrainingProgressPanelImage" :> takePanelSnapshot[],
	"InternalVersionNumber" -> $NeuralNetworksVersionNumber,

	(* more obscure properties for DL nerds *)
	"Weights" :> $currentArrays, 
	"WeightsVector" :> $currentArraysFlat,
	"Gradients" :> $currentArrayGradients,
	"GradientsVector" :> $currentArrayGradientsFlat,
	"WeightsRMS" :> $currentArraysRMS,
	"GradientsRMS" :> $currentArrayGradientsRMS,

	(* final plots *)
	"FinalPlots" :> makeFinalPlots[],
	"LossPlot" :> makeFinalLossPlot[],

	(* for re-creating plots *)
	"ValidationPositions" :> Round[Internal`BagPart[$validationTimes, All]*$batchesPerRound],
	"RoundPositions" :> Range[$round]*$batchesPerRound
];

$livePropertyKeys = Keys[$livePropertyData];
$livePropertyP = Alternatives @@ $livePropertyKeys;

(* return specs are basically the live specs, but with one or two special cases
to handle things at the end *)
$returnSpecFunctions = Normal @ Association @ {

	Sequence @@ Normal[$livePropertyData],

	(* what does 'Net' mean at the end? *)
	"Net" :> Missing["Ambigious"],

	(* these load checkpoints and so shouldn't be available during training *)
	"FinalNet" :> getFinalNet[],
	"TrainedNet" :> getBestNet[], 

	(* if we stopped early, we correct the total numbers *)
	"Round" :> Min[$round, maxTrainingRounds],
	"TotalRounds" :> Min[$round, maxTrainingRounds],
	"TotalBatches" :> $absoluteBatch,

	(* these deal with history property specs that are collecting
	values over the entire run *)
	"$Bag"[bag_] :> BagContents[bag],
	"$Bag"[bag_, func_] :> func[BagContents[bag]],

	"$Executor" :> $trainer["Executor"],
	"$ExecutorPlan" :> $lastPlan,
	"$ExecutorPlanPlot" :> NetPlanPlot[$lastPlan],

	(* these only make sense at the end *)
	"Properties" :> $userVisibleProperties,
	"ResultsObject" :> makeNetTrainResultsObject[],
	"ReasonTrainingStopped" :> $reasonTrainingStopped,
	"TotalTrainingTime" :> $totalTrainingTime
};

$executorProps = "$ExecutorPlanPlot" | "$ExecutorPlan";

$returnSpecKeys = Select[Keys[$returnSpecFunctions], StringQ];

$userVisibleProperties = Sort @ DiscardStrings[$returnSpecKeys, "$*" | "Net"];

RunInitializationCode[
	Compose[FE`Evaluate, FEPrivate`AddSpecialArgCompletion["NetTrain" -> {0, 0, $userVisibleProperties}]]
]

$returnSpecP = Apply[Alternatives, Append[$returnSpecKeys, (_Association ? AssociationQ) ? (KeyExistsQ["Property"])]];

(* aliases for properies that were removed/renamed in 12 but are still available either with a different name or with a custom property *)
$resultsObjectPropertyAliases = {	
	(* renamed properties *)
	"EvolutionPlots" -> "FinalPlots",
	"FinalExampleLosses" -> "ExampleLosses",
	"FinalRoundLoss" -> "RoundLoss",
	"FinalValidationLoss" -> "ValidationLoss",
	"FinalWeights" -> "Weights",
	"FinalWeightsVector" -> "WeightsVector",
	"LossEvolutionPlot" -> "LossPlot",
	"MeanInputsPerSecond" -> "MeanExamplesPerSecond",	
	"TotalInputs" -> "ExamplesProcessed"
};
(* ^ the results object only gets the renamed properties because the other propeties are not stored at the moment *)
(* TODO: add the properties bellow to the NTRO *)

$netTrainPropertyAliases = {

	Sequence @@ $resultsObjectPropertyAliases,

	(* removed properties that can be re-created with custom properties *)
	"BatchErrorRateList" -> 
		<|"Property" -> Function[#BatchMeasurements["ErrorRate"]], "Form" -> "List", "Interval" -> "Batch"|>,
	"BatchGradientsHistories" -> 
		<|"Property" -> "Gradients", "Form" -> "TransposedList", "Interval" -> "Batch"|>,
	"BatchGradientsVectorHistories" ->
		<|"Property" -> "GradientsVector", "Form" -> "List", "Interval" -> "Batch"|>,
	"BatchWeightsHistories" ->
		<|"Property" -> "Weights", "Form" -> "TransposedList", "Interval" -> "Batch"|>,
	"BatchWeightsVectorHistories" ->
		<|"Property" -> "WeightsVector", "Form" -> "List", "Interval" -> "Batch"|>,
	"ExampleLossHistories" -> <|"Property" -> "ExampleLosses", "Form" -> "TransposedList"|>,
	"RMSGradientsEvolutionPlot" ->
		<|"Property" -> "GradientsRMS", "Form" -> "Plot", "Interval" -> "Batch", "PlotOptions" -> {ScalingFunctions -> "Log", "PlotLabels" -> True}|>,
	"RMSGradientsHistories" ->
		<|"Property" -> "GradientsRMS", "Form" -> "TransposedList", "Interval" -> "Batch"|>,
	"RMSWeightsEvolutionPlot" ->
		<|"Property" -> "WeightsRMS", "Form" -> "Plot", "Interval" -> "Batch", "PlotOptions" -> {ScalingFunctions -> "Log", "PlotLabels" -> True}|>,
	"RMSWeightsHistories" -> 
		<|"Property" -> "WeightsRMS", "Form" -> "TransposedList", "Interval" -> "Batch"|>,
	"RoundErrorRateList" ->
		<|"Property" -> Function[#RoundMeasurements["ErrorRate"]], "Form" -> "List", "Interval" -> "Round"|>,
	"RoundWeightsHistories" ->
		<|"Property" -> "Weights", "Form" -> "TransposedList", "Interval" -> "Round"|>,
	"RoundWeightsVectorHistories" ->
		<|"Property" -> "WeightsVector", "Form" -> "List", "Interval" -> "Round"|>
	(* "ValidationErrorRateList" -> 
		<|"Property" -> Function[#ValidationMeasurements["ErrorRate"]], "Form" -> "List", "Interval" -> "Round"|> *)
	(* TODO: this ^ property is a little difficult to alias for two reasons.
	1. The interval of the validation set is arbitrary. Ideally we want the interval for this property to match that
	interval. However, this is not straight forward to do because of the order in which different parts of NetTrain are exectuted.
	2. The collection of validation measurements is currently setup to happen after the calculation of custom measurements. Again there
	is a non-trivial ordering dependancy. 
	The good news is that this property can be aquired from the "ValidationMeasurementsLists" property. 
	 *)
};
