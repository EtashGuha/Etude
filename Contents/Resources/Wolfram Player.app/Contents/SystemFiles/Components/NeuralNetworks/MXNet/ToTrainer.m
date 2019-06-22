Package["NeuralNetworks`"]


PackageExport["ToNetTrainer"]

Clear[ToNetTrainer];

SetUsage @ "
ToNetTrainer[net$, lossSpec$, opts$$] takes a net and instantiates it into an NetTrainer.
ToNetTrainer[net, opts$$] will use Automatic as a loss spec.
* The resulting NetTrainer[$$] can be used to perform training, by supplying it with batches.
* NetAttachLoss[net$, lossSpec$] is used to create a training graph.
* The following options are supported: 
| 'BatchSize' | Automatic | batch size to use |
| 'Context' | 1 | the device code to use |
| 'DataType' | 0 | floating point datatype to train with |
| 'MixedPrecisionQ' | False | whether to used mixed precision training |
| 'Optimizer' | 'ADAM' | the optimizer to use |
| 'TotalBatches' | 4096 | the anticipated number of batches that will be trained |
| 'Measurements' | {} | the NetPorts & built-in metrics to support while training |
| 'ErrorHandler' | Automatic | a function to handle errors when setting input arrays |
* 'TotalBatches' is used for SGD optimizer. It can be provided as a Hold[sym] in case it \
needs to be changed after creation (SGD supports this).
* See the usage of NetTrainer for more information.
* If net$ contains variable sequences, a BucketedExecutor will be created, and \
the master bucket will start out at size 4 just to get access to gradient arrays etc."

Options[ToNetTrainer] = {
	"BatchSize" -> Automatic,
	"MaxBatchSize" -> None,
	"Context" -> 1,
	"Optimizer" -> "ADAM",
	"TotalBatches" -> 4096,
	"LearningRateMultipliers" -> Automatic,
	"UpdatesPerBatch" -> 1,
	"GradientScale" -> 1,
	"DataType" -> 0, 
	"MixedPrecisionQ" -> False,
	"Validation" -> True,
	"Measurements" -> {},
	"ErrorHandler" -> Automatic
};

PackageExport["$DefaultRNNBatchSize"]
PackageScope["$DisableValidationExecutor"]
PackageScope["$DummyTrainerSeqLen"]
PackageScope["$LowPrecisionGradientRescaling"]
PackageScope["$TrainerLogger"]

$DummyTrainerSeqLen = 3;
$DefaultRNNBatchSize = 32;
$LowPrecisionGradientRescaling = 5;
$DisableValidationExecutor = False;
$TrainerLogger = Hold;
(* ^ these three are controlled via InternalOptions in NetTrain *)

ToNetTrainer[net_, opts___Rule] := 
	ToNetTrainer[net, Automatic, opts];

ToNetTrainer[net_NetP, lossSpec_, OptionsPattern[]] := Timed @ Scope[
	UnpackOptions[
		batchSize, maxBatchSize, context, dataType, mixedPrecisionQ, optimizer, totalBatches, 
		learningRateMultipliers, updatesPerBatch, gradientScale, measurements, validation, 
		errorHandler
	];
	If[!FullySpecifiedNetQ[net], Panic["NetNotInitialized"]];
	
	$TrainerLogger[{"ToNetTrainerStarted", batchSize, context, dataType, optimizer}];
	CheckPortsForBadLengthVars[net];
	NetFinalCheck[net];
	
	userMeasurementPorts = GetUserMeasurementPorts[OutputNames[net], measurements];
	{trainNet, lossPorts, $hasPrefix} = AttachLoss[net, lossSpec, userMeasurementPorts];	
	If[!FullySpecifiedNetQ[trainNet], ThrowNotSpecifiedFailure[trainNet, "train", "Defined"]];

	measurementsInfo = ParseMeasurementSpecs[trainNet, measurements, $hasPrefix];
	(* ^ now that we have the loss network, fill in the missing metric info *)

	lossLayers = DeepCases[
		If[AssociationQ[trainNet], trainNet, NData[trainNet]], 
		<|"Type" -> t_, ___|> /; $LayerData[t, "IsLoss"]
	];

	(* figure out the per-array learning rates, as assoc from NetPath to value *)
	If[learningRateMultipliers === Automatic,
		learningRateMultipliers = 1.& /@ NetArrays[trainNet];
		pathsWithGradients = Keys[learningRateMultipliers];
	,
		learningRateMultipliers = toMXLearningRates[net, learningRateMultipliers];
		pathsWithGradients = Keys @ Select[learningRateMultipliers, # > 0 || # < 0&];
	];
	If[pathsWithGradients === {}, Return[None]]; (* <- signal that there is nothing to train *)
	
	checkRequestedMetrics[trainNet];

	bucketed = ContainsVarSequenceQ[Inputs[trainNet]];

	gradScale = If[dataType == 2, $LowPrecisionGradientRescaling, 1];
	$gradientData = {<||>, pathsWithGradients, True, gradScale};
	(* an important detail here is that ToNetExecutor will see a list of gradient paths (instead of an
	assoc from arb key to mx spec) and it will look those paths up in the NetPlan's LogicalWeights to
	get an mx spec *)

	measurementPaths = DeleteCases[measurementsInfo[[All, "Path"]], _CallbackMetric];
	$requiresValidationExecutor = NetHasTrainingBehaviorQ[trainNet] && TrueQ[validation] && !$DisableValidationExecutor;

	$TrainerLogger["CreatingDummyExecutor"];
	validationSyncFunction = Hold;
	If[!bucketed,
		If[Inputs[trainNet] === <||> && batchSize === Automatic, batchSize = 1];
		{executor, validationExecutor} = makeTrainingAndValidationExecutors[
			plan = ToNetPlan[trainNet, {lossPorts, <||>, $tvmode, {Min[context] =!= 1, dataType, mixedPrecisionQ}, measurementPaths}]; 
			SetAutomatic[batchSize, ChooseTrainingBatchSize[plan, pathsWithGradients, {context, dataType, mixedPrecisionQ}]];
			batchSize = maxBatch[batchSize, maxBatchSize];
			ToNetExecutor[
				plan, batchSize, "DataType" -> dataType, "MixedPrecisionQ" -> mixedPrecisionQ,
				"GradientData" -> $gradientData, "Context" -> context, "ArrayCaching" -> $arrayCacheVar
			]
		];
		If[$requiresValidationExecutor, 
			validationSyncFunction = validationExecutor["SyncFunction"];
		];
		isMulti = Head[executor] === MultiExecutor;
		executorData = Normal[executor];
	,
		SetAutomatic[batchSize, $DefaultRNNBatchSize];
		batchSize = maxBatch[batchSize, maxBatchSize];
		{executor, validationExecutor} = makeTrainingAndValidationExecutors[
			ToBucketedNetExecutor[trainNet, 
				{lossPorts, {context, dataType, mixedPrecisionQ}, batchSize, $tvmode, measurementPaths, $gradientData, $arrayCacheVar}
			]
		];
		dummyBucket = Values @ MakeSeqIDLens[trainNet, $DummyTrainerSeqLen];
		dummyExecutor = Block[{$MXNetDisablePanics = True}, CatchFailure @ GetBucketExecutor[executor, dummyBucket]];
		If[FailureQ[dummyExecutor], 
			dummyBucket = FindMinimalLengthVarSettings[trainNet];
			dummyExecutor = GetBucketExecutor[executor, dummyBucket];
		];
		If[$requiresValidationExecutor, 
			validationSyncFunction = GetBucketExecutor[validationExecutor, dummyBucket]["SyncFunction"];
		];
		(* ^ force the first validation executor to exist, so that we can guarantee that a sync happens 
		between possible fused arrays in the validation executor and non-fused parts in the training executor *)
		isMulti = Head[dummyExecutor] === MultiExecutor;
		executorData = Normal[dummyExecutor];
		(* ^ we want the GradientArrays NOW so we force the bucketed allocator
		to allocate an initial dummy master executor. there is a subtle business here:
		we can't discard this first minimal bucket, but we would like to, because the 
		generator has arranged to put its biggest bucket first, and that SHOULD be our
		master bucket. instead we rely on the bucketed generator maintaining a 'largest
		key' bucket, if it is totally dominated, we will create a new master. 
		TODO: require the max bucket size be passed into the ToTrainer directly to avoid
		these shenanigans. *)
	];

	UnpackAssociation[executorData["Arrays"], weightGradients, weights];
	(* ^ at these point these have been bound to NDArrays, so we have assocs
	from NetPaths to NDArrays and these will never change. The weights were
	actually derived from the list of NetPaths was passed as the wgrad GradientData,
	so weights and weightGradients have the same keys. *)

	kvStore = If[isMulti, MXKVStoreCreate[$DefaultKVStoreType], None];

	weightNames = Keys[weights];

	(* get per-array adjustments to L2Regularization, GradientClipping, and WeightClipping *)
	$TrainerLogger["CreatingOptimizer"];
	Block[{$net = net}, 
		optimizer = CreateArrayOptimizer[
			optimizer /. LearningRate -> "LearningRate", 
			Values @ weights, Values @ weightGradients, 
			"KVStore" -> kvStore, "GradientKeys" -> weightNames, 
			"LearningRateMultipliers" -> Lookup[learningRateMultipliers, weightNames],
			"GradientScale" -> gradScale * batchSize, "MaxIterations" -> totalBatches,
			"Resolver" -> methodParamResolver
		];
	];

	(* build a new net that contains NDArrays *)
	auxilliaryArrays = executorData["MXAuxArrays"];
	arrays = KeyMap[removePrefix, Join[KeyMap[MXUnmanglePath, auxilliaryArrays], weights]];
	arrayGradients = KeyMap[removePrefix, weightGradients];
	
	(* ^ mapping from possibly prefixed NetPath to NDArray *)
	net = ReplacePart[net, KeyValueMap[Apply[List, #] -> #2&, arrays]];
	System`Private`SetValid[net];

	checkpointArrays = Map[
		Lookup[arrays, #1, $dummyCheckpointArray]&,
		Keys @ NetArrays @ net
	]; 
	(* ^ the idea here is that any arrays without a corresponding grad shouldn't be
	checkpointed, but because the 'params' file doesn't contain a mapping, we need them
	to exist as placeholders to make the ndarrays on disk line up with those in the net,
	so instead we replace them with a tiny dummy array. *)

	inputArrays = executorData["Arrays", "Inputs"];
	outputArrays = executorData["Arrays", "Outputs"];
	metricArrays = executorData["Arrays", "Metrics"];

	NDArraySetConstant[metricArrays, 0.0];

	logicalLearningRates = KeyMap[removePrefix /* FromNetPath, learningRateMultipliers];

	If[Length[outputArrays] === 1, 
		lossMetricData = {makeLossMetricData["Loss", First @ Keys @ outputArrays]}
	,
		NetTrain::multilossname = "A net with multiple losses cannot have an output port named \"Loss\".";
		If[MemberQ[Keys[outputArrays], "Loss"], ThrowFailure["multilossname"]];
		(* ^ throw message if any of the output arrays are called 'Loss' in the multi-output net. See 363344. *)
		lossMetricData = Prepend[
			makeLossMetricData[#1, #1]& /@ Keys[outputArrays],
			makeLossMetricData["Loss", "TotalLoss"]
		];
	];
	measurementsInfo = Join[lossMetricData, measurementsInfo];

	SetAutomatic[errorHandler, InputErrorHandler[#1, #2[#1]]&];

	If[$TrainerLogger =!= Hold, 
		optimizer = $TrainerLogger["CallOptimizer"]& /* optimizer
	];
	inputs = Inputs[trainNet];
	$TrainerLogger["ToNetTrainerFinished"];

	NetTrainer @ Association[
		"Executor" ->			executor, 
		"ValidationExecutor" -> validationExecutor,
		"ValidationSyncFunction" -> validationSyncFunction,
		"Inputs" ->				inputs,
		"TrainingNet" -> 		trainNet,
		"LossLayers" -> 		lossLayers,
		"OutputNames" ->		Keys[outputArrays],
		"Optimizer" ->			optimizer,
		"BatchSize" ->			batchSize,
		"Context" ->			context,
		"MutableNet" ->			net, (* XXXX: How does MutableNet interact with MultiExecutor? *)
		"FullMutableNet" ->		ConstructNet[net],
		"Bucketed" ->			bucketed,
		"CurrentNet" ->			Module[{sym}, sym = None; Hold[sym]],
		"Arrays" ->				arrays,
		"ArrayGradients" ->		arrayGradients,
		"CheckpointArrays" ->	checkpointArrays,
		"MetricArrays" -> 		Module[{metsym}, metsym = metricArrays; Hold[metsym]],
		"OutputArrays" ->		Module[{outsym}, outsym = outputArrays; Hold[outsym]],
		"InputArrays" ->		Module[{insym}, insym = inputArrays; Hold[insym]],
		"ArrayLearningRateMultipliers" -> logicalLearningRates,
		"UpdatesPerBatch" ->	updatesPerBatch,
		"MeasurementsInfo" -> 	measurementsInfo,
		"ErrorHandler" ->       errorHandler
	]
];

(* macro that creates either one executor or two *)

SetHoldFirst[makeTrainingAndValidationExecutors];
makeTrainingAndValidationExecutors[body_] := Scope @ Module[
	{arrayCache = <||>},
	(* we create an arraycache here so that the training and validation executors can share arrays *)
	$arrayCacheVar = Hold[arrayCache];
	$tvmode = True; 
	exec = body;
	If[!$requiresValidationExecutor, 
		{exec, exec}
	,
		$gradientData ^= {<||>, <||>, "SharedViaCache", 1}; 
		(*$gradientData[[3]] = "SharedViaCache";*)
		$tvmode = False;
		exec2 = body;
		{exec, exec2}
	]
]

$dummyCheckpointArray := $dummyCheckpointArray = MXNetLink`NDArrayCreate[CTable[0, {1,1,1,1,1}], 1];

maxBatch[n_, None] := n;
maxBatch[n_, m_] := Min[n, m];

unmangleToList[key_] := FromNetPath @ unmangleToPath[key];
unmangleToPath[key_] := removePrefix @ MXUnmanglePath[key];

(* VVV because the original net got its shared arrays hoisted out into the training net, so the usual prefix
isn't there. *)
removePrefix[w:NetPath["SharedArrays", _]] := w; 
removePrefix[path_] := If[!$hasPrefix, path, Drop[path, 2]];

PackageScope["addPrefix"]

addPrefix[w:NetPath["SharedArrays", _]] := w;
addPrefix[path_] := Join[NetPath["Nodes", "Net"], path];

methodParamResolver[key_, spec_, checker_, noneValue_, defaultValue_] := Scope[
	spec = MapAt[checker, spec, {All, 2}];
	DeepCases[spec, $Failed[reason_] :> Return[Compose[$Failed, reason <> ", or a list of rules with these values"], Block]];
	NetArrayCases[$net, spec, key, noneValue, defaultValue]
];

_methodParamResolver := $Unreachable;


toMXLearningRates[net_, spec_] :=
	NetArrayCases[net, spec, LearningRateMultipliers, 0.0, 1.0];

NetArrayCases[net_NetP, spec_, head_, noneValue_, defaultValue_] := Scope[
	$nacasesHead = head; $noneValue = noneValue; $defaultValue = defaultValue;
	If[!ListQ[spec], 
		General::arrpattnlist = "Specification for `` should be a list of rules from layer specs to values.";
		ThrowFailure["arrpattnlist", $nacasesHead];
	];
	$net = net;
	$lrrules = Dispatch @ Append[Map[parseArraySpec, spec], _ -> $defaultValue];
	$arrayCaseResults = Association[]; 
	If[KeyExistsQ[$net, "SharedArrays"],
		(* because multiple arrays can correspond to one shared array, we have to decide which value
		should win -- there will always be at least the _ -> 1 case, so we can't just pick the value
		chosen  for first usage of a shared array. so instead we record which rule matched ('priority'),
		and we pick the left-most rule that matched any of the usages of a shared array. *)
		$saprio = Association[];
		$priorules = Normal[$lrrules];
		$priorules[[All, 2]] = Range[Length[$priorules]];
		(* explicit rules for NetSharedArray require a little juggling to re-use the same code *)
		ScanFields["SharedArrays", sowArrayCase[NetSharedArray @ Last @ $path]&, $net];
	];
	sowArrayCases[net];
	If[!$hasPrefix, $arrayCaseResults, KeyMap[addPrefix, $arrayCaseResults]]
	(* despite the name 'learningRates', this is also used to attach per-layer parameter
	adjustments besides the learning rate *)
];

(* because technically _ doesn't match the root, i guess... *)
parseArraySpec[Verbatim[_] -> rate_] := parseArraySpec[___ -> rate];

parseArraySpec[lspec_ -> rate_] := Scope[
	patt = ToNetPathPattern[$net, lspec];
	If[FailureQ[patt], 
		General::arrpattmiss = "The left hand side `` given as part of the setting for `` is not a valid position within the net.";
		ThrowFailure["arrpattmiss", lspec, $nacasesHead];
	];
	Which[
		rate === None, rate = $noneValue,
		NumericQ[rate], rate = N[rate],
		True, 
			General::arrpattnnum = "The right hand side `` given as part of the setting for `` is not a numeric value or None.";
			ThrowFailure["arrpattnnum",  rate, $nacasesHead];
	];
	Append[patt, ___] -> rate
];

parseArraySpec[_] := ThrowFailure["arrpattnlist", $nacasesHead];


DeclareMethod[sowArrayCases, sowArrayCaseMultipliers, Inherited, sowOperatorArrayCases];

sowArrayCaseMultipliers[layer_] :=
	ScanFields["Arrays", sowArrayCase, layer];

sowOperatorArrayCases[assoc_] := (
	sowArrayCaseMultipliers[assoc];
	ScanSubNets[sowArrayCases, assoc];
)

sowArrayCase[NetSharedArray[sname_]] := Scope[
	oprio = $saprio[sname];
	nprio = Replace[$path, $priorules];
	If[MissingQ[oprio] || nprio < oprio,
		$arrayCaseResults[NetPath["SharedArrays", sname]] = Replace[$path, $lrrules];
		$saprio[sname] = nprio;
	];
];

sowArrayCase[_] := 
	$arrayCaseResults[$path] = Replace[$path, $lrrules];


PackageExport["TrainerFinalize"]

TrainerFinalize[_] := (
	$TrainerLogger["TrainerFinalize"];
	NDArrayWaitForAll[]; 
	(* this prevents a crazy crash, presumably NDArrays are cleared before the trainer is done with the final
	optimization step or something *)
)


PackageExport["TrainerUpdate"]

SetUsage @ "
TrainerUpdate[NetTrainer[$$], <|'port$1'->input$1,$$|>] sets the input arrays of the trainer, applies a \
forward step, then a backward step, and then does a weight update via the internal optimizers.
TrainerUpdate[NetTrainer[$$], Bucket[data$, key$]] looks up a bucketed executor to use based on key$.
* the input$i should have encoding already performed, NetTrainer[$$] will ignore the net's encoders
* For the bucketed case, the trainer must have been created with the 'MaxLengths' set."

TrainerUpdate[NetTrainer[assoc_], data_Association] := Timed @ Scope[
	$TrainerLogger["TrainerUpdateStarted"];
	exec = assoc["Executor"];
	n = assoc["UpdatesPerBatch"];
	HoldSet[assoc["CurrentNet"], None];
	PreemptProtect[
		setInputArrays[assoc["InputArrays"], data, assoc["Inputs"], assoc["ErrorHandler"]];
		Do[
			$TrainerLogger["ForwardBackward"];
			NetExecutorForward[exec, True];
			NetExecutorBackward[exec];
			assoc["Optimizer"][],
			n
		]
	];
	$TrainerLogger["TrainerUpdateFinished"];
];

TrainerUpdate[NetTrainer[assoc_], Bucket[data_Association, key_]] := Timed @ Scope[
	$TrainerLogger["TrainerUpdateStarted"];
	HoldSet[assoc["CurrentNet"], None];
	n = assoc["UpdatesPerBatch"];
	PreemptProtect[
		exec = GetBucketExecutor[assoc["Executor"], key];
		HoldSet[assoc["OutputArrays"], exec["Arrays", "Outputs"]];
		HoldSet[assoc["MetricArrays"], exec["Arrays", "Metrics"]];
		HoldSet[assoc["InputArrays"], exec["Arrays", "Inputs"]];
		setInputArrays[assoc["InputArrays"], data, assoc["Inputs"], assoc["ErrorHandler"]];
		Do[
			$TrainerLogger["ForwardBackward"];
			NetExecutorForward[exec, True];
			NetExecutorBackward[exec];
			assoc["Optimizer"][];
		,
			n
		];
	];
	$TrainerLogger["TrainerUpdateFinished"];
];

setInputArrays[Hold[arrays_], data_, types_, errorHandler_] := 
	KeyValueScan[
		NDArraySetBatched[#2, data[#1], errorHandler[#1, types]]&,
		arrays
	]

NetTrainer[assoc_][p_] := assoc[p];
NetTrainer[assoc_][p_, rest__] := assoc[p][rest]; (* first get out ExecutorData *)


PackageExport["TrainerCurrentLoss"]

TrainerCurrentLoss[NetTrainer[assoc_]] := (
	$TrainerLogger["TrainerCurrentLoss"];
	NDArrayGetNormal /@ ReleaseHold @ assoc["OutputArrays"]
)

PackageExport["TrainerCurrentMetrics"]

TrainerCurrentMetrics[NetTrainer[assoc_]] := (
	$TrainerLogger["TrainerCurrentMetrics"];
	NDArrayGetTotalNormal /@ ReleaseHold @ assoc["MetricArrays"]
)

PackageExport["TrainerDoValidation"]

SetUsage @ "
TrainerDoValidation[NetTrainer[$$}, generator$, n$] returns a pair {losses$, errors$}.
* losses$ is an association of average losses, where generator$ generates batches in order \
to provide a total of n$ individual examples.
* errors$ is an association of average errors.
* generator$ should return batches as a list containing one array for every input port.
* generator$ should use a batchsize identical to that used to set up the trainer."

TrainerDoValidation[NetTrainer[assoc_], generator_, n_] := Scope[
	
	$TrainerLogger["TrainerDoValidation"];
	UnpackAssociation[assoc, validationSyncFunction, validationExecutor, batchSize, outputNames, measurementsInfo];

	validationSyncFunction[];
	(* this copies arrays over, if necessary, from the training to the validation executor. this currently
	is only needed when training RNN doesn't fuse (due to dropout), but validation does *)

	ExecutorMeasure[validationExecutor, generator, n, batchSize, outputNames, measurementsInfo]
]


PackageExport["TrainerSaveCheckpoint"]
PackageExport["TrainerLoadCheckpoint"]

SetUsage @ "
TrainerSaveCheckpoint[NetTrainer[$$],'file$'] saves all of the trainer's arrays to file$.
TrainerSaveCheckpoint[NetTrainer[$$], Hold[sym$]] saves all of the trainer's arrays to a list of CPU-based arrays in sym$."

SetUsage @ "
TrainerLoadCheckpoint[NetTrainer[$$],'file$'] loads all of the trainer's arrays from file$.
TrainerLoadCheckpoint[NetTrainer[$$], Hold[sym$]] loads all of the trainer's arrays to a list of CPU-based arrays in sym$."

(* save to in-memory copy *)
TrainerSaveCheckpoint[NetTrainer[assoc_], Hold[arrays_Symbol]] := PreemptProtect[
	$TrainerLogger[{"TrainerSaveCheckpoint", "InMemory"}];
	checkpointArrays = Replace[assoc["CheckpointArrays"], NDReplicaArray[{first_, __}] -> first, {1}];
	If[!FastValueQ[arrays], arrays = Map[ndArrayClone, checkpointArrays]];
	MapThread[NDArrayCopyTo, {arrays, checkpointArrays}]
];

(* save to disk *)
TrainerSaveCheckpoint[NetTrainer[assoc_], file_] := (
	$TrainerLogger[{"TrainerSaveCheckpoint", file}];
	PreemptProtect @ NDArrayExport[file, assoc["CheckpointArrays"]];
)

(* load from in-memory copy *)
TrainerLoadCheckpoint[NetTrainer[assoc_], Hold[arrays_Symbol]] := PreemptProtect[
	HoldSet[assoc["CurrentNet"], None];
	$TrainerLogger[{"TrainerLoadCheckpoint", "InMemory"}];
	If[FastValueQ[arrays], MapThread[NDArrayCopyTo, {assoc["CheckpointArrays"], arrays}]]
];

(* load from  disk *)
TrainerLoadCheckpoint[NetTrainer[assoc_], file_] := (
	HoldSet[assoc["CurrentNet"], None];
	$TrainerLogger[{"TrainerLoadCheckpoint", file}];
	PreemptProtect @ NDArrayImport[file, assoc["CheckpointArrays"]];
)

ndArrayClone[nd_NDArray] := NDArrayCreateEmpty[NDArrayDimensions[nd], 1, NDArrayDataType[nd]];


PackageExport["TrainerCurrentNet"]

SetUsage @ "
TrainerCurrentNet[NetTrainer[$$]] substitutes the current weights \
from the trainer into the original net given to ToNetTrainer.
* There is a fair cost associated with constructing the net, \
and any evaluations performed with the net will currently cause new \
executors to be created (rather than using the training executor), \
so use with caution.
* The current net is cached after being constructed, but the cache \
is flushed when the trainer undergoes a training update."

TrainerCurrentNet[NetTrainer[assoc_]] := Scope[
	sym = assoc["CurrentNet"];
	Replace[
		First @ sym, 
		None :> HoldSet[sym,
			ConstructNet @ ReplaceAll[
				assoc["MutableNet"], {
				rep_NDReplicaArray :> RuleCondition[NDArrayGet[rep[[1,1]]]],
				nd_NDArray :> RuleCondition[NDArrayGet[nd]]
			}]
		]
	]
];


PackageExport["TrainerMemoryUsage"]

SetUsage @ "
TrainerMemoryUsage[NetTrainer[$$]] gives the memory usage of the trainer."

TrainerMemoryUsage[NetTrainer[assoc_]] := (
	$TrainerLogger["TrainerMemoryUsage"];
	Total[NetExecutorMemoryInformation[assoc["Executor"]], Infinity]
);


(* BOXES *)

PackageExport["NetTrainer"]

SetUsage @ "
NetTrainer[<|$$|>] represents an object that can be used to train a network, one batch at a time.
* Use TrainerUpdate[NetTrainer[$$], inputs$] to perform an update step, which does a forward and backward step \
and updates the weights using the optimizer.
* Use TrainerCurrentLoss[trainer$] to get the current total loss.
* Use TrainerCurrentNet[trainer$] to reconstruct the current net.
* An NetTrainer[$$] object contains the following fields:
| 'Executor' | the underlying MXExecutorData[$$] object |
| 'Optimizer' | a function that will be called to perform one optimization update |
| 'Arrays' | mapping from original net paths to parameter NDArrays |
| 'ArrayGradients' | mapping from original net paths to gradient NDArrays |
| 'BatchSize' | the batch size the trainer was created with |
| 'MaxLengths' | the mapping from sequence id to max lengths |
| 'Context' | the device the trainer was instantiated on |"

DefineCustomBoxes[NetTrainer,
	t:NetTrainer[_Association] :> NetTrainerBoxes[t]
];

NetTrainerBoxes[trainer:NetTrainer[assoc_]] := Scope[
	UnpackAssociation[assoc, executor, context, batchSize, bucketed];
	execID = ManagedLibraryExpressionID[executor["MXExecutor"]];
	arrayInfo = If[bucketed, {}, {
		makeItem["Executor inputs", executor["Arrays", "Inputs"]],
		makeItem["Executor outputs", executor["Arrays", "Outputs"]]
	}];
	BoxForm`ArrangeSummaryBox[
		NetTrainer,
		trainer,
		None,
		ToList[
			makeItem["Bucketed", bucketed],
			makeItem["BatchSize", batchSize],
			arrayInfo, 
			If[bucketed, {
				makeItem["BucketCount", Length[executor[[1]]]]
			}, Nothing]
		],
		{
			makeItem["Context", context],
			makeItem["ExecutorID", execID],
			(*makeItem["CurrentLoss", TrainerCurrentLoss[trainer]],*)
			makeItem["MemoryUsage", TrainerMemoryUsage[trainer]]
		},
		StandardForm
	]
];

memStr[x_] := Which[
	x > 1000000, StringForm["`` MB", NumberForm[x/1000000., {2,2}]],
	x > 1000, StringForm["`` KB", NumberForm[x/1000., {2,2}]],
	True, StringForm["`` B", x]
];


PackageExport["LoadParamsIntoNet"]

SetUsage @ "
LoadParamsIntoNet[net$, 'file.params'] loads a list of parameters from the file into the net.
* Existing arrays in net$ will be overwritten.
* The params file should have been produced by NetTrain's params checkpoint format.
* Any frozen arrays are not included in the checkpoint and will be skipped by LoadParamsIntoNet.
* This form of loading is substantially faster and more memory efficient than Import of an entire WLNet."

LoadParamsIntoNet::invcount = "Array count mismatch: file contained `` arrays but net contains `` arrays."

LoadParamsIntoNet[net_NetP, paramsFile_] := CatchFailureAsMessage @ Scope[
	If[!FileExistsQ[paramsFile], ReturnFailed[]];
	arrays = NDArrayImport[paramsFile];
	existing = DeleteCases[None] @ KeySort @ NetArrays[net];
	If[Length[arrays] =!= Length[existing], ThrowFailure["invcount", Length[arrays], Length[existing]]];
	replacements = KeyValueMap[i = 1; loadOne[#1, #2, arrays[[i++]]]&, existing];
	ConstructNet @ ReplacePart[net, replacements]
];

LoadParamsIntoNet::missd1 = "Dimensions of `` don't match: `` vs ``.";

loadOne[path_, Nullable[t_], new_] := loadOne[path, t, new];

loadOne[path_, old:(_NumericArray | _TensorT), new_NumericArray] := Scope[
	If[NumericArrayQ[old], told = TensorT[Dimensions[old]], told = old];
	dnew = Dimensions[new]; tnew = TensorT[dnew];
	If[UnifyTypes[told, tnew] === $Failed,
		If[dnew =!= {1,1,1,1,1}, ThrowFailure["missd1", MsgForm[path], MsgForm[told], MsgForm[tnew]]];
		res = old,
		res = new
	];
	Apply[List, path] -> res
]


PackageScope["NetHasArraysQ"]

NetHasArraysQ[net_NetP] := 
	KeyExistsQ[net, "SharedArrays"] || Catch[netHasArraysQ[net]; False];

DeclareMethod[netHasArraysQ, layerHasArraysQ, Inherited, operatorHasArraysQ];

layerHasArraysQ[net_] := If[net["Arrays"] =!= <||>, Throw[True]];

operatorHasArraysQ[net_] := (
	layerHasArraysQ[net];
	ScanSubNets[netHasArraysQ, net]
);

PackageScope["NetArrays"]

NetArrays[net_NetP, includeAux_:True] := Scope[
	$arrays = Bag[]; 
	$includeAuxArrays = includeAux;
	sowNetArrays @ net;
	If[KeyExistsQ[net, "SharedArrays"],
		ScanFields["SharedArrays", sowArray, net]];
	(* ^ shared arrays are always outermost in the net *)
	KeySort @ Association @ BagContents[$arrays]
];

DeclareMethod[sowNetArrays, sowLayerArrays, Inherited, sowOperatorArrays]

sowLayerArrays[layer_] := Scope[
	$auxArrays = NProperty[layer, "AuxArrays"];
	ScanFields["Arrays", sowNonsharedArray, layer];
];

sowOperatorArrays[op_] := (
	sowLayerArrays[op]; 
	ScanSubNets[sowNetArrays, op];
)

sowArray[arr_] := BagPush[$arrays, $path -> arr];
sowArray[None] := Null;

sowNonsharedArray[e_] := If[$includeAuxArrays || !MemberQ[$auxArrays, Last[$path]], sowArray[e], Null];
sowNonsharedArray[_NetSharedArray] := Null;
