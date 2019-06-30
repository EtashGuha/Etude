Package["NeuralNetworks`"]


PackageScope["ToNetExecutor"]

SetUsage @ "
ToNetExecutor[NetPlan[$$], batchsize$, opts$$] takes a plan and instantiates it on a device, returning an MXExecutorData[$$] object.
The following options are supported: 
| 'Context' | 1 | the device code to use |
| 'DataType' | 0 | floating point datatype to use |
| 'MixedPrecisionQ' | False | whether to use mixed precision |
| 'ArrayCaching' | False | whether to use ToCachedNDArray to cache NumericArrays per-context |
| 'GradientData' | {<||>, <||>, False, 1} | information about how to bind gradients |
| 'SharedExecutor' | None | an existing executor to share memory with |
* 'GradientData' should be a tuple {igrads$, wgrads$, matchWeights$, gradscale$}:
* igrads$ should be an association from arb logical name to mxname of input
* wgrads$ should be an association from arb logical name to mxname of weight, or a list \
of keys from the LogicalWeights field of the plan whose gradients we want.
* matchWeights$ is True if a 'Weights' association should be generated that matches the 'WeightGradients' \ 
association, this is used during training as we sometimes need to be able to read off the current weights \
using their high-level names (see TrainingProgressFunction). 
* gradscale$ is a modifier to the gradient magnitude, this feature isn't used yet but is important \
for gradient compensation that is needed to make low precision training stable.
* 'ArrayCaching' can be Hold[sym$] where sym$ is an association, and NDArrays will be cached
within the association. This is useful for training where buckets in NetTrainer need to share 
NDArrays per-NetPath, but not per-array."

(* TODO: need to add ability to pass an existing MXExecutor as the Gradients, in which case we 
inherit the gradients from that executor *)

Options[ToNetExecutor] = {
	"Context" -> 1,
	"ArrayCaching" -> False,
	"DataType" -> 0,
	"MixedPrecisionQ" -> False,
	"GradientData" -> {<||>, <||>, False, 1},
	"MemoryLimit" -> Automatic
};

ToNetExecutor[NetPlan[assoc_], batchSize_, OptionsPattern[]] := Timed @ Scope[
	
	UnpackOptions[
		context, arrayCaching, dataType, mixedPrecisionQ, gradientData, memoryLimit
	];

	(* allow CUDA/cuDNN to automatically cast higher-precision floats to lower-precision
	in order to use tensorcores *)
	If[context =!= 1, SetCUDAMixedPrecision[mixedPrecisionQ]];

	SetAutomatic[memoryLimit, GetAvailableMemory[context]];
	(* ^ there is a small amount of performance being left on the table here, because
	getting the available memory isn't free and we might already have got it for ChooseXXXBatchSize,
	but its a few dozen microseconds. at least for CPU, GPU is cached  *)

	UnpackAssociation[assoc,
		symbol,
		weightArrays, fixedArrays, auxilliaryArrays, 
		inputDims, zeroArrays, 
		inputs, outputs, inputStates, outputStates, metrics,
		logicalWeights
	];

	Switch[
		gradientData[[2]],
		<||>, 
			Null,
		_Association,
			gradientData[[2]] = Map[logicalWeights, gradientData[[2]]],
			(* ^ this feature exists because an RNN evaluator needs to make a BucketedExecutor that
			has this gradientData stored in it, and though we have the keys for wgrads at that point,
			we don't have the actual values, because it takes a plan to know what they are
			(becuase of the things like NDSubArrays for RNN optimization). so this lets us delay
			the creation of wgrad spec until the plan is turned into an executor *)
		_List,
			gradientData[[2]] = KeyTake[logicalWeights, gradientData[[2]]],
			(* ^ ditto for trainer, but which wants to provide the gradients to preserve as a simple list
			of NetPaths *)
		All,
			gradientData[[2]] = logicalWeights,
			(* ^ this is just for ease-of-debugging *)
		_,
			Panic["InvalidWGrad", "`` is invalid gradient data", gradientData]
	];

	If[$lastPlan === Automatic, $lastPlan ^= NetPlan[assoc]];
	$planAssoc = assoc; $MXLibraryErrorHandler = ExecutorCreateError;
	If[IntegerQ[context], NetExecutorCreate, MultiExecutorCreate][
		context, batchSize, dataType, memoryLimit, assoc["Symbol"], 
		weightArrays, fixedArrays, auxilliaryArrays, 
		inputDims, zeroArrays, 
		arrayCaching, 
		inputs, inputStates, outputs, outputStates, metrics,
		Sequence @@ gradientData,
		logicalWeights
	]
];

General::badnnexec = "Could not set up an execution context for given network. The error was: ``. Please contact Wolfram Research.";

ExecutorCreateError[err_] := Scope[
	plan = NetPlan[$planAssoc];
	Panic["ExecutorCreationError", err,
		"Information" -> <|
			"Plan" -> plan,
			"PlanPlot" -> NetPlanPlot[plan]
		|>
	]
];


PackageScope["$lastPlan"]

(* set this to Automatic if upstream code wants to capture the last plan *)


PackageScope["ToBucketedNetExecutor"]

SetUsage @ "
ToBucketedNetExecutor[net$, {outputs$, devFlags$, batchspec$, tmode$, metrics$, gradients$, arrayCache$}] 
The second args should be as follows:
| outputs$ | which outputs to return |
| devFlags$ | tuple of flags related to datatype and device type |
| batchspec$ | whether batched or not (or a specific integer) |
| tmode$ | whether to apply dropout, etc |
| metrics$ | list of requested metrics |
| gradientData$ | gradient data tuple |
| arrayCache$ | Automatic if a new cache should be created, or a Hold[sym] for an existing cache |
* devFlags$ is a tuple {context$, dtype$, mixedPrecisionQ$}
	* context$ is the device context to evaluate on.
	* dtype$ is the real floating type to use
	* mixedPrecisionQ$ sets whether mixed precision evaluation is allowed
* The batchsize is currently fixed at 16 until we can figure out a better way to pick one.
* metrics$ is a list of NetPaths."

General::netnullseq = "All sequences provided to net must have non-zero length."

PackageScope["$BucketRoundingExponent"]

SetUsage @ "
$BucketRoundingExponent controls how the bucket size is rounded, and should be a number > 1. 
* Larger exponents will produce fewer buckets but more sequence wastage."

$BucketRoundingExponent = 1.5;

PackageScope["RoundBucket"]
SetAttributes[RoundBucket, Listable];
RoundBucket[0] := ThrowFailure["netnullseq"];
RoundBucket[n_] := Which[
	n <= 8, n, 
	n <= 16, Ceiling[n, 2],
	n <= 32, Ceiling[n, 4],
	n <= 64, Ceiling[n, 8],
	True, Ceiling[Power[$BucketRoundingExponent, Ceiling @ Log[$BucketRoundingExponent, n]], 8]
];

(* for debugging *)
ToBucketedNetExecutor[net_] := 
	ToBucketedNetExecutor[net, {All, {{"CPU", 0}, 0, False}, False, False, {}, {<||>, <||>, True, 1}, Automatic}];

ToBucketedNetExecutor[net_NetP, spec_] :=
	Cached[iToBucketedNetExecutor, net, spec];

PackageScope["$DefaultSequenceBatchSize"]
$DefaultSequenceBatchSize = 32;

PackageScope["$CurrentExecutorBucketCount"]
$CurrentExecutorBucketCount = 0;

(* complicated to pick automatically, because it ultimately depends on the size of the maximum
sequence we expect to come along. and we also suffer a major penalty from bigger batch sizes 
because it makes each batch more likely to be ragged when we have a large variance in 
sequence lengths *)

iToBucketedNetExecutor[net_, {outputs_, {context_, dataType_, mixedQ_}, batchq_, tmode_, metrics_, gradientData_, arrayCache_}] := ModuleScope[
	seqIDs = MakeSeqIDLens[net, 0];

	If[arrayCache === Automatic, 
		arrayCacheVar = Association[];
		arrayCache = Hold[arrayCacheVar];
	];

	copiedNet = Association[Normal[net]];
	(* ^ seemingly pointless, but forces a copy to be made, which breaks a ref-count cycle
	     between the net and the bucketedexecutor, which is cached against the net 
	*)
	{ignore, checker, toIndep, fromIndep} = Cached[NetConstraintData, net];

	batchSize = Which[IntegerQ[batchq], batchq, TrueQ[batchq], $DefaultSequenceBatchSize, True, 1];
	reshapeTemplate = Indeterminate;
	bucketer = (toIndep /* RoundBucket /* fromIndep /* Map[Max]);

	info = Association[
		"Net" -> copiedNet, 
		"BatchSize" -> batchSize,
		"Outputs" -> outputs,
		"Context" -> context,
		"DataType" -> dataType,
		"MixedPrecisionQ" -> mixedQ,
		"GradientData" -> gradientData,
		"TMode" -> tmode,
		"SequenceIDs" -> Keys[seqIDs],
		"ArrayCache" -> arrayCache,
		"ReshapeTemplate" -> Hold[reshapeTemplate],
		"Bucketer" -> bucketer,
		"ConstraintChecker" -> checker,
		"RequestedMetrics" -> metrics
	];
	buckets = Association[];
	max = None;

	BucketedNetExecutor[buckets, max, Evaluate @ info]
];

SetHoldFirst[createBucket];
createBucket[{buckets_, max_}, info_, bucket_] := Timed @ Scope[
	UnpackAssociation[
		info, net, outputs, batchSize, context, dataType, sequenceIDs, tMode, 
		gradientData, reshapeTemplate, requestedMetrics, mixedPrecisionQ
	];
	master = buckets[max];
	templateValue = First[reshapeTemplate];
	If[Head[templateValue] === Function, 
		sizes = templateValue[batchSize, bucket];
		$BucketingLogger[{"NewReshapedBucket", bucket, sizes}];
		Return @ NetExecutorReshape[master, sizes]
	];
	$BucketingLogger[{If[MissingQ[master], "NewMasterBucket", "NewBucket"], bucket}];
	plan = ToNetPlan[net, {
		outputs, AssociationThread[sequenceIDs, bucket], tMode,
		{Min[context] =!= 1, dataType, mixedPrecisionQ}, 
		requestedMetrics
	}];
	If[templateValue === Indeterminate, HoldSet[reshapeTemplate, plan["ReshapeTemplate"]]];
	If[MissingQ[master],
		max ^= bucket;
		ToNetExecutor[
			plan, batchSize,
			"Context" -> context, "DataType" -> dataType,
			"GradientData" -> gradientData,
			"ArrayCaching" -> info["ArrayCache"]
			(* this cache is no longer needed, but it doesn't hurt *)
		]
	,
		If[dominatesQ[bucket, max], 
			$BucketingLogger[{"ClearBuckets", bucket, max}];
			$CurrentExecutorBucketCount -= Length[buckets];
			max ^= bucket;
			buckets ^= <||>;
		];
		NetExecutorInherit[
			plan["Symbol"], master,
			plan["InputDims"]
		]
	]
];

dominatesQ[b1_, b2_] := And @@ Thread[b1 >= b2];

PackageExport["BucketedNetExecutor"]

PackageScope["Bucket"]

SetHoldAll[BucketedNetExecutor]

BucketedNetExecutor[_, max_, info_][key_] := info[key];

NetExecutorMemoryInformation[BucketedNetExecutor[buckets_, max_, info_]] :=
	NetExecutorMemoryInformation[First[buckets]];


PackageScope["ClearBuckets"]

ClearBuckets[BucketedNetExecutor[buckets_, max_, info_]] := (
	$CurrentExecutorBucketCount -= Length[buckets];
	buckets = <||>
);


PackageScope["GetMasterExecutor"]

GetMasterExecutor[BucketedNetExecutor[buckets_, max_, info_]] :=
	Lookup[buckets, Key[max], None];


PackageScope["GetBucketExecutor"]

GetBucketExecutor[BucketedNetExecutor[buckets_, max_, info_], inbucket_] := Block[
	{bucket = info["Bucketer"][inbucket]},
	info["ConstraintChecker"] @ inbucket;
	Lookup[
		buckets, Key @ bucket, 
		$CurrentExecutorBucketCount++; 
		buckets[bucket] = createBucket[{buckets, max}, info, bucket]
	]
]

DefineCustomBoxes[BucketedNetExecutor,
	t:BucketedNetExecutor[_, _, _Association] :> BucketedNetExecutorBoxes[t]
];

BucketedNetExecutorBoxes[exec:BucketedNetExecutor[buckets_, max_, info_Association]] := Scope[
	BoxForm`ArrangeSummaryBox[
		BucketedNetExecutor,
		exec,
		None,
		{makeItem["TMode", info["TMode"]],
		 makeItem["Max bucket", max], 
		 makeItem["Bucket count", Dynamic @ Length[buckets]],
		 makeItem["Bucketer", info["Bucketer"]]},
		{makeItem["Buckets", Dynamic @ Sort[Keys[buckets]]]},
		StandardForm
	]
];



PackageScope["$ExecutorLogger"]
PackageScope["$BucketingLogger"]

$ExecutorLogger = Hold;
$BucketingLogger = Hold;



PackageScope["ExecutorMeasure"]

SetUsage @ "
ExecutorMeasure[executor$, generator$, dataLength$, batchSize$, outputNames$, measureInfo$] returns a pair containing losses and measurements \
of the executor on the data set produced by generator$.
* outputNames$ is the list of names of the losses.
* measureInfo$ is the list of associations that describe the measurements to take.
* {losses$, measurements$} is returned, where both are associations."

(* technically we could get the outputNames from the executor itself, but it works slightly differently for the various kinds of
executor. *)

(* note that batchSize is usually gettable from the executor itself, but in the case that NetMeasurements
has cached the required results beforehand, the executor might not exist (why bother creating it?), and so 
the batchsize is NOT gettable via it, and hence we muts pass it in another argument *)

ExecutorMeasure[executor_, generator_, length_, batchSize_, outputNames_, measurementsInfo_, progFunc_:Hold, cacheVar_:None] := Scope[

	measurementPaths = measurementsInfo[[All, "Path"]];

	{batches, excess} = BatchCountExcess[length, batchSize];

	If[cacheVar =!= None && SubsetQ[Keys @ (metricTotals = ReleaseHold @ cacheVar), measurementPaths], 
		(* loss outputs aren't actually needed by NetMeasuremens, which is using the caching feaure *)
		losses = None; Goto[SkipComputation]];

	bucketed = Head[executor] === BucketedNetExecutor;

	lookupOutputs = Lookup[outputNames];
	outTotals = metricTotals = 0;
	If[!bucketed, 
		inputArrays = executor["Arrays", "Inputs"];
		outArrays = lookupOutputs @ executor["Arrays", "Outputs"];
		metricArrays = executor["Arrays", "Metrics"];
	];
	
	Do[
		progFunc[b/batches];
		PreemptProtect[
			batch = generator[b];
			If[bucketed, 
				subexec = GetBucketExecutor[executor, Last[batch]];
				NDArraySet[subexec["Arrays", "Inputs"], First[batch]];
				NetExecutorForward[subexec, False];
				outArrays = lookupOutputs @ subexec["Arrays", "Outputs"];
				metricArrays = subexec["Arrays", "Metrics"];
			,
				NDArraySet[inputArrays, batch];
				NetExecutorForward[executor, False];
			];
			
			If[b === 1 && excess > 0,
				(* initially, the excess batch elements (if any) are *always* at the beginning,
				so in that case we just chop off some elements of the metric arrays *)
				outTotals = Map[NDArrayGetPartialTotalNormal[#, excess, 1]&, outArrays];
				metricTotals = Map[NDArrayGetPartialTotalNormal[#, excess, 1]&, metricArrays];
			,
				outTotals += Map[NDArrayGetTotalNormal, outArrays];
				metricTotals += Map[NDArrayGetTotalNormal, metricArrays];
			];
		];
	,
		{b, batches}
	];	
	
	(* join on the freshly computed totals with the previous cached totals *)
	If[cacheVar =!= None, 
		JoinTo[metricTotals, ReleaseHold @ cacheVar];
		HoldSet[cacheVar, metricTotals];
	];

	Label[SkipComputation];

	losses = AssociationThread[outputNames, outTotals / length];
	losses = If[Length @ losses > 1, Prepend[losses, "TotalLoss" -> Total @ losses], losses];
	
	metricValues = RoundMetricFinalize[measurementsInfo, metricTotals, losses, length, batches, True];

	{losses, metricValues}
];


