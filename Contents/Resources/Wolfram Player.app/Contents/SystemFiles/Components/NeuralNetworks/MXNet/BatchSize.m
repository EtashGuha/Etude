Package["NeuralNetworks`"]

PackageExport["$DisableMemoryLimiting"]
$DisableMemoryLimiting = False;

PackageExport["$MaxExecutorMemoryFraction"]

PackageScope["GetTargetExecutorSize"]
PackageScope["GetAvailableMemory"]

(* this doesn't apply to bucketed evaluators, which have a fixed batchsize
(one for training and one for evaluation *)

$MaxExecutorMemoryFraction = <|
	"CPUTraining" -> 0.8, 
	"GPUTraining" -> 0.9,
	"CPUEvaluation" -> 0.5,
	"GPUEvaluation" -> 0.9
|>;

GetAvailableMemory[1] := Min[$RealisticSystemMemory, MemoryAvailable[]];
GetAvailableMemory[context_Integer] := If[$DisableMemoryLimiting, 10*^12, Last[GetGPUMemoryInformation[BitShiftRight[context, 2] + 1], $Failed]];
GetAvailableMemory[list_List] := Min[GetAvailableMemory /@ list];

GetTargetExecutorSize[istrain_, context_] := Scope[
	type = If[context === 1, "CPU", "GPU"] <> If[istrain, "Training", "Evaluation"];
	fraction = $MaxExecutorMemoryFraction[type];
	total = GetAvailableMemory[context];
	{Max[fraction * total, 100*^6], total}
];

PackageScope["ChooseTrainingBatchSize"]
PackageExport["$MaxTrainingBatchSize"]

$MXNetTrainingWarmedUp = False;
$MaxTrainingBatchSize = 64;

ChooseTrainingBatchSize[plan_, weightPaths_, {context_, dataType_, mixedQ_}] := Scope[
	gradientData = {<||>, weightPaths, False, 1};
	{target, maxmem} = GetTargetExecutorSize[True, context];
	executor = ToNetExecutor[
		plan, 1, 
		"Context" -> context, 
		"GradientData" -> gradientData, 
		"ArrayCaching" -> "SharedDummyArrays",
		"DataType" -> dataType,
		"MixedPrecisionQ" -> mixedQ,
		"MemoryLimit" -> maxmem
	];
	If[Count[executor["MXArrayData"][[3]], _NDArray] === 0, Return[1]]; (* <- dummy value, will be dealt with later *)
	time = timeExecutor[executor];
	If[!$MXNetTrainingWarmedUp || time < 0.05, 
		$MXNetTrainingWarmedUp = True;
		time = Min[time, timeExecutor[executor]]];
	meminfo = NetExecutorMemoryInformation[executor];
	fixedcost = meminfo["NonBatchedArrays"];
	batchcost = meminfo["BatchedArrays"] + meminfo["Internal"];
	reqmem = fixedcost + batchcost;
	If[reqmem > maxmem, OOMPanic[1, reqmem, maxmem, context]];
	(* we have a linear model for memory requirements as a function of batch size,
	and solve for the target memory size *)
	n1 = Ceiling[(target - fixedcost) / (batchcost + 1)];
	(* also prevent a batch from taking more than 2 seconds, because responsiveness
	starts to really suffer at that point *)
	n2 = Ceiling[2.0 / time, 4];
	(* take the minimum of the two *)
	n = Min[n1, n2];
	Clip[n, {1, $MaxTrainingBatchSize}] 
];

timeExecutor[exec_] := 
	First @ AbsoluteTiming[
		NetExecutorForward[exec, True];
		NetExecutorBackward[exec];
		NDArrayWaitForAll[];
	];

PackageScope["ChooseEvaluationBatchSize"]
PackageExport["$MaxEvaluationBatchSize"]
PackageExport["$MaxEvaluationBatchMemory"]

$MaxEvaluationBatchSize = 512;
$MaxEvaluationBatchMemory = 200*^6;

ChooseEvaluationBatchSize[plan_, {context_, dataType_, mixedQ_}] := 
	Cached[iChooseEvaluationBatchSize, plan, {context, dataType, mixedQ}];
	
iChooseEvaluationBatchSize[plan_, {context_, dataType_, mixedQ_}] := Scope[
	{target, maxmem} = GetTargetExecutorSize[False, context];
	executor = ToNetExecutor[
		plan, 1, 
		"Context" -> context, 
		"DataType" -> dataType,
		"MixedPrecisionQ" -> mixedQ,
		"ArrayCaching" -> "SharedDummyArrays",
		"MemoryLimit" -> maxmem
	];
	time = First @ AbsoluteTiming[
		NetExecutorForward[executor, False];
		NDArrayWaitForAll[];
	];
	meminfo = NetExecutorMemoryInformation[executor];
	fixedcost = meminfo["NonBatchedArrays"];
	batchcost = meminfo["BatchedArrays"] + meminfo["Internal"];
	reqmem = fixedcost + batchcost;
	If[reqmem > maxmem, OOMPanic[1, reqmem, maxmem, context]];
	(* we have a linear model for memory requirements as a function of batch size,
	and solve for the target memory size. this prevents crashing*)
	n1 = Ceiling[(target - fixedcost) / (batchcost + 1)];
	(* we also have another model that just wants to allocate 200 megabytes maximum
	to batch-variable purposes. the theory being that nets that are too much bigger
	than that are unlikely to benefit from large batch sizes *)
	n2 = Ceiling[$MaxEvaluationBatchMemory / batchcost];
	(* also prevent a batch from taking more than 2 seconds, where this is
	a proxy for how much parallelism the net is already probably taking advantage of,
	e.g. big slow convnets don't benefit from high batch sizes *)
	n3 = Ceiling[2.0 / time];
	(* take the minimum of the three *)
	n = Min[n1, n2, n3];
	Clip[n, {1, $MaxEvaluationBatchSize}] 
];