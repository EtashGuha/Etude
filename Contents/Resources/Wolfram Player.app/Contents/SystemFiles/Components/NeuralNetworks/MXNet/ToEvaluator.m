Package["NeuralNetworks`"]


PackageScope["FindEvaluators"]

FindEvaluators[net_] :=
	Cases[
		Normal @ CacheContents["SingletonCache", net],
		({iToNetEvaluator, info_} -> eval_) :> {info, eval}
	];


PackageScope["FindExecutors"]

$execP = _NetExecutor | _BucketedNetExecutor | _MultiExecutor;

FindExecutors[net_] := 
	DeepUniqueCases[
		FindEvaluators[net],
		exec:$execP | (exec_Symbol ? HoldSymbolQ /; And[
			StringStartsQ[SymbolName @ Unevaluated @ exec, "exec$" | "bucketedExecutor$"], 
			MatchQ[exec, $execP]
		])
	];


PackageExport["NetPortGradient"]
(* ^ actually a system symbol *)

PackageScope["ToNetEvaluator"]

SetUsage @ "
ToNetEvaluator[net$, {batchspec$, tmode$, devFlags$, pspec$}] \
creates and caches a function that takes as its arguments the inputs \
to net$, and returns the outputs as specified by the port spec pspec$.
ToNetEvaluator[net$] is for quick debugging and uses the CPU \
context, no batching, and the default port spec.
* If any of the inputs of net$ are a a variable-length seq, \
ToNetEvaluator will delegate to ToBucketedNetEvaluator.
* tmode$ specifies whether to apply Dropout, etc.
* devFlags$ is a tuple {context$, dtype$, mixedPrecisionQ$}
	* context$ is the device context to evaluate on.
	* dtype$ is the real floating type to use
	* mixedPrecisionQ$ sets whether mixed precision evaluation is allowed
* The batchspec$ can be True, False, or an integer. 
* If batchspec$ is True, the batchsize will be chosen automatically.
* The port spec pspec$ is any of the specs supported by top-level
nets."

(* for debugging, deliberately doesn't cache *)
ToNetEvaluator[net_] := 
	iToNetEvaluator[net, {False, False, {1, 0, False}, Automatic}];

ToNetEvaluator[net_, meta_] := 
	RecentNetCached[iToNetEvaluator, net, meta];

iToNetEvaluator[net_, info:{batchspec_, tmode_, {context_, dtype_, mixedQ_}, pspec_}] := Scope @ PreemptProtect[
	If[!FullySpecifiedNetQ[net], Panic["InvalidNet"]];
	inputs = Inputs[net];
	outputs = Outputs[net];

	NetFinalCheck[NData @ net];
	
	If[ContainsVarSequenceQ[inputs], 
		Return @ ToBucketedNetEvaluator[net, info];
	];

	{outputDescriptors, outs, igrads, wgrads, doAssoc} = 
		ParseNetProperty[net, pspec, batchspec =!= False];
	hasGrads = igrads =!= <||> || wgrads =!= <||>;
	tmode2 = If[tmode, True, If[hasGrads, "InferenceGradients", False]];
	tmode = tmode || hasGrads;

	plan = ToNetPlan[net, {outs, <||>, tmode2, {context =!= 1, dtype, mixedQ}, {}}];

	isBatched = True;
	batchSize = Switch[batchspec,
		True, 
			If[hasGrads, 
				ChooseTrainingBatchSize[plan, wgrads, {context, dtype, mixedQ}],
				ChooseEvaluationBatchSize[plan, {context, dtype, mixedQ}]]
		,
		False, 
			isBatched = False; 
			1
		,
		_Integer, 
			batchspec
	];
	executor = ToNetExecutor[
		plan, batchSize, 
		"GradientData" -> {igrads, wgrads, False, 1},
		"Context" -> context, "DataType" -> dtype, "MixedPrecisionQ" -> mixedQ, "ArrayCaching" -> True
	];
	(* to amortize the cost of all the logic associated with evaluating a net, we build a pure
	function that does exactly what needs to be done, and we cache that pure function against
	the network. *)

	{dummy, dummy, setterCode} = BodyFunctionReplaceAll[
		makeInputProcessingCode[makeDefaultSetter, inputs, isBatched],
		$InputData -> Slot
	];

	If[isBatched,
		isGrads = #[[1,0]] === NetPortGradient& /@ outputDescriptors;
		setterFunction = CreateFunction[setterCode];
		List /* BatchEvaluator[executor, batchSize, outputDescriptors, isGrads, setterFunction, tmode, doAssoc]
	,
		$ExecutorArrays = executor["Arrays"];
		inputStateArrays = $ExecutorArrays["InputStates"];
		outGradArrays = $ExecutorArrays["OutputGradients"];
		code = Module[{exec = executor}, {
			setterCode,
			If[inputStateArrays === <||>, 
				Hold @ If[FastValueQ[$RSData2], ThrowFailure["norsnet"]], 
				Hold @ SetupRecurrentStateArrays[Eval @ inputStateArrays, False]
			],
			If[hasGrads, makeGradSetter[outGradArrays]],
			Hold @ NetExecutorForward[exec, Eval @ tmode],
			If[hasGrads, Hold @ NetExecutorBackward[exec]],
			assembleOutputs[outputDescriptors, doAssoc]
		}];
		code = BodyFunctionReplaceAll[code, HoldPattern[e_$ExecutorArrays] :> RuleCondition[e]];
		(* ^ substitue in the executor arrays *)
		CreateUnpreemptableFunction @ code
	]
];

General::nobatchsupp = "`` is not currently supported when providing batches of inputs."
General::enetragged = "When evaluating batches of inputs, all input ports must be given lists of the same length."

doBatchTests[inputs_] := UseMacros[
	If[Length[inputs] > 1 && Not[SameQ @@ Map[Length, inputs]], ThrowFailure["enetragged"]];
	If[FastValueQ[$RSData2], ThrowFailure["nobatchsupp", "Setting of NetPort[All,\"States\"]"]];
	If[FastValueQ[$NPGData], ThrowFailure["nobatchsupp", "Setting of NetPortGradients"]]
];

assembleOutputs[{OutputDescriptor[name_, code_]}, False] :=
	Hold[code] // doArrayReplacements;

assembleOutputs[list:{__OutputDescriptor}, _] :=
	Hold[AssociationThread][
		list[[All, 1]], 
		Extract[list, {All, 2}, Hold]
	] // doArrayReplacements;

doArrayReplacements[e_] := BodyFunctionReplaceAll[e, HoldPattern[lookup_$ExecutorArrays] :> RuleCondition[lookup]];


(* no-one actually uses this in batched mode. TODO: add support for batched recurrent eval *)
SetupRecurrentStateArrays[arrays_, _] /; !FastValueQ[$RSData] && !FastValueQ[$RSData2] := 
	NDArraySetConstant[arrays, 0.];

General::norsnet = "Net does not contain any recurrent states."
General::invrsin = "Value of NetPort[All,\"States\"] -> ... should be an association mapping state specifications to initial values."
General::invrsinkey = "`` does not identify a recurrent state within the net."

(* we have to have two globals $RSData and $RSData2 because NetApply wants to Block one of them
to set it from the association key, and NetStateObject can't penetrate that Block. this is the
simplest way to handle that, though not elegant. *)

SetupRecurrentStateArrays[arrays_, batched_] := Scope[
	setter = If[batched, NDArraySetBatched, NDArraySetUnbatched];
	If[MatchQ[$RSData, _RecurrentStateContainer],
		MapThread[setter, {Values @ arrays, First[$RSData]}];
		Return[];
	];
	rsdata = If[FastValueQ[$RSData2], $RSData2, $RSData];
	NDArraySetConstant[arrays, 0.];
	If[rsdata === None, Return[]];
	If[!AssociationQ[rsdata], ThrowFailure["invrsin"]];
	KeyValueScan[
		setter[
			arr = Lookup[arrays, Key @ ToList[#1], ThrowFailure["invrsinkey", #]],
			#2,
			InputErrorHandler[#, TensorT @ Rest @ NDArrayDimensions[arr]]
		]&,
		rsdata
	]
];

(* TODO: Unify these all into InputDescriptors *)

makeGradSetter[outGradArrays_] := ModuleScope[
	ograds = outGradArrays;
	Hold @ If[FastValueQ[$NPGData], 
		NDArraySetConstant[ograds, 0.];
		KeyValueMap[setOutGrad[LookupOr[ograds, #1, noOutGrad], #2, #1]&, $NPGData]
	,
		NDArraySetConstant[ograds, 1.];
	]
];

General::nooutgrad = "`` is not a valid output gradient of the net."
noOutGrad[name_] := ThrowFailure["nooutgrad", name];

setOutGrad[arr_, None, _] := NDArraySetConstant[arr, 0.];
setOutGrad[arr_, val_ ? NumericQ, _] := NDArraySetConstant[arr, N[val]];
setOutGrad[arr_, data_, name_] := NDArraySetUnbatched[arr, data, InputErrorHandler[name, TensorT[Rest @ NDArrayDimensions[arr]]]];

makeSetter[i_, name_ -> arr_, _ -> type_] := With[
	{encoder = ToEncoderFunction[type, False]},
	Hold @ NDArraySetUnbatched[arr, Slot[i], InputErrorHandler[name, type]]
];


PackageScope["BatchEvaluator"]

(* fast path for when we don't need to evaluate multiple batches: we can skip bags, looping, etc *)
BatchEvaluator[executor_, batchSize_, outputDescriptors_, isGrads_, setterFunction_, tmode_, doAssoc_][inputs_] /; Length[First[inputs]] <= batchSize := Scope[
	doBatchTests[inputs];
	length = Length @ First @ inputs;
	If[length < batchSize, executor = Cached[NetExecutorReshape, executor, length]];
	$ExecutorArrays = executor["Arrays"];
	NDArraySetConstant[$ExecutorArrays["InputStates"], 0.];
	Apply[setterFunction, inputs];
	NetExecutorForward[executor, tmode];
	If[Or @@ isGrads, NetExecutorBackward[executor]];
	toResult[outputDescriptors, doAssoc]
];

toResult[outputDescriptors_, doAssoc_] := 
	If[Length[outputDescriptors] === 1 && !doAssoc, 
		outputDescriptors[[1, 2]],
		Association[Rule @@@ outputDescriptors]
	]

BatchEvaluator[executor_, batchSize_, outputDescriptors_, isGrads_, setterFunction_, tmode_, doAssoc_][inputs_] := Scope[
	doBatchTests[inputs];
	length = Length @ First @ inputs;
	outputBags = Table[Internal`Bag[], Length @ outputDescriptors];
	$ExecutorArrays = executor["Arrays"];
	inputArrays = Values @ $ExecutorArrays["Inputs"];
	NDArraySetConstant[$ExecutorArrays["InputStates"], 0.];
	doGrads = Or @@ isGrads;
	n = Ceiling[length / batchSize];
	Do[
		$BatchProgress ^= {i, n} * batchSize; $BatchHook;
		maxIndex = Min[length, batchSize * i];
		minIndex = batchSize * (i - 1) + 1;
		If[(maxIndex - minIndex + 1) < batchSize,
			executor = Cached[NetExecutorReshape, executor, maxIndex - minIndex + 1];
			$ExecutorArrays = executor["Arrays"];
			inputArrays = Values @ $ExecutorArrays["Inputs"];
		];	
		Apply[setterFunction, Take[inputs, All, {minIndex, maxIndex}]];
		NetExecutorForward[executor, tmode];
		If[doGrads, NetExecutorBackward[executor]];
		ScanThread[BagPush[#1, Last[#2], If[#3, 0, 1]]&, {outputBags, outputDescriptors, isGrads}];
	,
		{i, n}
	];
	bagsToResult[outputDescriptors, outputBags, isGrads, All, doAssoc]
]

arrayTotal[na:{__NumericArray}] := 
	Block[{val = 0}, Do[val += Normal[a], {a, na}]; toNumericArray[val]];

arrayTotal[other_] := Total[other];

bagsToResult[outputDescriptors_, outputBags_, isGrads_, part_, doAssoc_] :=
	If[Length[outputBags] === 1 && !doAssoc,
		If[isGrads[[1]], arrayTotal, Identity] @ BagPart[outputBags[[1]], part],
		Association @ MapThread[
			Rule[
				First[#1],
				If[#3, arrayTotal, Identity] @ BagPart[#2, part]
			]&, 
			{outputDescriptors, outputBags, isGrads}
		]
	]

(* the support I tried to for $NPGData and $RSData looked like this:

outside the loop:
	$oldNPG = $NPGData; $oldRS = $RSData; 
	$doNPG = ValueQ[$NPGData]; $doRS = ValueQ[$RSData];
	$NPGData = $RSData = Null;
	take = Function[Take[#, {minIndex, maxIndex}]];

inside the loop:
		If[$doNPG, 
			$NPGData ^= take /@ $oldNPG;
		];
		If[$doRS, 
			$RSData ^= take /@ $oldRS; 
			SetupRecurrentStateArrays[$ExecutorArrays["InputStates"]]
		];
*)


PackageScope["ToBucketedNetEvaluator"]

SetUsage @ "
ToBucketedNetEvaluator[net$, {isBatched$, devFlags$, pspec$}] \
creates and caches a function that takes as its arguments the inputs \
to net$, and returns the outputs as specified by the port spec pspec$.
ToBucketedNetEvaluator[net$] is for quick debugging and uses \
the CPU context, no batching, and the default port spec.
* At least one of the inputs of net$ should be a variable-length seq.
* tmode$ determines whether dropout, etc is applied.
* A BucketedNetExecutor is used internally, whereaby the actual \
sequence lengths of the input sequences are used to lookup the \
concrete MXExecutor to use.
* devFlags$ is a tuple {context$, dtype$, mixedPrecisionQ$}
	* context$ is the device context to evaluate on.
	* dtype$ is the real floating type to use
	* mixedPrecisionQ$ sets whether mixed precision evaluation is allowed
* Currently, batched-mode uses a hard-coded batch size of 16.
* The port spec pspec$ is any of the specs supported by top-level
nets."

(* for debugging *)
ToBucketedNetEvaluator[net_] := 
	ToBucketedNetEvaluator[net, {False, False, {{"CPU", 0}, 0, False}, Automatic}];

ToBucketedNetEvaluator[net_, {batchspec_, tmode_, {context_, dataType_, mixedQ_}, pspec_}] := ModuleScope @ Block[
	{ospec, outputs},
	If[!FullySpecifiedNetQ[net], Panic["InvalidNet"]];

	inputs = Inputs[net];
	outputs = Outputs[net];
	hasStates = GetInteriorStates[net] =!= <||>;

	{outputDescriptors, outs, igrads, wgrads, doAssoc} = ParseNetProperty[net, pspec, batchspec =!= False];

	hasGrads = igrads =!= <||> || wgrads =!= <||>;
	tmode2 = If[tmode, True, If[hasGrads, "InferenceGradients", False]];
	tmode = tmode || hasGrads;

	bucketedExecutor = ToBucketedNetExecutor[net, {
		outs, {context, dataType, mixedQ}, batchspec, 
		tmode2, {}, {igrads, wgrads, False, 1}, Automatic
	}];

	(* this generates code to obtain the length of sequence inputs so we can pick a bucket *)
	{lenCode, lens, setterCode} = BodyFunctionReplaceAll[
		makeInputProcessingCode[makeDefaultSetter, inputs, batchspec =!= False],
		$InputData -> Slot
	];

	If[batchspec =!= False,
		batchSize = bucketedExecutor["BatchSize"];
		genFactory = ConstructBucketedGeneratorFactory[inputs, batchSize];
		bucketFactory = CreateFunction @ Hold[List][Append[lenCode, lens], Hold[Function][setterCode]];
		isGrads = #[[1,0]] === NetPortGradient& /@ outputDescriptors;
		List /* BatchedBucketedNetEvaluator[
			genFactory, bucketFactory,
			bucketedExecutor, batchSize,
			outputDescriptors, isGrads,
			tmode, doAssoc
		]
	,
		ComposeTo[{lenCode, lens, setterCode}, ReplaceAll[$TempData -> TempVar]];
		CreateUnpreemptableFunction	@ List[
			lenCode,
			Hold[TempVar[executor] ^= GetBucketExecutor[bucketedExecutor, Eval @ lens]],
			Hold[TempVar[$ExecutorArrays] ^= executor["Arrays"]],
			setterCode, (* <- set $InputArrays *)
			If[!hasStates,
				Hold @ If[FastValueQ[$RSData], ThrowFailure["norsnet"]], 
				Hold @ SetupRecurrentStateArrays[$ExecutorArrays["InputStates"], False]
			],
			If[hasGrads, makeGradSetter[$ExecutorArrays["OutputGradients"]]],
			Hold @ NetExecutorForward[executor, Eval @ tmode],
			If[hasGrads, Hold @ NetExecutorBackward[executor]],
			assembleOutputs[outputDescriptors, doAssoc]
		]
	]
];


PackageExport["$BatchProgress"]
PackageExport["$BatchHook"]

SetUsage @ "
$BatchProgress is temporarily blocked during batch evaluations, and can be monitored to track large batch evaluations. 
* It's value is {curr$, max$}, where curr$ is the current element being processed, and max$ is the total number of elements.
* See also $BatchHook."

SetUsage @ "
$BatchHook is a symbol that can be SetDelayed to an expression to evaluate at the start of each batch during batch evaluation.
* See also $BatchProgress."

PackageScope["BatchedBucketedNetEvaluator"]

(* fast path for when we don't need to evaluate multiple batches: we can skip permutations, bags, looping, etc *)
BatchedBucketedNetEvaluator[genFactory_, bucketFactory_, bucketedExecutor_, batchSize_, outputDescriptors_, isGrads_, tmode_, doAssoc_][inputs_] /; Length[First[inputs]] <= batchSize := 
Scope @ Block[{$TempData = <||>, $LastGeneratorName = None, $LastGeneratorIndices = Automatic},
	doBatchTests[inputs];
	length = Length @ First @ inputs;
	{bucket, setter} = Apply[bucketFactory, inputs]; (* <- this will set keys on $TempData *)
	executor = GetBucketExecutor[bucketedExecutor, bucket];
	If[length < batchSize, executor = Cached[NetExecutorReshape, executor, length]];
	$ExecutorArrays = executor["Arrays"];
	NDArraySetConstant[$ExecutorArrays["InputStates"], 0.];
	Apply[setter, inputs]; $TempData = Null;
	NetExecutorForward[executor, tmode];
	If[Or @@ isGrads, NetExecutorBackward[executor]];
	toResult[outputDescriptors, doAssoc]
]

BatchedBucketedNetEvaluator[genFactory_, bucketFactory_, bucketedExecutor_, batchSize_, outputDescriptors_, isGrads_, tmode_, doAssoc_][inputs_] := Scope[
	$LastGeneratorName = None; $LastGeneratorIndices = Automatic;
	doBatchTests[inputs];	
	{generator, max, iperm, n} = Apply[genFactory, inputs];
	outputBags = Table[Bag[], Length[outputDescriptors]];
	lastBucket = None;
	doGrads = Or @@ isGrads;
	GetBucketExecutor[bucketedExecutor, max]; 
	Do[
		$BatchProgress ^= {i-1, n} * batchSize; $BatchHook;
		batch = generator[i];
		thisBucket = Last[batch];
		If[lastBucket =!= thisBucket,
			executor = GetBucketExecutor[bucketedExecutor, thisBucket];
			$ExecutorArrays = executor["Arrays"];
			NDArraySetConstant[$ExecutorArrays["InputStates"], 0.];
			lastBucket = thisBucket;
		];
		NDArraySet[$ExecutorArrays["Inputs"], First @ batch];
		NetExecutorForward[executor, tmode];
		If[doGrads, NetExecutorBackward[executor]];
		ScanThread[BagPush[#1, Last[#2], 1]&, {outputBags, outputDescriptors}];
	,
		{i, n}
	];
	bagsToResult[outputDescriptors, outputBags, isGrads, iperm, doAssoc]
];
