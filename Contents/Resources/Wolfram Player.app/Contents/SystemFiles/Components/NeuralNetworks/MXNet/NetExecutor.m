Package["NeuralNetworks`"]


DefineAlias[MLEID, ManagedLibraryExpressionID];

DefineAlias[mxCall, MXNetLink`PackageScope`mxlCall];
DefineAlias[mxExecutorBind, MXNetLink`PackageScope`mxlMXExecutorBind];
DefineAlias[mxExecutorOutputs, MXNetLink`PackageScope`mxlMXExecutorOutputs];

PackageExport["NetExecutorCreate"]

SetUsage @ "
NetExecutorCreate[context$, batchSize$, dataType$, memoryLimit$, symbol$, weightArrays$, fixedArrays$, auxArrays$, inputDims$, zeroArrays$, arrayCaching$, inputs$, inputStates$, outputs$, outputStates$, metrics$, inputGradients$, weightGradients$, matchWeights$, gradientScale$] constructs a NetExecutor[$$] object.
* context$ is a NDArray context code
* batchSize$ is an integer that will be used to BatchSize in inputDims$
* dataType$ is an integer that specifies the numeric type of arrays
* memoryLimit$ specifies a limit in bytes for all the arrays used in the net. Exceeding this will produce a General::netmem failure, avoiding a possible crash.
* symbol$ is an MXSymbol
* weightArrays$, fixedArrays$, auxArrays$ are associations from mxname to NumericArray
* inputDims$ is an association from input mxname to dimensions
* zeroArrays$ is a list of mxnames that should be pre-initialized to zero
* gradArrayKeys$ is All, None, or a list of mxnames; these will get gradients attached to them
* arrayCaching$ is False, True, or Hold[sym]; this determines how NumericArrays are converted to NDArrays
* inputs$, inputStates$, outputs$, outputStates$, metrics$, inputGradients$, weightGradients$ are mappings from top-level specs (e.g. strings/netports) to mx specs
* mx specs are either 'mxname', or NDSequenceArray['mxname', 'lenname'] 
* matchWeights$ is True if a 'Weights' association should be generated that matches the weightGradients$ association.
* gradientScale$ is a number, the constant value used in the outgrads.
* logicalWeights$ is an assoc, similar to weightArrays$, but contains higher-level arrays (e.g. NDSubArray specs)"

General::netmem1 = "Insufficient memory to evaluate the network: at least ``bytes are required but only ``bytes are available for TargetDevice -> ``. Try specify a BatchSize smaller than the current value of ``."
General::netmem2 = "Insufficient memory to evaluate the network: at least ``bytes are required but only ``bytes are available for TargetDevice -> ``."

PackageScope["OOMPanic"]
OOMPanic[n_, required_, max_, context_] := ThrowFailure[
	Evaluate @ If[n === 1, "netmem2", "netmem1"], 
	PositiveSIString @ required, PositiveSIString @ max, UnparseContext @ context, 
	If[n === 1, Sequence @@ {}, n]
];

dimsToBytes[<||>] := 0;
dimsToBytes[assoc_] := 4 * Total[Times @@@ Values[assoc]];
(* ^ assume float32 TODO: fix for half/double precision *) 

NetExecutorCreate[context_Integer, batchSize_Integer, dataType_Integer, memoryLimit_Integer, mxSymbol_MXSymbol, weightArrays_Association, fixedArrays_Association, auxArrays_Association, inputDims_Association, zeroArrays_, arrayCaching_, inputs_Association, inputStates_Association, outputs_Association, outputStates_Association, metrics_Association, inputGradients_Association, weightGradients_Association, matchWeights_, gradientScale_, logicalWeights_] := Scope[
		
	batchedInputDims = Select[inputDims, ContainsQ[BatchSize]] /. BatchSize -> batchSize;
	batchedArrayKeys = Keys @ batchedInputDims;

	inputDims = inputDims /. BatchSize -> batchSize;
	allArgs = Join[weightArrays, fixedArrays, inputDims];
	
	gradArrayKeys = If[EmptyQ[inputGradients] && EmptyQ[weightGradients], {},
		Cases[
			Join[Values[inputGradients], Values[weightGradients]], 
			mx_String | NDSequenceArray[mx_String, _] | NDSubArray[mx_String, _, _] :> mx
		] (* might contain duplicates but that doesn't matter *)
	];

	allDims = getArrayDims /@ allArgs;
	inferred = MXSymbolInferShape[mxSymbol, allDims];
	{mxArgDims, mxAuxDims, mxOutDims} = Lookup[inferred, {"ArgumentArrays", "AuxilliaryArrays", "OutputArrays"}];
	(* ^ we don't need this, because we already know all sizes. but we need to get the 
	canonical order of keys to use to call bind with. *)

	outBytes = dimsToBytes[mxOutDims];

	totalArgBytes = dimsToBytes[mxArgDims]; totalGradBytes = dimsToBytes[KeyTake[mxArgDims, gradArrayKeys]];
	totalBytes = totalArgBytes + totalGradBytes + dimsToBytes[mxAuxDims] + outBytes;
	If[totalBytes > memoryLimit, OOMPanic[batchSize, totalBytes, memoryLimit, context]];
	batchedArgBytes = dimsToBytes[batchedInputDims]; batchedGradBytes = dimsToBytes[KeyTake[batchedInputDims, gradArrayKeys]];
	batchedBytes = batchedArgBytes + batchedGradBytes + outBytes;
	nonBatchedBytes = totalBytes - batchedBytes;
	(* ^ here we calculate the total batch and non-batch cost of all arrays. a bit complicated because of how
	mxnet groups all input and weight arrays into one set *)

	argOrder = Keys @ mxArgDims; nArgs = Length[argOrder];
	auxOrder = Keys @ mxAuxDims;
	argOrder = Join[argOrder, Complement[batchedArrayKeys, argOrder]];
	(* ^ some len arrays don't get used in the graph and so will be left off by infershape,
	but we want mxArgArrays to still contain them, so we add them back here. luckily mxExecutorBind
	will ignore these extra values at the end! TODO: DOES IT? *)

	mxArgArrays = KeyTake[allArgs, argOrder];
	mxAuxArrays = KeyTake[auxArrays, auxOrder];
	(* ^ get these guys into the right order *)

	$DefaultContext = context;
	$DefaultDataTypeCode = dataType;
	initf = arrayInitFunc[arrayCaching, mxArgArrays, mxAuxArrays];
	ComposeTo[mxArgArrays, initf];
	ComposeTo[mxAuxArrays, initf];

	If[EmptyQ[gradArrayKeys],
		mxGradArrays = AssociationThread[argOrder, $NullNDArray];
		mxGradCodes = CTable[0, nArgs];
		outputGradients = None;
	,
		skipGrad = Complement[argOrder, gradArrayKeys];
		mxArgDims2 = mxArgDims;
		mxArgDims2[[skipGrad]] = Null;
		mxGradArrays = toNDArray /@ mxArgDims2;
		mxGradCodes = Replace[Values[mxGradArrays], {$NullNDArray -> 0, _ -> 1}, {1}];
		outputGradients = KeyMap[NetPortGradient, outputs];
	];
	(* grad codes: None -> 0, "Write" -> 1, "InPlace" -> 2, "Add" -> 3 *)
	
	(* create new managed expression *)
	mxExecutor = System`Private`SetNoEntry @ CreateManagedLibraryExpression["MXExecutor", MXExecutor];

	If[zeroArrays =!= {}, NDArraySetConstant[Lookup[mxArgArrays, zeroArrays], 0.]];

	$ExecutorLogger[{NetExecutorCreate, inputDims, context, dataType, memoryLimit}];

	(* bind *)
	mxCall[
		mxExecutorBind,
		MLEID @ mxSymbol,
		context,
		MLEID /@ Values[mxArgArrays], 
		MLEID /@ Values[mxGradArrays],
		mxGradCodes,
		MLEID /@ Values[mxAuxArrays],
		MLEID @ $NullExecutor,
		MLEID @ mxExecutor
	];

	mxOutArrays = getExecutorOutputArrays[mxSymbol, mxExecutor];
	
	If[outputGradients === None, 
		outGradArrays = None;
	, 
		outGradArrays = NDArrayCloneShape[#, gradientScale]& /@ mxOutArrays;
		batchedBytes += outBytes;
	];
	arrayByteCounts = <|"BatchedArrays" -> batchedBytes, "NonBatchedArrays" -> nonBatchedBytes|>;

	ioMapping = {{inputs, inputStates}, {outputs, outputStates, metrics}, {inputGradients}, {outputGradients}};
	finalIO = Join[remapIO[ioMapping], remapW[weightGradients, matchWeights]];

	syncFunction = Hold;
	If[matchWeights === "SharedViaCache" && MatchQ[arrayCaching, _Hold],
		$syncCode = Internal`Bag[]; $acache = First[arrayCaching];
		KeyValueScan[pushSubArraySync, logicalWeights];
		syncFunction = Internal`BagPart[$syncCode, All, toFunc];
	];
	(* ^ ensure that if we are constructing a validation executor and it has fused arrays,
	we generate the code to sync their parts from the training executors non-fused arrays, if these exist *)

	NetExecutor @ Association[
		"MXExecutor" -> mxExecutor,
		"MXSymbol" -> mxSymbol, 
		"Context" -> context,
		"DataType" -> dataType,
		"BatchSize" -> batchSize,
		"BatchedKeys" -> batchedArrayKeys,
		"MXArrayData" -> {mxArgArrays, mxAuxArrays, mxGradArrays, mxGradCodes, mxOutArrays, outGradArrays, mxArgDims},
		"MXAuxArrays" -> mxAuxArrays, (* <- duplicated because MultiExecutor also must exposes this *)
		"IOMapping" -> ioMapping,
		"Arrays" -> finalIO,
		"SyncFunction" -> syncFunction,
		"GradientScale" -> gradientScale,
		"ArrayByteCounts" -> arrayByteCounts
	]
]

pushSubArraySync[name_, sub_NDSubArray] := ModuleScope[
	parent = Lookup[$acache, Key[{$DefaultContext, MXManglePath @ name}], None];
	If[parent =!= None,
		child = sliceSubArray[mxArgArrays, sub];
		Internal`StuffBag[$syncCode, Unevaluated @ NDArrayCopyTo[child, parent]]
		(* ^ if we are sharing arrays and another executor already has a NDArray for this sub array, we can
		initialize from that. this is used for synching validation executors from training executors *)
	];
]

SetHoldAll[toFunc];
toFunc[] := Hold;
toFunc[args___] := Function[CompoundExpression[args]];

SetUsage @ "
NetExecutor[assoc$] wraps an MXExecutor handle.
* assoc has the following keys:
| 'MXExecutor', 'MXSymbol' | MXNetLink handles |
| 'Context' | the NDArray context code |
| 'BatchedKeys' | mxnames of arg nodes that should be batch-resized |
| 'MXArrayData' | tuple containing all actual NDArrays |
| 'IOMapping' | relationship between top-level inputs, outputs, etc. and their arg node constitutients |
| 'Arrays' | actual top-level inputs, outputs etc. and their NDArray constituents |
* the following methods can efficiently derive new NetExecutors from old ones: NetExecutorInherit, NetExecutorReshape."

getArrayDims[e_NDArray] := NDArrayDimensions[e];
getArrayDims[NDSubArray[_, _, dims_]] := dims;
getArrayDims[ra_NumericArray] := Dimensions[ra];
getArrayDims[SymbolicRandomArray[_, dims_]] := dims;
getArrayDims[l_List] := l;

toNDArray[dims_List] := NDArrayCreateZero[dims];
toNDArray[ra_NumericArray] := NDArrayCreate[ra, $DefaultContext, $DefaultDataTypeCode];
toNDArray[Null] := $NullNDArray;
toNDArray[other_] := other;

toNDArray[SymbolicRandomArray[dist_, dims_]] := Scope[
	nd = NDArrayCreateEmpty[dims];
	If[Head[dist] === NNConstantDist, 
		NDArraySetConstant[nd, First[dist]],
		NDArraySet[nd, RandomVariate[dist, dims]];
	];
	nd
];

getExecutorOutputArrays[sym_MXSymbol, exec_MXExecutor] := Scope[
	outNames = MXSymbolOutputs[sym];
	outArrays = Table[CreateManagedLibraryExpression["NDArray", NDArray], Length[outNames]];
	mxCall[mxExecutorOutputs, MLEID @ exec, MLEID /@ outArrays];
	AssociationThread[outNames, outArrays]
];

toCachedNDArray[ra_NumericArray] := ToCachedNDArray[ra, $DefaultContext, $DefaultDataTypeCode];
toCachedNDArray[other_] := toNDArray[other];

arrayInitFunc[Hold[sym_], _, _] := MapIndexed[cacheArrTo[sym]];
arrayInitFunc[True, _, _] = Map[toCachedNDArray];
arrayInitFunc[False, _, _] = Map[toNDArray];

(* this ignores actual NumericArrays etc and just creates 1-initialized arrays,
for when we need to time an executor but don't care about behavior *)
arrayInitFunc["DummyArrays", _, _] := Map[toDummyNDArray];

(* this is like DummyArrays but minimizes memory usage by having all
arrays share from a single array *)
arrayInitFunc["SharedDummyArrays", a1_, a2_] := ModuleScope[
	maxsize = Max @ Map[getArraySize, Join[a1, a2]]; 
	masternd = toDummyNDArray[{maxsize}];
	Map[NDArrayReshape[masternd, getArrayDims[#]]&]
];


toDummyNDArray[e_] := Scope[
	dims = getArrayDims[e];
	arr = NDArrayCreateEmpty[dims];
	NDArraySetConstant[arr, 1.0];
	arr
];

arrayInitFunc[_, _, _] := $Unreachable;

getArraySize[e_] := Times @@ getArrayDims[e];

SetHoldFirst[cacheArrTo];

cacheArrTo[sym_][array:(_NumericArray | _SymbolicRandomArray), {Key[path_]}] := With[
	{path2 = {$DefaultContext, path}},
	Lookup[sym, Key @ path2, sym[path2] = toNDArray @ array]
]

cacheArrTo[sym_][other_, _] := toNDArray[other];

(******************************************************************************)

(*

NetExecutorInherit makes some key assumptions: the executor it is inheriting from shares
almost all arrays in common with it, except for the 'IO' arrays.

This *has* to be true of symbols produced via unrolling.

It means we can do substantially less bookkeeping, and re-use everything we produced for
the original executor.
*)

PackageScope["NetExecutorInherit"]

NetExecutorInherit[symbol_MXSymbol, NetExecutor[exec_], inputDims_Association] := Scope[
	
	$DefaultContext = exec["Context"];
	$DefaultDataTypeCode = exec["DataType"];
	mxArgArrays = exec[["MXArrayData", 1]];

	batchSize = exec["BatchSize"];

	(* this is the only change: new input dimensions *)
	newDims = KeyValueMap[#1 -> NDArrayReshape[mxArgArrays[#1], #2 /. BatchSize -> batchSize]&, inputDims];
	AssociateTo[mxArgArrays, newDims];

	$ExecutorLogger[{NetExecutorInherit, inputDims, $DefaultContext, $DefaultDataTypeCode}];

	bindDerivedExecutor[exec, symbol, mxArgArrays]
];


PackageExport["$DisableNewReshapeAPI"]

SetUsage @ "
$DisableNewReshapeAPI (default False) specifies whether the new MXNet reshaping API should be used."

$DisableNewReshapeAPI = False;

(******************************************************************************)

(*

NetExecutorReshape makes even more assumptions than NetExecutorInherit: the symbol must
be the same, and all that changes is the input sizes (probably only the batch size,
though for sequence networks that don't need unrolling it could even be the first
two parameters).

NDArray space of the inputs will be reused if possible.

*)

PackageScope["NetExecutorReshape"]

replaceValues[<||>, {}] := <||>;
replaceValues[oldAssoc_, newValues_] :=
	AssociationThread[Keys @ oldAssoc, newValues];

NetExecutorReshape[NetExecutor[execData_], inputDims_] /; !$DisableNewReshapeAPI := Scope[

	exec = execData;
	$ExecutorLogger[{NetExecutorReshape, inputDims, exec["Context"]}];

	{mxArgArrays, mxAuxArrays, mxGradArrays, mxGradCodes, mxOutArrays, outGradArrays, mxArgDims} = exec["MXArrayData"];

	If[IntegerQ[inputDims],
		batchSize = inputDims;
		inputDims = Map[
			ReplacePart[getArrayDims[#], 1 -> batchSize]&,
			KeyTake[mxArgArrays, exec["BatchedKeys"]]
		]
	,
		batchSize = First @ First @ DeleteMissing @ Lookup[inputDims, exec["BatchedKeys"]];
	];

	Assert @ AssociationQ @ inputDims;

	mxArgDims = Join[mxArgDims, inputDims];

	{newExecutor, newArgArrays, newGradArrays, newAuxArrays, newOutArrays} = 
		MXExecutorReshapeFast[exec["MXExecutor"], mxArgDims, exec["Context"]];

	If[outGradArrays =!= None,
		gradientScale = exec["GradientScale"];
		outGradArrays = MapThread[
			(grad2 = NDArrayReshape[#1, NDArrayDimensions[#2]]; NDArraySetConstant[grad2, gradientScale]; grad2)&, 
			{outGradArrays, replaceValues[outGradArrays, newOutArrays]}
		]
	];

	mxArgArrays = replaceValues[mxArgArrays, newArgArrays];
	mxAuxArrays = replaceValues[mxAuxArrays, newAuxArrays];
	mxGradArrays = replaceValues[mxGradArrays, newGradArrays];
	mxOutArrays = replaceValues[mxOutArrays, newOutArrays];

	exec["MXExecutor"] = newExecutor;
	exec["MXArrayData"] = {mxArgArrays, mxAuxArrays, mxGradArrays, mxGradCodes, mxOutArrays, outGradArrays, mxArgDims};
	exec["MXAuxArrays"] = mxAuxArrays;
	exec["BatchSize"] = batchSize;

	(* re-map the input, output, gradient fields *)
	AssociateTo[exec["Arrays"], remapIO[exec["IOMapping"]]];

	NetExecutor @ exec
]

(******************************************************************************)

(* TODO: Once we are confident in the new reshaping API, remove the below definitions of NetExecutorReshape *)

PackageScope["NetExecutorReshape"]

NetExecutorReshape[NetExecutor[exec_], inputDims_Association] /; $DisableNewReshapeAPI := Scope[
	
	$DefaultContext = exec["Context"];
	$DefaultDataTypeCode = exec["DataType"];
	symbol = exec["MXSymbol"];
	mxArgArrays = exec[["MXArrayData", 1]];
	mxArgDims = exec[["MXArrayData", 7]];

	(* use MXNet shape inference to quickly propogate the new dimensions to all arrays *)
	mxArgDims = Join[exec[["ArgDims"]], getArrayDims /@ inputDims];

	$ExecutorLogger[{NetExecutorReshape, mxArgDims, $DefaultContext}];

	inferred = MXSymbolInferShape[symbol, mxArgDims];
	mxInferDims = inferred["ArgumentArrays"];

	(* find all dimensions that changed, and reshape the corresponding arrays *)
	mxArgArrays = Association @ KeyValueMap[
		#1 -> NDArrayReshape[#2, mxInferDims[#1]]&,
		mxArgArrays
	];

	bindDerivedExecutor[exec, symbol, mxArgArrays]
]

NetExecutorReshape[NetExecutor[exec_], n_Integer] /; $DisableNewReshapeAPI := Scope[

	$DefaultContext = exec["Context"];
	$DefaultDataTypeCode = exec["DataType"];
	batchedKeys = exec["BatchedKeys"];
	mxArgArrays = exec[["MXArrayData", 1]];

	$ExecutorLogger[{NetExecutorReshape, n, $DefaultContext}];

	exec["BatchSize"] = $batchsize = n;
	mxArgArrays = MapAt[reshapeBatch, mxArgArrays, List /@ batchedKeys];

	bindDerivedExecutor[exec, exec["MXSymbol"], mxArgArrays]
];

reshapeBatch[nd_NDArray] :=
	NDArrayReshape[nd, ReplacePart[NDArrayDimensions[nd], 1 -> $batchsize]];

(*****************************************************)
(* shared by NetExecutorInherit and NetExecutorReshape *)

bindDerivedExecutor[exec_, mxSymbol_, mxArgArrays_] := Scope[

	{unused, mxAuxArrays, mxGradArrays, mxGradCodes, mxOutArrays, outGradArrays, mxArgDims} = exec["MXArrayData"];
	gradientScale = exec["GradientScale"];
	
	(* put in the new value of mxArgArrays *)
	exec = exec;
	exec[["MXArrayData", 1]] = mxArgArrays;

	(* create new managed expression *)
	mxExecutor = System`Private`SetNoEntry @ CreateManagedLibraryExpression["MXExecutor", MXExecutor];

	(* we have to make a copy because the symbol retains some subtle state that will interfere with
	reshaping otherwise *)
	mxSymbol = MXSymbolCopy[mxSymbol];
	exec["MXSymbol"] = mxSymbol;

	(* bind *)
	mxCall[
		mxExecutorBind,
		MLEID @ mxSymbol,
		$DefaultContext,
		MLEID /@ Values[mxArgArrays], 
		MLEID /@ Values[mxGradArrays],
		mxGradCodes,
		MLEID /@ Values[mxAuxArrays],
		MLEID @ exec["MXExecutor"], (* <- share memory with previous executor *)
		MLEID @ mxExecutor
	];

	exec["MXExecutor"] = mxExecutor;

	(* get the fresh output arrays *)
	mxOutArrays = getExecutorOutputArrays[mxSymbol, mxExecutor];
	exec[["MXArrayData", 5]] = mxOutArrays;
	If[outGradArrays =!= None, 
		exec[["MXArrayData", 6]] = MapThread[
			(grad2 = NDArrayReshape[#1, NDArrayDimensions[#2]]; NDArraySetConstant[grad2, gradientScale]; grad2)&, 
			{outGradArrays, mxOutArrays}
		];
	];

	(* re-map the input, output, gradient fields *)
	AssociateTo[exec["Arrays"], remapIO[exec["IOMapping"]]];

	NetExecutor @ exec
];

(****************************************************)
(* used only by NetExecutorCreate                   *)

remapW[<||>, _] := <|"WeightGradients" -> <||>, "Weights" -> <||>|>;

remapW[wspec_, matchWeights_] := <|
	"WeightGradients" -> remapWArrays[wspec, mxGradArrays],
	"Weights" -> If[TrueQ @ matchWeights, remapWArrays[wspec, mxArgArrays], <||>]
|>

remapWArrays[arrays_, assoc_] := Replace[
	arrays, 
	{
		mx_String :> Lookup[assoc, mx, Lookup[mxAuxArrays, mx]],
		sub_NDSubArray :> sliceSubArray[assoc, sub]
	},
	{1}
]

sliceSubArray[assoc_, NDSubArray[name_, offset_, dims_]] := 
	NDArrayReshape[
		NDArraySliceFast[assoc[name], offset, offset + (Times @@ dims)],
		dims
	]

(****************************************************)
(* used by bindDerivedExecutor and NetExecutorCreate *)

remapIO[iospecs_] := Association @ MapThread[
	Thread[#1 -> remapIOArrays[#1, #2, #3]]&,
	{
		{{"Inputs", "InputStates"}, {"Outputs", "OutputStates", "Metrics"}, {"InputGradients"}, {"OutputGradients"}},
	 	iospecs, 
	 	{mxArgArrays, mxOutArrays, mxGradArrays, outGradArrays}
	}
]

remapIOArrays[_, arrays_, assoc_] := Replace[
	arrays,
	{
		mx_String :> assoc[mx],
		NDSequenceArray[mx_String, len_String] :> NDSequenceArray[setPad @ assoc[mx], setOne @ Lookup[mxArgArrays, len, mxOutArrays[len]]],
		(* ^ if it a seq node is from outside, its in mxArgArrays, if its derived, its in mxOutArrays *)
		group_Association :> Replace[group, mx_String :> assoc[mx], {1}],
		(* ^ this recurses through nested association that specifically can occur in Metrics *)
		NDSparseCountsArray[mx_String, dims_] :> NDSparseCountsArray[assoc[mx], dims],
		NDNoTotalArray[mx_String] :> NDNoTotalArray[assoc[mx]],
		bad_ :> Panic["InvalidIOSpec", "`` is not a valid IO spec.", bad]
	},
	{2}
];

(* we do this to be sure there are no NaNs hiding in the new IO seq arrays.
if we ever want to fetch the next bucket while waiting for the current bucket to finish evaluating, 
this should be removed *)
setPad[nd_] := (NDArraySetConstant[nd, $SequencePaddingValue]; nd)
setOne[nd_] := (NDArraySetConstant[nd, 1.0]; nd)

setPad[nd_] := nd;
setOne[nd_] := nd;

(******************************************************************************)

PackageExport["NetExecutorForward"]

SetUsage @ "
NetExecutorForward[NetExecutor[$$], trainMode$] does a forward pass, modifying the output NDArrays of the computation graph.
* trainMode$ should be True if NetExecutorBackward is going to be called afterwards."

NetExecutorForward[NetExecutor[exec_], trainMode_:False] :=
	MXExecutorForward[exec["MXExecutor"], trainMode];


PackageExport["NetExecutorBackward"]

SetUsage @ "
NetExecutorBackward[NetExecutor[$$]] does a backward pass to get the gradients. 
* The executor must have been created with non-trivial gradientData request, otherwise there \
will be no allocated out grads in MXArrayData."

NetExecutorBackward[NetExecutor[exec_]] := 
	MXExecutorBackward[exec["MXExecutor"], Values @ exec["MXArrayData"][[6]]];


(******************************************************************************)

NetExecutor /: Normal[NetExecutor[data_]] := data

NetExecutor[exec_][query__String] := exec[query]

PackageExport["NetExecutorMemoryInformation"]

SetHoldFirst[echoSize]
echoSize[e_] := (Echo[HoldForm[e] -> Cases[e, nd_NDArray :> NDArrayByteCount @ nd, Infinity]]; e);
echoSize[e_] := e;

splitLookup[assoc_, keys_] := Scope[
	a = Lookup[assoc, keys, Null];
	b = Values @ KeyDrop[assoc, keys];
	{a, b}
];

NetExecutorMemoryInformation[NetExecutor[exec_]] := Scope[
	internalBytes = MXExecutorRequiredMemory[exec["MXExecutor"]] * 2^20;
	res = Append[exec["ArrayByteCounts"], "Internal" -> internalBytes];

	batchedKeys = exec["BatchedKeys"];
	{mxArgArrays, mxAuxArrays, mxGradArrays, mxGradCodes, mxOutArrays, outGradArrays, argDims} = exec["MXArrayData"];

	(*
	Print[batchedKeys];
	Print[NDArrayByteCount /@ mxArgArrays];
	Print[NDArrayByteCount /@ mxOutArrays];
	{batchedArgs, unbatchedArgs} = splitLookup[mxArgArrays, batchedKeys];
	{batchedGrads, unbatchedGrads} = splitLookup[DeleteCases[$NullNDArray] @ mxGradArrays, batchedKeys];

	batchedArrays = totalNDArrayByteCount[echoSize @ batchedArgs, echoSize @ batchedGrads, echoSize @ mxOutArrays, echoSize @ outGradArrays];
	nonBatchedArrays = totalNDArrayByteCount[echoSize @ unbatchedArgs, echoSize @ unbatchedGrads, echoSize @ mxAuxArrays];

	res2 = Association[
		"BatchedArrays" -> batchedArrays,
		"NonBatchedArrays" -> nonBatchedArrays,
		"Internal" -> internalBytes
	];

	If[res =!= res2, Print["MISMATCH: ", res, res2]];
	(* NOTE: this code should be uncommented if you ever want to double check that the rather complex array bytecount
	estimation code in NetExecutorCreate is ok *)
	*)

	(* there ARE small mismatches, but they come from the fact that seq length vectors that don't participate in the
	DAG become invisible to MXInferShapes, and hence they don't get counted. But these are so tiny the chances they 
	will tip the balance into an OOM condition are basically zero, so we don't care *)

	res
];

totalNDArrayByteCount[args___] := Total @ Cases[{args}, nd_NDArray :> NDArrayByteCount[nd], Infinity];

(******************************************************************************)

PackageExport["NetExecutor"]

(* Custom Display form *)

DefineCustomBoxes[NetExecutor,
	exec:NetExecutor[_Association] :> NetExecutorBoxes[exec]
];

DefineCustomBoxes[MultiExecutor,
	exec:MultiExecutor[_Association] :> NetExecutorBoxes[exec]
];

NetExecutorBoxes[exec:(head_Symbol)[data_]] := Scope[
	isExec = head === NetExecutor;
	If[!isExec, execDatas = data[["Executors", All, 1]]];
	plot = With[{sym = If[isExec, data["MXSymbol"], execDatas[[1, "MXSymbol"]]]}, 
		Dynamic[Framed[MXSymbolPlot[sym], FrameStyle -> LightGray], TrackedSymbols :> {}]
	];
	context = If[isExec, UnparseContext @ data["Context"], data["Contexts"]];
	execID = If[isExec, MLEID@data["MXExecutor"], MLEID /@ execDatas[[All, "MXExecutor"]]];
	other = Map[makeItem[#, data[#]]&, If[isExec, {"BatchSize", "Context", "ArrayByteCounts"}, {"BatchSizesList", "Contexts"}]];
	arrayInfo = Map[
		makeItem[#, data["Arrays", #]]&, 
		{"Inputs", "Outputs", "WeightGradients", "OutputGradients", "Weights",
			"InputGradients", "InputStates", "OutputStates", "Metrics"
		}
	];
	BoxForm`ArrangeSummaryBox[
		head, None, None,
		Join[arrayInfo, other],
		{makeItem["ExecutorID", execID], plot},
		StandardForm
	]
];
