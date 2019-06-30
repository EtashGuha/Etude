Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXPredictIterator"]

SetUsage @ "
MXPredictIterator[MXExecutorData[$$], arrays$] creates an Iterator[$$] object which returns the \
prediction of a network defined by the network MXExecutorData[$$], given an Association of input \
NDArray's arrays$. 
MXPredictIterator[MXExecutorData[$$], Iterator[$$]] is the same as MXPredictIterator[MXExecutorData[$$], arrays$], \
only the arrays are given in the form of an Iterator[$$] object.
* all inputs inputAssoc$ must be batched parameters.
* the datatype and context information is inherited from the MXExecutorData[$$] object.
* the following options are available: 
  * 'ReturnInputs' -> False: whether to return the inputs along with the outputs.
  * 'ExecutorResizing' -> True: use executor resizing to evaluate data smaller than MXExecutor[$$] is \
initialized to deal with. If False, padding will be used, which can be slower."

(* Iterator version *)
Options[MXPredictIterator] = {
	"ReturnInputs" -> False,
	"PaddingMethod" -> None
}

MXPredictIterator[executor_MXExecutorData, data_Iterator, OptionsPattern[]] := CatchFailure @ Scope[
	MapIterator[batchPredictionFunction[executor, #, OptionValue@"ReturnInputs"]&, data]
]

MXPredictIterator[executor_MXExecutorData, data_Association, OptionsPattern[]] := CatchFailure @ Scope[
	UnpackOptions[returnInputs, paddingMethod];
	
	firstInputName = First@Keys@data;
	batchSizeExec = NDArrayLength@executor["ArgumentArrays", firstInputName];	
	iter = MXBatchIterator[data, batchSizeExec, "PaddingMethod" -> paddingMethod];
	MapIterator[batchPredictionFunction[executor, #, OptionValue@"ReturnInputs"]&, iter]
]

batchPredictionFunction[exec_MXExecutorData, {data_, pad_}, returnInputs_] := CatchFailure @ Scope[
	(* We don't know batch size: *)
	firstInputName = First@Keys@data;
	batchSizeData = NDArrayLength@data[firstInputName];
	
	(* infer batch size from initialized NDArrays *)
	batchSizeExec = NDArrayLength@exec["ArgumentArrays", firstInputName];
	(* Don't allow data batch larger than executor batch *)
	If[batchSizeData > batchSizeExec, ThrowFailure["invalidDataBatchDim"]];
	(* If data batch is smaller than executor + no padding, dynamically resize executor *)
	execFinal = exec;
	If[batchSizeData < batchSizeExec, (* check whether we need to do exec resizing *)
		shapes = NDArrayDimensions /@ data;
		execFinal = MXExecutorReshape[exec, shapes]
	];
	
	(* Set input vals *)
	KeyValueMap[NDArraySet[execFinal["ArgumentArrays", #1], #2]&, data];	
	(* Do forward pass *)
	MXExecutorForward[execFinal["Executor"], False];
	
	(* deal with outputs *)
	outputs = execFinal["OutputArrays"];
	If[IntegerQ@pad,  (* check whether we need to do slicing *)
		outputs = NDArraySlice[#, 1, pad]& /@ outputs
	];
	
	(* deal with non-return input case *)
	If[returnInputs === False, Return@outputs];
	
	(* deal with return input case *)		
	inputs = data;
	If[IntegerQ@pad, (* check whether we need to do slicing *)
		inputs = NDArraySlice[#, 1, pad]& /@ inputs
	];	
	<|
		"Outputs" -> outputs,
		"Inputs" -> inputs
	|>
]

(******************************************************************************)

PackageExport["MXEvaluate"]

SetUsage @ "
MXEvaluate[MXExecutorData[$$], <|key$1 -> {$$}, $$|>, {encoder$1, $$}, {outname$1, $$}] \
does a forward pass on a network defined by the executor MXExecutorData[$$] using the association of \
input data <|key$1 -> {$$}, $$|>, which must match the InputArray keys in the executor. \
The encoders {encoder$1, $$} are applied to the input data in canonical order. \
Only the outputs in {outname$1, $$} are returned as a list in the same order. 
MXEvaluate[MXExecutorData[$$], {{$$}, {$$}, $$}, $$] \
does the same, but relies on the order of InputArrays in the executor."

MXEvaluate[executorData_MXExecutorData, input_Association | input_List, outputNames_List, ndsetters_:NDArraySet, ndgetters_:NDArrayGet, tmode_:False] := Scope[
	(* Create outputs *)
	inputArrays = executorData["InputArrays"];
	If[AssociationQ[input],
		{inputNames, inputs} = KeysValues[input];
		inputND = Lookup[inputArrays, inputNames, Panic["MissingInput"]];
	,
		{inputNames, inputND} = KeysValues[inputArrays];
		inputs = input;
	];
	If[!ListQ[ndsetters], ndsetters = ConstantArray[ndsetters, Length[input]]];
	If[!ListQ[ndgetters], ndgetters = ConstantArray[ndgetters, Length[input]]];
	outputND = Lookup[executorData["OutputArrays"], outputNames];
	executor = executorData["Executor"];
	(* get batch dims: assume there is at least one input array *)	
	batchSizeData = Length @ First @ inputs;
	batchSizeExecutor = NDArrayLength @ First @ inputND;
	(* Will store outputs in Bag *)
	outputBags = Table[Internal`Bag[], {Length @ outputND}];
	(* Loop over slices *)
	Do[
		(* Get data slice + apply encoders *)
		maxIndex = Min[batchSizeData, batchSizeExecutor * i];
		minIndex = batchSizeExecutor * (i - 1) + 1;
		(* Check whether to use Executor resizing *)
		If[(maxIndex - minIndex + 1) < batchSizeExecutor,
			reshapedExecutor = MXExecutorReshapeUniform[executorData, maxIndex - minIndex + 1];
			inputND = Lookup[reshapedExecutor["InputArrays"], inputNames];
			outputND = Lookup[reshapedExecutor["OutputArrays"], outputNames];
			executor = reshapedExecutor["Executor"];
		];	
		(* Do forward pass *)
		ScanThread[#1[#2, Take[#3, {minIndex, maxIndex}]]&, {ndsetters, inputND, inputs}];
		MXExecutorForward[executor, tmode];
		(* add output to bags *)
		ScanThread[Internal`StuffBag[#3, #1[#2], 1]&, {ndgetters, outputND, outputBags}];
	,
		{i, 1, Ceiling[batchSizeData/batchSizeExecutor]}
	];
	(* return outputs *)
	Internal`BagPart[#, All]& /@ outputBags
]

(******************************************************************************)

(* Utility function: loads a model parameter file into form 
<| \"AuxilliaryArrays\" -> NDArray[...], \"ArgumentArrays\" -> NDArray[...]|>
*)

PackageExport["MXModelLoadParameters"]

SetUsage @ "
MXModelLoadParameters[file$] loads a MXNet parameter file, returning an association \
<| 'AuxilliaryArrays' -> {$$}, 'ArgumentArrays' -> {$$} |>."

Options[MXModelLoadParameters] = {
	"Context" :> $DefaultContext
};

MXModelLoadParameters[parameterFile_, opts:OptionsPattern[]] := CatchFailure @ Scope[
	UnpackOptions[context];	
	params = NDArrayImport[parameterFile];
	If[FailureQ[params], Return[params]];
	params = KeyMap[StringSplit[#, ":"]&, params];
	auxParams = KeySelect[params, (First@# == "aux")&];
	argParams = KeySelect[params, (First@# == "arg")&];
	auxParams = Association@KeyValueMap[
		(Last@#1 -> NDArrayCreate[#2, context])&, 
		auxParams
	];
	argParams = Association@KeyValueMap[
		(Last@#1 -> NDArrayCreate[#2, context])&, 
		argParams
	];
	<|"ArgumentArrays" -> argParams, "AuxilliaryArrays" -> auxParams|>
]
