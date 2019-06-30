Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXSymbolBind"]

SetUsage @ "
MXSymbolBind[MXSymbol[$$], <|'arg$1' -> spec$1, $$|>, opts$] instantiates the Variable \
symbols in the provided MXSymbol with NDArrays and returns both an MXExecutor[$$] object \
and an association of initialized arrays. The association of arguments args$ can consist \
of RawArrays, dimension lists, or NDArrays. If NDArrays are specified, they will be used \
without copying. RawArrays will be copied. Sufficient args$ (or their shapes) must be \
specified to infer all other shapes.
* The following options are available:
| 'Context' | $DefaultContext | the context of the output executor, along with the automatically generated NDArrays |
| 'SharedExecutor' | None | executor whose memory can be re-used; intended for runtime reshaping, variable length sequences, etc. |
| 'AuxilliaryArrays' | <||> | supply auxilliary arrays to bind to Executor |
| 'GradientArrays' | None | an association of existing NDArrays to re-use for the grad arrays |
| 'GradientUpdateMethod' | 'Write' | one of 'Write', 'Add', None, specifies what method to use to update gradients during a backward pass |
* When specifying 'SharedExecutor', the new executor and the original \
executor will share intermediate state memory and should not be used \
concurrently.
"

Options[MXSymbolBind] = {
	"Context" :> $DefaultContext,
	"SharedExecutor" -> None,
	"AuxilliaryArrays" -> <||>,
	"GradientArrays" -> None,
	"GradientUpdateMethod" -> "Write"
};

getInDim[l_List] := l;
getInDim[e_NDArray] := NDArrayDimensions[e];
getInDim[ra_NumericArray] := Dimensions[ra];
getInDim[e_] := $Unreachable;

PackageScope["mxlMXExecutorBind"]

mxlDeclare[mxlMXExecutorBind, {"Integer", "Integer", "IntegerVector", "IntegerVector", "IntegerVector", "IntegerVector", "Integer", "Integer"}];

MXSymbolBind[symbol_MXSymbol, argumentArrays_Association, opts:OptionsPattern[]] := Scope[
	
	UnpackOptions[
		context, sharedExecutor,
		auxilliaryArrays, 
		gradientArrays, gradientUpdateMethod
	];
	
	inferredShapes = MXSymbolInferShape[symbol, getInDim /@ argumentArrays];
		
	mxArgumentArrays = inferredShapes["ArgumentArrays"];
	mxAuxilliaryArrays = Join[inferredShapes["AuxilliaryArrays"], auxilliaryArrays];
	argKeys = Keys[mxArgumentArrays];

	$context = ToContextCode[context];
	
	(* Initialize argument + aux arrays *)
	mxArgumentArrays = initArray /@ mxArgumentArrays;
	mxAuxilliaryArrays = initArray /@ mxAuxilliaryArrays;
	
	(* Resolve the len arrays now that they've been created *)
	mxArgumentArrays = mxArgumentArrays /. 
		NDSequenceArray[nd_, lens_String] :> 
			RuleCondition @ NDSequenceArray[nd, Lookup[mxArgumentArrays, lens, Panic["UnresolvableNDSequenceArray"]]];
	
	Which[
		gradientUpdateMethod === None,
			(* fast path for no gradients *)
			mxGradientArrays = gradientUpdateMethod = ConstantAssociation[argKeys, None],
		StringQ[gradientUpdateMethod],
			(* If single gradient update method given, use for all argument arrays *)
			gradientUpdateMethod = ConstantAssociation[argKeys, gradientUpdateMethod],
		AssociationQ[gradientUpdateMethod],
			(* otherwise, ensure canonical ordering + completeness *)
			gradientUpdateMethod = Join[ConstantAssociation[argKeys, None], gradientUpdateMethod],
		True,
			Panic["BadGradUpdateMethodSpec"]
	];

	(* init gradients: three cases, 1) supplied gradient NDArray 2) Null grad 3) need to init to zero *)
	gradInitFunc = Function[key,
		Which[
			gradientUpdateMethod[key] === None, None,
			(gradientArrays =!= None) && KeyExistsQ[gradientArrays, key], gradientArrays[key],
			True, NDArrayCreateZero[inferredShapes["ArgumentArrays", key], $context]
		]
	];
	mxGradientArrays = AssociationMap[gradInitFunc, argKeys];

	(* check whether existing executor can be used. If not, create new executor *)
	Switch[sharedExecutor,
		None, sharedExecutor = CreateManagedLibraryExpression["MXExecutor", MXExecutor],
		_MXExecutorData, sharedExecutor = sharedExecutor["Executor"],
		_, Null
	];
	
	mxSymbolBindFast[
		symbol, $context,
		mxArgumentArrays, mxGradientArrays, mxAuxilliaryArrays, 
		gradientUpdateMethod, sharedExecutor
	]
]

initArray[in_] := Match[in,
	_NDArray :> in,
	_NumericArray :> NDArrayCreate[in, $context],
	_List :> NDArrayCreateZero[in, $context],
	NDSequenceArray[data_, lens_] :> NDSequenceArray[initArray[data], lens],
	Panic["InvalidArraySpec"]
];


PackageScope["mxlMXExecutorOutputs"]

mxlDeclare[mxlMXExecutorOutputs, {"Integer", "IntegerVector"}];

(* fast binding: assumes all preprocessing is already done. Used internally *)
mxSymbolBindFast[symbol_, context_, argArrays_, gradArrays_, auxArrays_, gradUpdateMethod_, sharedExecutor_] := Scope[
	
	(* Create Output executor *)
	executorHandle = System`Private`SetNoEntry @ CreateManagedLibraryExpression["MXExecutor", MXExecutor];
	
	(* Deal with some grad arrays being None. Use null array in this case *)
	newGradArrays = If[# === None, $NullNDArray, #]& /@ gradArrays;

	(* bind *)
	mxlCall[
		mxlMXExecutorBind,  
		MLEID @ symbol, 
		context,
		MLEID /@ Values[argArrays],
		MLEID /@ Values[newGradArrays],
		Lookup[$GradientUpdateCode, Values[gradUpdateMethod], Panic["InvGradUpdate"]], (* 8 *)
		MLEID /@ Values[auxArrays],
		MLEID @ sharedExecutor,
		MLEID @ executorHandle
	];

	(* Outputs are given to us by mxExecutorOutputs *)
	outputs = MXSymbolOutputs[symbol];
	If[FailureQ[outputs], Return[outputs]];

	outputHandles = Table[
		System`Private`SetNoEntry@CreateManagedLibraryExpression["NDArray", NDArray], 
		{Length[outputs]}
	];

	outputArrays = AssociationThread[outputs, outputHandles];
	mxlCall[mxlMXExecutorOutputs, MLEID @ executorHandle, MLEID /@ outputHandles];
	
	(* return executor object *)
	System`Private`SetNoEntry @ MXExecutorData @ Association[
		"Executor" -> executorHandle,
		"Symbol" -> MXSymbolCopy[symbol], (* copy original symbol *)
		"Context" -> context,
		"GradientArrays" -> gradArrays,
		"GradientUpdateMethod" -> gradUpdateMethod,
		"OutputArrays" -> outputArrays,
		"AuxilliaryArrays" -> auxArrays,
		"ArgumentArrays" -> argArrays
	]
]

_mxSymbolBindFast := $Unreachable;


(******************************************************************************)

PackageExport["MXExecutorReshape"]

MXExecutorReshape[MXExecutorData[execdata_], shapes_Association] := Scope[	
	UnpackAssociation[execdata, 
		symbol, context, 
		argumentArrays, gradientArrays, auxilliaryArrays,
		gradientUpdateMethod, executor
	];
	
	inferredShapes = MXSymbolInferShape[symbol, shapes];

	auxilliaryArrays = reshapeAssoc[auxilliaryArrays, inferredShapes["AuxilliaryArrays"]];
	gradientArrays = reshapeAssoc[gradientArrays, inferredShapes["ArgumentArrays"]];
	argumentArrays = reshapeAssoc[argumentArrays, Join[shapes, inferredShapes["ArgumentArrays"]]];

	mxSymbolBindFast[
		symbol, context, 
		argumentArrays, gradientArrays, auxilliaryArrays, 
		gradientUpdateMethod, executor
	]
]

reshapeAssoc[arrays_, shapes_] := 
	Association @ KeyValueMap[
		#1 -> NDArrayReshape[arrays[#1], #2]&, 
		shapes
	];

(******************************************************************************)

PackageExport["MXExecutorReshapeFast"]

mxlDeclare[mxlMXExecutorReshape, {"Integer", "Integer", "String", "IntegerVector", "IntegerVector", "Integer"}, "IntegerVector"]
mxlDeclare[mxlMXExecutorReshapeBindNewArrays, "IntegerVector"]

MXExecutorReshapeFast[exec_MXExecutor, argShapes_Association, context_Integer] := Scope[	
	
	(* put shapes into CSR format *)
	{argKeys, argDims} = KeysValues @ argShapes;
	csrIndices = Accumulate @ Prepend[Map[Length, argDims], 0];
	csrData = Flatten @ argDims;

	(* create output executor *)
	newExecutor = CreateManagedLibraryExpression["MXExecutor", MXExecutor];

	result = mxlCall[
		mxlMXExecutorReshape,
		MLEID @ exec, context,
		mxlPackStringVector @ argKeys, csrIndices, csrData,
		MLEID @ newExecutor
	];

	(* the result contains the count of new arrays to allocate, as well as 4 vectors of indices into
	these new arrays for the arg, grad, aux, and out arrays. all integers are in one flat list along
	with info about how to split them up. *)
	{header, argIndices, gradIndices, auxIndices, outIndices}  = TakeList[result, result[[2 ;; 6]]];
	numNewArrays = result[[1]];

	newArrays = Table[CreateManagedLibraryExpression["NDArray", NDArray], {numNewArrays}];
	mxlMXExecutorReshapeBindNewArrays[MLEID /@ newArrays];
	PrependTo[newArrays, $NullNDArray]; (* index 1 means NullNDArray *)

	{newExecutor, newArrays[[argIndices]], newArrays[[gradIndices]], newArrays[[auxIndices]], newArrays[[outIndices]]}
]

MXExecutorReshapeFast[MXExecutorData[assoc_Association], argShapes_Association] := Scope[

	UnpackAssociation[assoc, 
		executor, symbol, context, 
		gradientArrays, gradientUpdateMethod, outputArrays, 
		auxilliaryArrays, argumentArrays
	];

	{newExecutor, newArgArrays, newGradArrays, newAuxArrays, newOutArrays} = 
		MXExecutorReshapeFast[executor, argShapes, context];

	argKeys = Keys @ argumentArrays;

	MXExecutorData @ Association[
		"Executor" -> newExecutor,
		"Symbol" -> symbol,
		"Context" -> context,
		"GradientArrays" -> AssociationThread[argKeys, newGradArrays],
		"GradientUpdateMethod" -> gradientUpdateMethod,
		"OutputArrays" -> AssociationThread[Keys @ outputArrays, newOutArrays],
		"AuxilliaryArrays" -> AssociationThread[Keys @ auxilliaryArrays, newAuxArrays],
		"ArgumentArrays" -> AssociationThread[argKeys, newArgArrays]
	]
];