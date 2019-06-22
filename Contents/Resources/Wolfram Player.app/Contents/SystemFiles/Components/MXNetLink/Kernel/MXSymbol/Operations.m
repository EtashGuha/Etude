Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXSymbolGroup"]

SetUsage @ "
MXSymbolGroup[{MXSymbol[$$], $$}] creates a new symbol by grouping together the input symbols.
Used to create multi-output networks."

mxlDeclare[mxlMXSymbolCreateGroup, {"IntegerVector", "Integer"}]

MXSymbolGroup[symbols_:{__MXSymbol}] := CatchFailure @ Scope[
	outputSymbol = CreateManagedLibraryExpression["MXSymbol", MXSymbol];
	mxlCall[mxlMXSymbolCreateGroup, MLEID /@ symbols, MLEID @ outputSymbol];
	System`Private`SetNoEntry @ outputSymbol
] 

(******************************************************************************)

PackageExport["MXSymbolVariable"]

SetUsage @ "
MXSymbolVariable[name$] createss a variable symbol with the given name."

mxlDeclare[mxlMXSymbolCreateVariable, {"Integer", "String"}]

MXSymbolVariable[name_String] := CatchFailure @ Scope[
	symbolHandle = CreateManagedLibraryExpression["MXSymbol", MXSymbol];
	mxlCall[mxlMXSymbolCreateVariable, MLEID @ symbolHandle, name];
	System`Private`SetNoEntry @ symbolHandle
] 

(******************************************************************************)

PackageExport["MXSymbolCopy"]

SetUsage @ "
MXSymbolCopy[MXSymbol[$$]] returns a copy of the MXSymbol.
MXSymbolCopy[inSymbol$, outSymbol$] copies the MXSymbol inSymbol$ to outSymbol$."

mxlDeclare[mxlMXSymbolCopy, {"Integer", "Integer"}]

MXSymbolCopy[symbol_MXSymbol, outSymbol_MXSymbol] := CatchFailure[
	mxlCall[mxlMXSymbolCopy, MLEID @ symbol, MLEID @ outSymbol];	
	outputSymbol
]

MXSymbolCopy[symbol_MXSymbol] := CatchFailure @ Scope[
	outputSymbol = CreateManagedLibraryExpression["MXSymbol", MXSymbol];
	MXSymbolCopy[symbol, outputSymbol];
	System`Private`SetNoEntry @ outputSymbol
]

(******************************************************************************)

PackageExport["MXSymbolGradient"]

SetUsage @ "
MXSymbolGradient[MXSymbol[$$], params$] returns the autodiff gradient symbol with \
respect to params$, a list of parameter names. 
* this function can only be used if symbol is a loss function."

mxlDeclare[mxlMXSymbolGrad, {"Integer", "Integer", "String"}]

MXSymbolGradient[symbol_MXSymbol, params_List] := CatchFailure @ Scope[
	gradHandle = CreateManagedLibraryExpression["MXSymbol", MXSymbol];
	mxlCall[mxlMXSymbolGrad, 
		MLEID @ symbol, MLEID @ gradHandle, 
		mxlPackStringVector @ params];		
	System`Private`SetNoEntry @ gradHandle
]

(******************************************************************************)

PackageExport["MXSymbolGetInternals"]

SetUsage @ "
MXSymbolGetInternals[MXSymbol[$$]] returns a new symbol whose outputs are the internal states of the MXSymbol."

mxlDeclare[mxlMXSymbolGetInternals, {"Integer", "Integer"}];

MXSymbolGetInternals[symbol_MXSymbol] := CatchFailure @ Scope[
	internalSymbol = CreateManagedLibraryExpression["MXSymbol", MXSymbol];
	mxlCall[mxlMXSymbolGetInternals, MLEID @ symbol, MLEID @ internalSymbol];
	System`Private`SetNoEntry @ internalSymbol
]

(******************************************************************************)

PackageExport["MXSymbolInferShape"]

SetUsage @ "
MXSymbolInferShape[MXSymbol[$$], <|'name$1' -> dim1$, $$|>, returnall$, allowpartial$] infers 
dimensions of all arrays in a symbol, and returns the inferred dimensions as one or more 
associations.
* returnall$ specifies whether to return an association of associations with keys \
'ArgumentArrays', 'AuxilliaryArrays', and 'OutputArrays', or just the argument arrays as \
a single association (which is faster).
* allowpartial$ specifies whether incomplete shape inference is allowed.
"

mxlDeclare[mxlMXSymbolInferShape, {"Integer", "Integer", "IntegerVector", "IntegerVector", "Boolean", "Boolean"}, "IntegerVector"]

MXSymbolInferShape[symbol_MXSymbol, argShapes_Association, returnAll_:True, allowPartial_:False] := Scope[
		
	(* get all relevant symbol details *)
	argKeys = MXSymbolArguments @ symbol;
	numArgs = Length @ argKeys;

	(* put into CSR format *)
	dims = Lookup[argShapes, Prepend[argKeys, ""], {}];
	csrIndices = Accumulate @ Map[Length, dims];
	csrData = Flatten @ dims;

	(* get results from mxnet shape inference *) 
	result = mxlCall[
		mxlMXSymbolInferShape, MLEID @ symbol, numArgs,
		csrIndices, csrData, returnAll, allowPartial
	];

	(* reconstruct the dimensions returned from mxnet, which is a single flat vector of integers.
	first is the ranks, and then all the flattened dimensions *)
	If[!returnAll,
		argDims = splitDims @ TakeDrop[result, numArgs];
		AssociationThread[argKeys, argDims]
	,
		auxKeys = MXSymbolAuxilliaryStates[symbol]; numAux = Length[auxKeys];
		outKeys = MXSymbolOutputs[symbol]; numOut = Length[outKeys];
		allDims = splitDims @ TakeDrop[result, numArgs + numAux + numOut];
		{argDims, auxDims, outDims} = TakeList[allDims, {numArgs, numAux, numOut}];
		Association[
			"ArgumentArrays" -> AssociationThread[argKeys, argDims],
			"AuxilliaryArrays" -> AssociationThread[auxKeys, auxDims],
			"OutputArrays" -> AssociationThread[outKeys, outDims]
		]
	]
]

splitDims[{ranks_, dims_}] := TakeList[dims, ranks]

(******************************************************************************)

PackageExport["MXSymbolCompose"]

SetUsage @ "MXSymbolCompose[MXSymbol[$$], name$, args$] returns a new symbol with name name$ that \
represents the composition of the given symbol with args$, which should be an association mapping \
argument names to existing MXSymbols."

mxlDeclare[mxlMXSymbolCompose, {"Integer", "String", "IntegerVector", "String"}];

MXSymbolCompose[inputSym_MXSymbol, name_String, arguments___] := CatchFailure @ Scope[
	
	args = Discard[arguments, RuleQ];
	kwargs = Association @ Select[arguments, RuleQ];
	
	(* error check *)
	If[Length[args] > 0 && Length[kwargs] > 0, 
		ThrowFailure["Can only accept input Symbols either as positional or keyword arguments, not both"]
	];
	
	numArgs = Length[args] + Length[kwargs];
	
	If[Length[kwargs] > 0, 
		symbols = MLEID /@ Values[kwargs];
		names = mxlPackStringVector @ Keys @ kwargs
		,
		symbols = MLEID /@ args;
		names = "";
	];

	outSymbol = MXSymbolCopy[inputSym];
	If[FailureQ @ outSymbol, Return @ outSymbol];

	mxlCall[mxlMXSymbolCompose, MLEID @ outSymbol, name, symbols, names];
	outSymbol
]

(******************************************************************************)

PackageExport["MXSymbolOutputPart"]

SetUsage @ "MXSymbolOutputPart[MXSymbol[$$], index$] returns a new symbol with \
a single output specified by index$. index$ is either a string, in which case it \
is an output name of MXSymbol[$$], or it is an integer specifying the nth output. 
"

mxlDeclare[mxlMXSymbolGetOutput, {"Integer", "Integer", "Integer"}];

MXSymbolOutputPart[symbol_MXSymbol, index_] := Scope[
	indexInt = If[StringQ[index],
		outNames = MXSymbolOutputs @ symbol;
		indexInt = Position[outNames, index][[1, 1]],
		index
	];
	indexInt -= 1; (* use zero indexing *)
	outputSymbol = CreateManagedLibraryExpression["MXSymbol", MXSymbol];
	mxlCall[mxlMXSymbolGetOutput, MLEID @ symbol, indexInt, MLEID @ outputSymbol];
	System`Private`SetNoEntry @ outputSymbol
]