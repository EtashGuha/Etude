Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["NDArrayCreate"]

SetUsage @ "
NDArrayCreate[data$] creates an NDArray from data$, where data$ is either a RawArray or a numeric array. 
NDArrayCreate[data$, context$] creates the NDArray in the given context. Context is typically a \
list, eg {'GPU', 0}.
NDArrayCreate[data$, context$, type$] creates an NDArray with numeric type type$. type$ is a \
string (eg 'Real32') or the MXNet integer type code. If type is not specified, the default type is used."

NDArray::notmatrix = "Argument to NDArray was not a numeric matrix."

NDArray[data_List] := 
	If[MachineArrayQ[data],
		NDArrayCreate[data], 
		Message[NDArray::notmatrix]; $Failed
	];

Default[NDArrayCreate] = Automatic;

NDArrayCreate[data_NumericArray, context_.] := 
	NDArrayCreate[data, context, NumericArrayType @ data];

NDArrayCreate[data_, context_., type_.] := Scope[
	handle = NDArrayCreateEmpty[arrayDimensions[data], context, type];
	NDArraySetBatched[handle, data];
	handle
]

(******************************************************************************)

PackageExport["NDArrayCreateEmpty"]

SetUsage @ "
NDArrayCreateEmpty[dims$] creates an uninitialized NDArray of dimensions dims$ on the default device.
NDArrayCreateEmpty[dims$, context$] creates the NDArray in the given context.
NDArrayCreateEmpty[dims$, context$, type$] creates the NDArray of numeric type type$."

mxlDeclare[mxlNDArrayCreateEx, {"Integer", "IntegerVector", "Integer", "Integer"}];

NDArrayCreateEmpty[dims_, context_:Automatic, type_:Automatic] := Scope[
	checkPosDims[dims];
	handle = CreateManagedLibraryExpression["NDArray", NDArray];
	mxlCall[mxlNDArrayCreateEx, 
		MLEID[handle], dims,
		ToContextCode[context], 
		ToDataTypeCode[type]
	];
	System`Private`SetNoEntry @ handle
]

(******************************************************************************)

PackageExport["NDArrayCreateNone"]

SetUsage @ "
NDArrayCreateNone[] creates an NDArray without any memory allocated.
"

mxlDeclare[mxlNDArrayCreateNone, "Integer"];

NDArrayCreateNone[] := Scope[
	handle = CreateManagedLibraryExpression["NDArray", NDArray];
	mxlCall[mxlNDArrayCreateNone, MLEID[handle]];
	System`Private`SetNoEntry @ handle
]

(******************************************************************************)

PackageExport["NDArrayCreateZero"]

SetUsage @ "
NDArrayCreateZero[dims$] creates an NDArray of dimensions dims$ whose entries are equal to zero.
NDArrayCreateZero[dims$, context$] creates the NDArray in the given context.
NDArrayCreateZero[dims$, context$, type$] creates the NDArray with given numeric type."

NDArrayCreateZero[dims_, context_:Automatic, type_:Automatic] := Scope[
	arr = NDArrayCreateEmpty[dims, context, type];
	NDArraySetConstant[arr, 0.];
	arr
]

checkPosDims[dims_] := 
	If[!VectorQ[dims, Internal`PositiveMachineIntegerQ] || dims === {}, Panic["InvalidDimensions"]]

(******************************************************************************)

PackageExport["NDArrayCloneShape"]

mxlDeclare[mxlNDArrayCloneShape, {"Integer", "Integer"}]

SetAttributes[NDArrayCloneShape, Listable]

NDArrayCloneShape[NDSequenceArray[nd_, len_], scalar_] := 
	NDSequenceArray[NDArrayCloneShape[nd, scalar], len];

NDArrayCloneShape[nd_NDArray, scalar_] := Scope[
	arr = CreateManagedLibraryExpression["NDArray", NDArray];
	mxlCall[mxlNDArrayCloneShape, MLEID[nd], MLEID[arr]];
	NDArraySetConstant[arr, scalar];
	System`Private`SetNoEntry @ arr
]


