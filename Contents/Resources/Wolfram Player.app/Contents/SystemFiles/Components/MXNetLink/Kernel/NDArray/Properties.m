Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["NDArrayDataTypeCode"]

SetUsage @ "
NDArrayDataTypeCode[NDArray[$$]] returns the data type of array as an integer."

mxlDeclare[mxlNDArrayGetDType, "Integer", "Integer"]

NDArrayDataTypeCode[ndarray_] := 
	mxlNDArrayGetDType[MLEID @ ndarray]

(******************************************************************************)

PackageExport["NDArrayDataType"]

SetUsage @ "
NDArrayDataType[NDArray[$$]] returns the data type of array as a string name."

NDArrayDataType[ndarray_] := 
	$DataTypeCodeReverse @ NDArrayDataTypeCode[ndarray]

(******************************************************************************)

PackageExport["NDArrayByteCount"]

SetUsage @ "
NDArrayByteCount[NDArray[$$]] returns the ByteCount of an array."

NDArrayByteCount[None] := 0;

NDArrayByteCount[nd_NDArray] := Times[
	Lookup[$DataTypeBytes, NDArrayDataType @ nd],
	NDArraySize @ nd
];

(******************************************************************************)

PackageExport["NDArrayLength"]

PackageScope["mxlNDArrayLengthAtLevel"]

SetUsage @ "
NDArrayLength[NDArray[$$]] returns length of the array."

mxlDeclare[mxlNDArrayLengthAtLevel, {"Integer", "Integer"}, "Integer"]

NDArrayLength[ndarray_NDArray, level_:1] := 
	mxlNDArrayLengthAtLevel[MLEID @ ndarray, level];

NDArrayLength[NDSequenceArray[data_, _]] := 
	NDArrayLength[data, 1]

If[$SetupUpValues,
NDArray /: Length[nd_NDArray] := NDArrayLength[nd];
]

(******************************************************************************)

PackageExport["NDArrayRank"]

SetUsage @ "
NDArrayRank[NDArray[$$]] returns the rank of the array."

mxlDeclare[mxlNDArrayRank, "Integer", "Integer"]

NDArrayRank[ndarray_NDArray] := 
	mxlNDArrayRank[MLEID @ ndarray]

(******************************************************************************)

PackageExport["NDArrayDimensions"]

SetUsage @ "
NDArrayDimensions[NDArray[$$]] returns the dimensions of the array."

PackageScope["mxlNDArrayDimensions"]
mxlDeclare[mxlNDArrayDimensions, "Integer", "IntegerVector"];

NDArrayDimensions[ndarray_NDArray] := 
	mxlNDArrayDimensions[MLEID @ ndarray]

NDArrayDimensions[NDSequenceArray[data_, _]] := 
	NDArrayDimensions[data];

If[$SetupUpValues,
NDArray /: Dimensions[nd_NDArray] := NDArrayDimensions @ nd;
NDSequenceArray /: Dimensions[nd_NDSequenceArray] := NDArrayDimensions @ nd;
];

(******************************************************************************)

PackageExport["NDArrayContextCode"]

SetUsage @ "
NDArrayContextCode[NDArray[$$]] returns an integer that encodes a \
device type and id."

mxlDeclare[mxlNDArrayGetContext, "Integer", "Integer"]

NDArrayContextCode[ndarray_] := 
	mxlNDArrayGetContext[MLEID @ ndarray]

(******************************************************************************)

PackageExport["NDArrayContext"]

SetUsage @ "
NDArrayContext[NDArray[$$]] returns tuple of the form {'device', id}."

NDArrayContext[ndarray_] := 
	FromContextCode @ NDArrayContextCode[ndarray]

(******************************************************************************)

PackageExport["NDArrayExistsQ"]

SetUsage @ "
NDArrayExistsQ[NDArray[$$]] returns True if the array handle is valid.
NDArrayExistsQ[id$] takes a raw ID rather than an NDArray handle.
* The only invalid NDArray[$$] handles are ones used to indicate null arrays."

mxlDeclare[mxlNDArrayExistsQ, "Integer", "Boolean"]

NDArrayExistsQ[1] = False (* <- that's $NullNDArray *)
NDArrayExistsQ[id_Integer] := mxlNDArrayExistsQ[id]
NDArrayExistsQ[nd_NDArray] := NDArrayExistsQ[MLEID[nd]]
NDArrayExistsQ[_] := False

(******************************************************************************)

PackageExport["NDArrayStatistics"]

SetAttributes[NDArrayStatistics, Listable]

mxlDeclare[mxlNDArrayStatistics, "Integer", "RealVector"]

NDArrayStatistics[nd_NDArray] := mxlNDArrayStatistics[MLEID @ nd];
_NDArrayStatistics := $Unreachable;

(******************************************************************************)

PackageExport["NDArraySummary"]

NDArraySummary[e_NDArray] := Scope[
	If[!ManagedLibraryExpressionQ[e],
		Return @ Missing["InvalidHandle", First @ e];
	];
	dims = NDArrayDimensions[e];
	If[!VectorQ[dims, IntegerQ], Return @ Missing["CorruptHandle", First @ e]];
	size = Times @@ dims;
	If[size < 10*^8,
		{min, max, mean, tot, sd, size} = NDArrayStatistics[e],
		min = max = mean = sd = Missing["TooLarge"];
	];
	depth = Length[dims];
	If[size < 128 && depth < 4 && Max[dims] < 16,
		data = NDArrayGetNormal[e];
		table = NumericArrayForm[data],
		table = None
	];
	If[4 < size < 16384 && Length[dims] == 2,
		data = NDArrayGetNormal[e];
		size = Max[Min[Floor[512 / First[dims]], Floor[256 / Last[dims]], 8], 2];
		plot = MatrixPlot[data, PixelConstrained -> size, FrameTicks -> None, Frame -> None],
		plot = None
	];
	Association[
		"Depth" -> depth, "Dimensions" -> dims,
		"Size" -> size, "Min" -> min, 
		"Max" -> max, "Mean" -> mean, 
		"StandardDeviation" -> If[size > 1, sd, Indeterminate],
		If[plot === None, {}, "Plot" -> plot],
		If[table === None, {}, "Table" -> table]
	]
]

(******************************************************************************)

PackageExport["NDArraySummaryGrid"]

NDArraySummaryGrid[e_NDArray] := Scope[
	assoc = NDArraySummary[e];
	If[MissingQ[assoc], Return[assoc]];
	grid = Grid[
		KeyValueMap[
			If[MissingQ[#2], Nothing, {Replace[#1, "dimensions" -> "dims"], #2}]&,
			assoc
		],
		Alignment -> Left,
		ItemStyle -> {{{}, {FontFamily -> "Courier"}}}
	];
	Framed[grid, FrameStyle -> None]
]

(******************************************************************************)

PackageExport["NDArrayDebugString"]

NDArrayDebugString[nd_NDArray] := 
	minmaxstr[nd]

NDArrayDebugString[assoc_Association] :=
	StringRiffle[KeyValueMap[ToString[#1] <> "=" <> minmaxstr[#2]&, assoc], "  "]

NDArrayDebugString[list_List] :=
	StringRiffle[Map[minmaxstr, list], "  "]

ndMinMax[nd_NDArray] := Take[NDArrayStatistics[nd], 2];
ndMinMax[e_] := MinMax @ NDArrayGetNormal[e];

minmaxstr[nd_] := Scope[
	{min, max} = fstr /@ ndMinMax[nd];
	ndid[nd] <> ":" <> If[min =!= max, min <> "~" <> max, min]
]

ndid[nd_NDArray] := TextString @ MLEID[nd]
ndid[NDTotaledArray[subs_]] := "T[" <> Riffle[ndid /@ subs, ","] <> "]"
ndid[NDReplicaArray[subs_]] := "R[" <> Riffle[ndid /@ subs, ","] <> "]"
ndid[NDSequenceArray[seq_, len_]] := "S[" <> ndid[seq] <> ":" <> ndid[len] <> "]"
ndid[NDCatenatedArray[subs_, _]] := "C[" <> Riffle[ndid /@ subs, ","] <> "]"
ndid[rest_] := ToString[rest]

fstr[0] := "0"
fstr[r_] := Scope[
	{m,e} = MantissaExponent[r];
	TextString[Round[m]] <> "e" <> TextString[e]
]
