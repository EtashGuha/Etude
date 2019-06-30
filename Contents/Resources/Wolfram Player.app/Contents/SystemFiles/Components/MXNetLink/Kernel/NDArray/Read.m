Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]

(******************************************************************************)

PackageScope["mxlReadNDArray$numeric"]
PackageScope["mxlReadNDArray$packed"]

(* these are the same DLL function, which dispatches on the third argument to
return either a packed or numeric array *)
mxlDeclare[mxlReadNDArray$packed, {"Integer", "Integer", "Boolean"}, "Tensor"]
mxlDeclare[mxlReadNDArray$numeric, {"Integer", "Integer", "Boolean"}, "NumericArray"]

(******************************************************************************)

PackageExport["ArrayNormal"]

SetUsage @ "
ArrayNormal[NDArray[$$]] returns a packed array of the contents of the NDArray.
ArrayNormal[NumericArray[$$]] unpacks the NumericArray.
ArrayNormal[array$] returns array$."

ArrayNormal[nd_NDArray] := NDArrayGetNormal @ nd
ArrayNormal[array_List] := array
ArrayNormal[array_NumericArray] := Normal @ array
_ArrayNormal := $Unreachable

(******************************************************************************)

PackageExport["NDArrayGet"]
PackageExport["NDArrayGetNormal"]

SetUsage @ "
NDArrayGet[NDArray[$$]] returns a NumericArray of the contents of the NDArray."

SetUsage @ "
NDArrayGetNormal[NDArray[$$]] returns a packed array of the contents."

NDArrayGet[nd_NDArray] := 		CheckNotAbnormal @ mxlReadNDArray$numeric[MLEID @ nd, 0, True]
NDArrayGetNormal[nd_NDArray] := CheckNotAbnormal @ mxlReadNDArray$packed[MLEID @ nd, 0, False]

NDArrayGet[nd_NDSequenceArray] := NDArrayGetBatched[nd];
NDArrayGetNormal[nd_NDSequenceArray] := NDArrayGetBatchedNormal[nd];

_NDArrayGet := $Unreachable
_NDArrayGetNormal := $Unreachable

If[$SetupUpValues,
NDSequenceArray /: Normal[nd_NDSequenceArray] := NDArrayGet[nd];
NDArray /: Normal[nd_NDArray] := NDArrayGet[nd];
];

(******************************************************************************)

SetRelatedSymbolGroup[
	NDArrayGetUnbatched, NDArrayGetUnbatchedNormal,
	NDArrayGetBatched,   NDArrayGetBatchedNormal
]

PackageExport["NDArrayGetUnbatched"]
PackageExport["NDArrayGetUnbatchedNormal"]

SetUsage @ "
NDArrayGetUnbatched[NDArray[$$]] returns a NumericArray of the contents of the NDArray, flattening \
off the initial singleton dimension."

SetUsage @ "
NDArrayGetUnbatchedNormal[NDArray[$$]] returns a packed array of the contents of the NDArray, flattening \
off the initial singleton dimension."

NDArrayGetUnbatched[nd_NDArray] := 
	CheckNotAbnormal @ getUnbatchedNumeric @ nd;

NDArrayGetUnbatchedNormal[nd_NDArray] := 
	CheckNotAbnormal @ getUnbatchedPacked @ nd;

getUnbatchedNumeric[nd_] := mxlReadNDArray$numeric[MLEID @ nd, -1, True]
getUnbatchedPacked[nd_] := mxlReadNDArray$packed[MLEID @ nd, -1, False]

NDArrayGetUnbatched[NDSequenceArray[data_NDArray, len_NDArray]] :=       
	Take[getUnbatchedNumeric @ data, checkPositive @ getUnbatchedPacked @ len]

NDArrayGetUnbatchedNormal[NDSequenceArray[data_NDArray, len_NDArray]] := 
	Take[getUnbatchedPacked @ data, checkPositive @ getUnbatchedPacked @ len]

_NDArrayGetUnbatched := $Unreachable
_NDArrayGetUnbatchedNormal := $Unreachable

checkPositive[_] := ThrowFailure["netseqlen"];
checkPositive[lens_List | lens_Real] := If[Min[lens] > 0, lens, ThrowFailure["netseqlen"]];

(******************************************************************************)

PackageExport["NDArrayGetBatched"]
PackageExport["NDArrayGetBatchedNormal"]

SetUsage @ "
NDArrayGetBatched[NDArray[$$]] returns a list of NumericArrays, or a list of scalars.
NDArrayGetBatched[NDSequenceArray[$$]] returns a list of NumericArrays."

SetUsage @ "
NDArrayGetBatchedNormal[NDArray[$$]] returns a packed array.
NDArrayGetBatchedNormal[NDSequenceArray[$$]] returns a list of packed arrays."

mxlDeclare[mxlReadNDArrayBatchedBegin, {"Integer", "Boolean"}, "Integer"]

mxlDeclare[mxlReadNDArrayBatchedYield$numeric, {}, "NumericArray"]
mxlDeclare[mxlReadNDArrayBatchedYield$packed, {}, "Tensor"]

(* batch of scalars cannot become a batch of scalar numeric arrays, so we return as packed here *)
NDArrayGetBatched[nd_NDArray] := 
	CheckNotAbnormal @ getBatchedNumeric @ MLEID @ nd

NDArrayGetBatchedNormal[nd_NDArray] := 
	CheckNotAbnormal @ getBatchedPacked @ MLEID @ nd

mxlDeclare[mxlNDArrayVectorQ, "Integer", "Boolean"]
getBatchedNumeric[id_ ? mxlNDArrayVectorQ] := mxlReadNDArray$packed[id, 0, False];
getBatchedNumeric[id_] := Table[mxlReadNDArrayBatchedYield$numeric[], {mxlReadNDArrayBatchedBegin[id, True]}]
getBatchedPacked[id_ ? mxlNDArrayVectorQ] := mxlReadNDArray$packed[id, 0, False];
getBatchedPacked[id_] := Table[mxlReadNDArrayBatchedYield$packed[], {mxlReadNDArrayBatchedBegin[id, False]}]

NDArrayGetBatched[NDSequenceArray[data_NDArray, len_NDArray]] := 
	getBatchSequence[getBatchedNumeric, data, len]

NDArrayGetBatchedNormal[NDSequenceArray[data_NDArray, len_NDArray]] :=
	getBatchSequence[getBatchedPacked, data, len]

getBatchSequence[readfunc_, data_, len_] := 
	MapThread[Take /* CheckNotAbnormal, {
		readfunc @ MLEID @ data, 
		checkPositive @ mxlReadNDArray$packed[MLEID @ len, 0, False]
	}]

_NDArrayGetBatched := $Unreachable
_NDArrayGetBatchedNormal := $Unreachable