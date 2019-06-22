Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]

(******************************************************************************)

mxlDeclare[mxlCopyArrayToNDArray, {"Array", "Integer", "Integer", "Integer", "Boolean"}]

(******************************************************************************)

(* 
NDArraySet is convenient but doesn't take an error function, and does
irrelevant things like map over lists and assocs and allow setting an array to a 
scalar.

NDarraySetBatched and NDArraySetUnbatched are used by the NN executors to handle
batch-1 and batch-n cases efficiently. 
*)

SetRelatedSymbolGroup[
	NDArraySet,
	NDArraySetConstant, 
	NDArraySetBatched,
	NDArraySetUnbatched
]


(******************************************************************************)

DeclarePostloadCode[
General::mxbadset1 = "Could not set `` to ``.";
General::mxbadset2 = "Could not set `` to data of size ``, dimensions `` and head ``.";
]

SetAttributes[NDArraySetMessage, HoldAll]
NDArraySetMessage[f_, to_][data_] := If[ByteCount[data] < 1000,
	Message[f::mxbadset1, to, data],
	Message[f::mxbadset2, to, ByteCount @ data, Dimensions @ data, Head @ data]
]

(******************************************************************************)

PackageExport["NDArraySet"]

SetUsage @ "
NDArraySet[NDArray[$$], scalar$] sets all entires of an array to a scalar value.
NDArraySet[NDArray[$$], array$] sets the contents of an array to another array, which can be numeric array, another NDArray, or a RawArray.
NDArraySet[NDSequenceArray[$$], ragged$] sets a sequence array to the ragged input, saving the data and lengths in the two NDArrays that comprise the sequence.
NDArraySet[{a$1, a$2, $$}, {v$1, v$2, $$}] threads over the lists, setting a$i to v$i.
NDArraySet[<|k$1 -> a$1, $$|>, <|k$1 -> v$1, $$|>] threads over the associations, setting a$i to v$i."

NDArraySet[to_NDArray | to_NDSequenceArray, from_List | from_NumericArray] := 
	NDArraySetBatched[to, from, NDArraySetMessage[NDArraySet, to]];

NDArraySet[to_NDArray, from_Integer | from_Real | from_Rational] := 
	NDArraySetConstant[to, from]

NDArraySet[dst_NDArray, src_NDArray] := NDArrayCopyTo[dst, src]

NDArraySet[to_List, from_List] /; Length[to] === Length[from] := 
	ScanThread[NDArraySet, {to, from}]

NDArraySet[to_Association, from_List] /; Length[to] === Length[from] :=
	ScanThread[NDArraySet, {Values[to], from}]

NDArraySet[to_Association, from_Association] := 
	KeyValueScan[NDArraySet[#2, Lookup[from, #1, Panic["NDArrayAssocMismatch"]]]&, to]

_NDArraySet := $Unreachable

(******************************************************************************)

PackageExport["NDArraySetConstant"]

NNSetUsage @ "
NDArraySetConstant[NDArray[$$],scalar$] sets all entries in an NDArray to a scalar value.
NDArraySetConstant[{arr$1,arr$2,$$},scalar$] sets all entris of all arrays to a scalar value."

mxlDeclare[mxlNDArraySetConstant, {"Integer", "Real"}]

NDArraySetConstant[to_NDArray, from_] := 
	mxlNDArraySetConstant[MLEID @ to, from]

NDArraySetConstant[NDSequenceArray[data_NDArray, _], from_] :=
	NDArraySetConstant[data, from]

NDArraySetConstant[to_List | to_Association, from_] :=
	Scan[NDArraySetConstant[#, from]&, to]

_NDArraySetConstant := $Unreachable

checkN[n_] := Replace[N[n], Except[_Real] :> Panic["NotReal"]]

DeclarePostloadCode[
General::invmxval = "`` is not a scalar, RawArray, NDArray, or tensor of machine numbers.";
]

_NDArraySetConstant := $Unreachable

(******************************************************************************)

PackageExport["NDArraySetBatched"]
PackageExport["NDArraySetUnbatched"]

SetUsage @ "
NDArraySetBatched[NDArray[$$], array$, errorf$] sets an NDArray to a packed array or a list of packed arrays.
NDArraySetBatched[NDArray[$$], NumericArray[$$], errorf$] sets an NDArray to a numeric array.
NDArraySetBatched[NDArray[$$], {array$1, array$2, $$}, errorf$] sets an NDArray to a list of numeric arrays.
* errorf$ is called if there is a dimension or invalid data issue. It is passed the original argument."

NDArraySetBatched[to_, from_] := NDArraySetBatched[to, from, NDArraySetMessage[NDArraySetBatched, to]]
NDArraySetUnbatched[to_, from_] := NDArraySetUnbatched[to, from, NDArraySetMessage[NDArraySetUnbatched, to]]

(* case 1: packed array or list of scalars *) 

fastPackableQ[e_] := Developer`PackedArrayQ[e] || VectorQ[e, NumberQ]

(* subcase: for batched setting, we won't force packing of partially packed inputs, because tryCopyListToNDArray
can handle that *)
NDArraySetBatched[to_NDArray, from_List ? fastPackableQ, failf_] := 
	tryCall[failf, from, mxlCopyArrayToNDArray[from, MLEID @ to, 0, 0, False]]

NDArraySetUnbatched[to_NDArray, from_List ? Developer`PackedArrayQ, failf_] := 
	tryCall[failf, from, mxlCopyArrayToNDArray[from, MLEID @ to, 0, -1, False]]

(* if it wasn't already packed, we try force packing at this point *)
NDArraySetUnbatched[to_NDArray, from_List, failf_] := 
	tryCall[failf, from, mxlCopyArrayToNDArray[Developer`ToPackedArray @ N @ from, MLEID @ to, 0, -1, False]]

(* case 2: numeric array *) 

NDArraySetBatched[to_NDArray, from_NumericArray, failf_] := 
	tryCall[failf, from, mxlCopyArrayToNDArray[from, MLEID @ to, 0, 0, False]]

NDArraySetUnbatched[to_NDArray, from_NumericArray, failf_] := 
	tryCall[failf, from, mxlCopyArrayToNDArray[from, MLEID @ to, 0, -1, False]]

(* case 3: list of arrays *) 

NDArraySetBatched[to_NDArray, from_List, failf_] := 
	tryCopyListToNDArray[failf, from, MLEID @ to, False]

(* only remaining case is scalars *)

NDArraySetUnbatched[to_NDArray, from_ ? NumberQ, failf_] := 
	tryCall[failf, from, mxlCopyArrayToNDArray[{N @ from}, MLEID @ to, 0, 0, False]]

(* or invalid data *)

NDArraySetBatched[to_, from_, failf_] := failf[from]
NDArraySetUnbatched[to_, from_, failf_] := failf[from]

(* or sequence arrays *)

NDArraySetBatched[NDSequenceArray[nd_NDArray, len_NDArray], sequence_, failf_] := 
	tryCopySequencesToNDArray[failf, sequence, MLEID @ nd, MLEID @ len]

NDArraySetUnbatched[NDSequenceArray[nd_NDArray, len_NDArray], sequence_, failf_] :=
	tryCopySingleSequenceToNDArray[failf, sequence, MLEID @ nd, MLEID @ len];

_NDArraySetBatched := $Unreachable
_NDArraySetUnbatched := $Unreachable

(******************************************************************************)

(* 
tryCopyListToNDArray: code to copy a list of independent arrays over. they can be
packed or numeric arrays.
*)

mxlDeclare[mxlNDArrayGetStagingArray, {"Integer", "Boolean"}, "NumericArray"]
mxlDeclare[mxlNDArrayCPUContextQ, "Integer", "Boolean"] 

(* gpu-hosted array will have a rawarray created on CPU as a temporary buffer,
which is copied to element-by-element, and then this buffer is copied over to
GPU *)
tryCopyListToNDArray[failf_, list_, ndid_, allowPartial_] := Block[
	{stage = mxlNDArrayGetStagingArray[ndid, allowPartial]},
	tryCopyListToArray[failf, list, stage, allowPartial];
	mxlCopyArrayToNDArray[stage, ndid, 0, 0, allowPartial]
]

tryCopyListToNDArray[failf_, list_, ndid_ ? mxlNDArrayCPUContextQ, allowPartial_] := Block[
	{srcLen = Length[list]},
	If[srcLen =!= mxlNDArrayLengthAtLevel[ndid, 1], failf[list]];
	Do[
		tryCall[failf, list, mxlCopyArrayToNDArray[list[[i]], ndid, 0, i, allowPartial]],
		{i, srcLen}
	];
]

tryCopyListToArray[failf_, list_, dst_, allowPartial_] := Block[
	{srcLen = Length[list]},
	If[srcLen =!= Length[dst], failf[list]];
	Do[
		tryCall[failf, list, mxlCopyArrayToArray[list[[i]], dst, 0, i, allowPartial]],
		{i, srcLen}
	];
]

_tryCopyListToArray := $Unreachable

(******************************************************************************)

PackageScope["$SequencePaddingValue"]

$SequencePaddingValue = 1.0;

(* 
tryCopySequenceToNDArray: code to copy a list of independent sequence arrays over. 
they can be packed or numeric arrays.
*)

tryCopySequencesToNDArray[failf_, sequence_NumericArray, dataID_, lenID_] := (
	mxlNDArraySetConstant[dataID, $SequencePaddingValue];
	tryCopyListToNDArray[failf, Developer`ArrayNormalToLevel[sequence, 1], dataID, True];
	(* ^ we have to unpack because partial copying on the second dimension isn't supported in C++ land *)
	tryCall[failf, sequence, mxlNDArraySetConstant[lenID, Part[Dimensions @ sequence, 2]]];
)

tryCopySequencesToNDArray[failf_, sequence_List, dataID_, lenID_] := Scope[	
	mxlNDArraySetConstant[dataID, $SequencePaddingValue];
	tryCopyListToNDArray[failf, sequence, dataID, True];
	tryCall[failf, sequence, mxlCopyArrayToNDArray[Length /@ sequence, lenID, 0, 0, False]];
]

tryCopySequencesToNDArray[failf_, data_, _, _] := failf[data]

_tryCopySequencesToNDArray := $Unreachable

(******************************************************************************)

tryCopySingleSequenceToNDArray[failf_, sequence_, dataID_, lenID_] := Block[
	{stage = mxlNDArrayGetStagingArray[dataID, True]},
	tryCall[failf, sequence, mxlCopyArrayToArray[sequence, stage, 0, -1, True]];
	mxlCopyArrayToNDArray[stage, dataID, 0, 0, True];
	tryCall[failf, sequence, mxlNDArraySetConstant[lenID, Length @ sequence]];
]

tryCopySingleSequenceToNDArray[failf_, sequence_, dataID_?mxlNDArrayCPUContextQ, lenID_] := (
	mxlNDArraySetConstant[dataID, $SequencePaddingValue];
	tryCall[failf, sequence, mxlCopyArrayToNDArray[Developer`ToPackedArray @ sequence, dataID, 0, -1, True]];
	tryCall[failf, sequence, mxlNDArraySetConstant[lenID, Length @ sequence]];
)

_tryCopySingleSequenceToNDArray := $Unreachable

(******************************************************************************)

PackageExport["NDArrayAssign"]

SetUsage @ "
NDArrayAssign[NDArray[dst$], NDArray[src$], req$] sets dst$ according to the update \
type req$, where req$ is either an integer in the range 0 to 3, or one of the \
values {None, 'Write', 'InPlace', 'Add'}. "

NDArrayAssign[dst_NDArray, src_NDArray, req_] := Scope[
	Which[
		(req === "Write") || (req === "InPlace"),
			NDArrayCopyTo[dst, src],
		req === "Add",
			NDArraySetPlus[dst, src],
		req === None,
			Nothing,
		True,
			Panic["NDArrayAssign"]
	];
]

NDArrayAssign[dst_NDArray, src_NDArray, req_Integer] := Scope[
	reqString = Lookup[$GradientUpdateCodeReverse, req];
	If[MissingQ[reqString], Panic["NDArrayAssign"]];
	NDArrayAssign[dst, src, reqString]  
]

(******************************************************************************)

PackageExport["NDArrayCopyTo"]

SetUsage @ "
NDArrayCopyTo[dst$, src$] copies src to dst$.
* src$ and dst$ should be NDArrays, or NDArray IDs.
* dst$ can also be an NDReplicaArray, in which case all replicas will be set."

mxlDeclare[mxlNDArrayCopyFromNDArray, {"Integer", "Integer"}] 

Clear[NDArrayCopyTo];
NDArrayCopyTo[dst_NDArray, src_] := NDArrayCopyTo[MLEID @ dst, src];
NDArrayCopyTo[dst_, src_NDArray] := NDArrayCopyTo[dst, MLEID @ src];
NDArrayCopyTo[dst_Integer, src_Integer] := mxlCall[mxlNDArrayCopyFromNDArray, dst, src];
NDArrayCopyTo[NDReplicaArray[dst_List], src_] := Scan[NDArrayCopyTo[#, src]&, dst];
_NDArrayCopyTo := $Unreachable
