Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["NDSequenceArray"]

SetUsage @ "
NDSequenceArray[data$,lens$] wraps two NDArray[$$] objects, data$ containing \
sequence data, and lens$ containing sequence lengths."

(******************************************************************************)

PackageExport["NDCatenatedArray"]

SetUsage @ "
NDCatenatedArray[{arr$1, arr$2, $$}, {n$1, n$2, $$}] represents a set of arrays that are combined in order to represent a larger array.
* This is used for batch inputs and outputs in batch-parallel training.
* The n$i give the number of elements in the arr$i. This is for speed, to avoid calling in to MXNet."

NDArrayGet[NDCatenatedArray[subs_, _]] := 
	Join @@ Map[NDArrayGet, subs]

NDArrayGetNormal[NDCatenatedArray[subs_, _]] := 
	Join @@ Map[NDArrayGetNormal, subs]

NDArraySetConstant[NDCatenatedArray[subs_, _], value_] := 
	NDArraySetConstant[subs, value]

NDArraySetBatched[nd:NDCatenatedArray[subs_, splits_], data_, failf_] :=
	If[Or[!ListQ[data] && !NumericArrayQ[data], Length[data] =!= Total[splits]],
		failf[data],
		setCat[NDArraySetBatched[#1, #2, failf[data]&]&, nd, data];
	];

NDArraySet[nd_NDCatenatedArray, data_] := 
	setCat[NDArraySet, nd, data]

setCat[setter_, NDCatenatedArray[subs_, splits_], data_] :=
	MapThread[setter, {subs, takeList[data, splits]}]

takeList[nd_NDArray, splits_] := TakeList[NDArrayGet[nd], splits];
takeList[list_List | list_NumericArray, splits_] := TakeList[list, splits]
takeList[other_, splits_] := Table[other, Length[splits]]

NDArrayGetTotalNormal[NDCatenatedArray[subs_, _]] :=
	Total[NDArrayGetTotalNormal /@ subs]

NDArraySetUnbatched[_NDCatenatedArray] := $Unreachable;
NDArrayGetUnbatched[_NDCatenatedArray] := $Unreachable;
NDArrayGetUnbatchedNormal[_NDCatenatedArray] := $Unreachable;
(* ^ because NDReplicaArrays represent distributed inputs and outputs, they
will never have Set/GetUnbatched called on them *)

(******************************************************************************)

PackageExport["NDTotaledArray"]

SetUsage @ "
NDTotaledArray[{arr$1, arr$2, $$}] represents a set of arrays that logically are totalled together.
* This is used for weight gradients in batch-parallel training.
* They can be set with with NDArrayGet.
* ArrayOptimizer[$$] objects can natively handle NDTotaledArray so updates are pushed efficiently to all replicas."

NDArrayGet[NDTotaledArray[subs_]] :=
	NumericArray[Total[NDArrayGetNormal /@ subs], "Real32"] (* for now *)

NDArrayGetNormal[NDTotaledArray[subs_]] :=
	Total[NDArrayGetNormal /@ subs] (* for now *)

NDArraySetUnbatched[_NDTotaledArray, _, _] := $Unreachable;
NDArrayGetUnbatched[_NDTotaledArray] := $Unreachable;
NDArrayGetUnbatchedNormal[_NDTotaledArray] := $Unreachable;
(* ^ because NDTotaledArray represent distributed weight gradients, they
will never have GetUnbatched called on them *)

NDArraySetConstant[NDTotaledArray[subs_], n_] := (
	NDArraySetConstant[subs, 0.];
	If[n != 0., NDArraySetConstant[First[subs], n]]
)

(******************************************************************************)

PackageExport["NDReplicaArray"]

SetUsage @ "
NDReplicaArray[{arr$1, arr$2, $$}] represents a set of arrays that are identical.
* This is used for weights in batch-parallel training.
* They can be Get (one representive will be gotten), or Set (all will be set).
* These should be used with the KVStore API for efficient setting.
* ArrayOptimizer[$$] objects can natively handle NDReplicaArray so updates are pushed efficiently to all replicas."

NDArrayGet[NDReplicaArray[subs_]] :=
	NDArrayGet @ First @ subs

NDArrayGetNormal[NDReplicaArray[subs_]] :=
	NDArrayGetNormal @ First @ subs

NDArrayGetUnbatched[_NDReplicayArray] := $Unreachable;
NDArrayGetUnbatchedNormal[_NDReplicayArray] := $Unreachable;
(* ^ because NDReplicaArrays represent distributed weights, they
will never have GetUnbatched called on them *)

NDArraySet[NDReplicaArray[subs_], nd_] :=
	Scan[NDArraySet[#, nd]&, subs]

NDArraySetBatched[NDReplicaArray[subs_], nd_, failf_] :=
	Scan[NDArraySetBatched[#, nd, failf]&, subs]

NDArraySetUnbatched[NDReplicaArray[subs_], nd_, failf_] :=
	Scan[NDArraySetUnbatched[#, nd, failf]&, subs]

NDArraySetConstant[NDReplicaArray[subs_], val_] :=
	NDArraySetConstant[subs, val]

(******************************************************************************)

PackageExport["NDSparseCountsArray"]

SetUsage @ "
NDSparseCountsArray[NDArray[$$], {nrows$, ncols$}] represents a sparse matrix that has the given dimensions, \
and whose entries are the stored in a (necessarily batched) NDArray[$$].
* The entries are just indices representing +1 counts, and can appear multiple times for higher counts.
* The indices in the entries are 0-indexed, and if both values are -1 the entry is ignored.
* This object is currently only used by the ConfusionMatrix and related measurements in NeuralNetworks.
* The NDArray should have an initial batched dimension, followed by zero or more dimensions, followed by a final dimension of 2.
* NDArrayGetBatchedNormal will return a list of SparseArrays, NDArrayGetTotalNormal will return a single SparseArray."

(* this is a bit weird because the result is not a NumericArray, but that's fine I suppose *)
NDArrayGetBatchedNormal[NDSparseCountsArray[nd_, dims_]] := 
	Map[toCountsMatrix[#, dims]&, NDArrayGetNormal @ nd]

NDArraySetConstant[NDSparseCountsArray[nd_, _], _] :=
	NDArraySetConstant[nd, -1.] (* we must assume the constant is for junk padding *)

toCountsMatrix[data_, dims_] := Block[
	{junkKey = Key @ ConstantArray[0, Length @ dims]},
	(* ^ the junk key, if present, comes from junk padding in sequence arrays *)
	SparseArray[
		Normal @ KeyDrop[junkKey] @ N @ Counts @ Floor @ flattenToVectors[1 + data], 
		dims
	]
]

flattenToVectors[data_] := Flatten[data, ArrayDepth[data]-2];

NDArrayGetPartialTotalNormal[NDSparseCountsArray[pairs_, dims_], excess_, 1] :=
	toCountsMatrix[
		Drop[NDArrayGetNormal @ pairs, excess],
		dims
	];

NDArrayGetTotalNormal[NDSparseCountsArray[pairs_, dims_], 1] :=
	toCountsMatrix[
		NDArrayGetNormal @ pairs, 
		dims
	]

NDArrayGetBatched[_NDSparseCountsArray] := $Unreachable
NDArrayGetUnbatched[_NDSparseCountsArray] := $Unreachable
NDArrayGetUnbatchedNormal[_NDReplicayArray] := $Unreachable