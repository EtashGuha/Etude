Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["NDArraySetPlus"]

SetUsage @ "
NDArraySetPlus[dst$, src$] sets dst$ to dst$ + src$.
NDArraySetPlus[dst$, scalar$] sets dst$ to dst$ + scalar$.";

NDArraySetPlus[dst_NDArray, src_NDArray] := MXOperatorInvoke["_plus", {src, dst}, {dst}, <||>]
NDArraySetPlus[dst_NDArray, r_?NumericQ] := MXOperatorInvoke["_plus_scalar", {dst}, {dst}, <|"scalar" -> N[r]|>]

(******************************************************************************)

PackageExport["NDArraySetTimes"]

SetUsage @ "
NDArraySetTimes[dst$, src$1, src$2] sets dst$ to src$1 * src$2.
NDArraySetTimes[dst$, src$1, scalar$] sets dst$ to dst$ * scalar$."

NDArraySetTimes[dst_NDArray, src1_NDArray, src2_NDArray] := MXOperatorInvoke["_mul", {src1, src2}, {dst}, <||>]
NDArraySetTimes[dst_NDArray, src1_NDArray, r_?NumericQ] := MXOperatorInvoke["_mul_scalar", {src1}, {dst}, <|"scalar" -> N[r]|>]

(******************************************************************************)

PackageExport["NDArrayWaitForAll"]

SetUsage @ "
NDArrayWaitForAll[] wait for all asynchronous operations to complete."

mxlDeclare[mxlNDArrayWaitAll, {}]

NDArrayWaitForAll[] := mxlCall[mxlNDArrayWaitAll]

(******************************************************************************)

PackageExport["NDArrayFree"]

SetUsage @ "
NDArrayFree[NDArray[$$]] frees the memory associated with an array."

mxlDeclare[mxlNDArrayFree, "Integer"]

NDArrayFree[nd_NDArray] := mxlCall[mxlNDArrayFree, MLEID @ nd]

(******************************************************************************)

PackageExport["NDArrayGetPartialTotalNormal"]

NDArrayGetPartialTotalNormal[nd_, excess_, level_] := 
	Total[Drop[NDArrayGetNormal[nd], excess], level];

(******************************************************************************)

PackageExport["NDArrayGetTotalNormal"]

SetUsage @ "
NDArrayGetTotalNormal[NDArray[$$]] returns the result of totalling the array at level 1.
NDArrayGetTotalNormal[NDArray[$$], levelspec$] gives the levels, as in Total.
* The result is a packed array."

NDArrayGetTotalNormal[nd_] := NDArrayGetTotalNormal[nd, 1];

NDArrayGetTotalNormal[nd_, level_] := 
	Total[NDArrayGetNormal[nd], level];

(*

This was tested to be slower:

PackageExport["NDArrayGetTotalNormal"]

$TotalResultsCache = <||>
(* TODO: support axes *)
NDArrayGetTotalNormal[nd_NDArray] := Scope[
	code = NDArrayContextCode[nd];
	type = NDArrayDataTypeCode[nd];
	out = CacheTo[$TotalResultsCache, {code, type}, NDArrayCreateZero[{1}, code, type]];
	MXOperatorInvoke["sum", {nd}, {out}, <|"axis" -> "()"|>];
	First @ ArrayNormal @ out
]

*)

(******************************************************************************)

PackageExport["NDArraySlice"]

SetUsage @ "
NDArraySlice[NDArray[$$], start$, stop$] returns a sub-array using elements start$ throught stop$.
The sub-array is a window onto the original NDArray."

mxlDeclare[mxlNDArraySlice, {"Integer", "Integer", "Integer", "Integer"}]

NDArraySlice[ndarray_, a_Integer, b_Integer] := Scope[
	(* NOTE: we don't want to allocate anything, so we don't call NDArrayCreateEmpty *)
	outputHandle = CreateManagedLibraryExpression["NDArray", NDArray];
	id = MLEID @ ndarray;
	dims = mxlCall[mxlNDArrayDimensions, id]; 
	len = First[dims];
	Which[
		a < 0, a = Max[a + 1 + len, 1], 
		a == 0, Panic["ZeroArraySlice"],
		a > len, a = len;
	];
	Which[
		b < 0, b = Max[b + 1 + len, 1], 
		b == 0, Panic["ZeroArraySlice"],
		b > len, b = len;
	];
	If[b < a, b = a-1];
	mxlCall[mxlNDArraySlice, id, a - 1, b, MLEID @ outputHandle];
	System`Private`SetNoEntry @ outputHandle
]

_NDArraySlice := $Unreachable

(******************************************************************************)

PackageExport["NDArraySliceFast"]

SetUsage @ "
NDArraySliceFast[NDArray[$$], start$, stop$] returns a sub-array using elements start$ throught stop$.
The sub-array is a window onto the original NDArray. Minimal error checking is done."

NDArraySliceFast[ndarray_, a_Integer, b_Integer] := Scope[
	outputHandle = CreateManagedLibraryExpression["NDArray", NDArray];
	mxlCall[mxlNDArraySlice, MLEID @ ndarray, a - 1, b, MLEID @ outputHandle];
	System`Private`SetNoEntry @ outputHandle
]

_NDArraySliceFast := $Unreachable

(******************************************************************************)

PackageExport["NDArraySetSlice"]

SetUsage @ "
NDArraySetSlice[dst$, src$, start$, stop$] sets dst$ to be src$[[start;;stop]]."

NDArraySetSlice[dst_NDArray, src_NDArray, a_Integer, b_Integer] := Scope[
	mxlCall[mxlNDArraySlice, MLEID @ src, a - 1, b, MLEID @ dst];
]

_NDArraySetSlice := $Unreachable

If[$SetupUpValues,
NDArray /: Take[nd_NDArray, {m_, n_} | Span[m_, n_]] := NDArraySlice[nd, Replace[m, All -> 1], Replace[n, All -> -1]];
NDArray /: Part[x_NDArray, range1_ ;; range2_] := NDArraySlice[x, range1, range2];
NDArray /: Part[x_NDArray, {indices__}] := Join@Apply[Sequence, x[[# ;; #]]& /@ {indices}]NDArray /: RandomChoice[x_NDArray, int_] := Part[x, RandomChoice[Range@Length@x, int]];
NDArray /: RandomChoice[x_NDArray] := RandomChoice[x, 1];
];

(******************************************************************************)

PackageExport["NDArrayReshape"]

SetUsage @ "
NDArrayReshape[NDArray[$$], dims$] returns a new NDArray sharing the memory of NDArray[$$], but \
with different dimensions. 
* NDArrayReshape[None, dims$] returns None.
* If the reshape cannot be performed (because the new array is larger), a new array of the same \
type will be allocated on the same device. 
"

mxlDeclare[mxlNDArrayReshape, {"Integer", "IntegerVector", "Integer"}]

NDArrayReshape[ndarray_NDArray, newDims_List] := Scope[
	outputHandle = CreateManagedLibraryExpression["NDArray", NDArray];
	mxlCall[mxlNDArrayReshape, MLEID @ ndarray, newDims, MLEID @ outputHandle]; 
	System`Private`SetNoEntry @ outputHandle
]

NDArrayReshape[NDSequenceArray[data_NDArray, lens_], dims_] := 
	NDSequenceArray[NDArrayReshape[data, dims], NDArrayReshape[lens, Take[dims, 1]]]

NDArrayReshape[None, _] := None

_NDArrayReshape := $Unreachable

(******************************************************************************)

PackageExport["NDArrayEquality"]

SetUsage @ "
NDArrayEquality[assoc1$, assoc2$] checks whether two associations of NDArray's \
are numerically equivalent, up to some default Tolerance (controlled via Options). If \
one association has different keys, False is returned."

Options[NDArrayEquality] = {
	Tolerance -> 0.0001
}

NDArrayEquality[nd1_List, nd2_List, opts:OptionsPattern[]] := Scope[
	dims1 = NDArrayDimensions /@ nd1;
	dims2 = NDArrayDimensions /@ nd2;

	If[dims1 =!= dims2, Return@False];
	(* Third check: numerical similarity *)
	tolerance = OptionValue@Tolerance;
	nd1Normal = NDArrayGet /@ nd1;
	nd2Normal = NDArrayGet /@ nd2;
	diff = Max@Abs@#& /@ (nd1Normal - nd2Normal);
	trues = (# < tolerance)& /@ diff;
	AllTrue[trues, TrueQ]
]

NDArrayEquality[assoc1_Association, assoc2_Association, opts:OptionsPattern[]] := Scope[
	(* Put into canonical order *)
	assoc1Sort = KeySort@assoc1;
	assoc2Sort = KeySort@assoc2;
	(* First check: key equality *) 
	If[Keys@assoc1 =!= Keys@assoc2, Return@False];
	(* otherwise, call list version *)
	NDArrayEquality[Values@assoc1Sort, Values@assoc2Sort, opts]
]

NDArrayEquality[nd1_NDArray, nd2_, opts:OptionsPattern[]] := 
	NDArrayEquality[{nd1}, {nd2}, opts]