(*******************************************************************************

Sorting: fast partial sorting implementations

*******************************************************************************)

Package["NumericArrayUtilities`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)
PackageExport["PartialOrdering"]

SetUsage[PartialOrdering, "
PartialOrdering[list$, k$] calculates the equivalent of Ordering[list$, k$]. k$ can be negative \
or positive, depending on whether to obtain the positions of the k$ largest or \
smallest elements respectively.
PartialOrdering[matrix$, k$] calculates the ordering for each row in the matrix$ in parallel.

$list and $matrix can be either PackedArrays or NumericArrays.
"
]

DeclareLibraryFunction[partialOrderingPacked, "sorting_partial_ordering_packed", 
	{
		{Real, 2, "Constant"},
		Integer
	},
	{Integer, 2}						
]

DeclareLibraryFunction[partialOrderingNumericArray, "sorting_partial_ordering_numericarray", 
	{
		{"NumericArray", "Constant"},
		Integer
	}, 
	{Integer, 2}						
]

PartialOrdering::invk = "`` is an invalid setting for top k ordering."
PartialOrdering[input_List, k_Integer] := Scope[
	(* Deal with matrix or vector *)
	dim = Dimensions@input;
	len = If[Length@dim == 1, First@dim, Last@dim];

	If[k > len || k == 0, ThrowFailure["invdim", k]];
	If[Length[dim] === 1, 
		First@partialOrderingPacked[{input}, k],
		partialOrderingPacked[input, k]
	]
]

PartialOrdering[input_NumericArray, k_Integer] := Scope[
	(* Deal with matrix or vector *)
	dim = Dimensions@input;
	len = If[Length@dim == 1, First@dim, Last@dim];

	If[k > len || k == 0, ThrowFailure["invdim", k]];
	If[Length[dim] === 1, 
		First@partialOrderingNumericArray[input, k],
		partialOrderingNumericArray[input, k]
	]
]