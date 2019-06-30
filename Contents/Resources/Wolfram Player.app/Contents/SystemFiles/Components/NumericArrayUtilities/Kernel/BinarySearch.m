(* ::Package:: *)

(**********************************************************************
BinarySearch Functions
**********************************************************************)

Package["NumericArrayUtilities`"]
PackageImport["GeneralUtilities`"]

(*********************************************************************)
PackageExport["LowLevelBinarySearch"]

SetUsage[LowLevelBinarySearch, "
LowLevelBinarySearch[l$, {k1$, k2$,...}] \
searches sorted list l$ for keys k1$, k2$,... and gives the position \
of l$ containing these keys, if they are present in l$. \
Otherwise, if a key k$ is absent in l$, the function returns \
-p$, where k$ falls between the elements of l$ in positions \
p$ and p$+1, or returns 0 if k$ is smaller than the first element of l$.

Options:
- Parallelization: If False, suppress parallelization
"]

DeclareLibraryFunction[mTensorBinarySearch, 
	"binary_search_MTensor",
	{
		{_, 1, "Constant"},
		{_, 1, "Constant"},
		"Boolean"  
	},
	{Integer, 1}
]

DeclareLibraryFunction[mNumericArrayBinarySearch, 
	"binary_search_MNumericArray",
	{
		{"NumericArray", "Constant"}, 
		{"NumericArray", "Constant"},
		"Boolean"
	},
	{Integer, 1}
]

Options[LowLevelBinarySearch] := {
	Parallelization -> True
};

LowLevelBinarySearch::wrtps = "Mixed types of arguments (MTensor and NumericArray).";

LowLevelBinarySearch[list_, keys_, opts:OptionsPattern[]] := Scope[
	If[Length[keys] == 0, Return[{}]];
	runOMP = OptionValue[Parallelization];
	iLowLevelBinarySearch[#1, #2, runOMP] & @@ conformLists[Developer`ToPackedArray[list], Developer`ToPackedArray[keys]]
];

iLowLevelBinarySearch[list_NumericArray, keys_NumericArray, runOMP_] := mNumericArrayBinarySearch[list, keys, runOMP];
iLowLevelBinarySearch[list_ , keys_, runOMP_] := mTensorBinarySearch[list, keys, runOMP];

iLowLevelBinarySearch[list_NumericArray, keys_NumericArray, runOMP_] := 
	mNumericArrayBinarySearch[list, keys, runOMP];

iLowLevelBinarySearch[list_ , keys_, runOMP_] := 
	mTensorBinarySearch[list, keys, runOMP];

conformLists[l1_NumericArray, l2_NumericArray] := {l1, l2};

conformLists[l1_ ? Developer`PackedArrayQ, l2_ ? Developer`PackedArrayQ] := {l1, l2};

conformLists[l1:{__Real}, l2:{__Real}] := 
	{NumericArray[l1, "Real64"], NumericArray[l2, "Real64"]};

conformLists[l1:{__Integer}, l2:{__Integer}] := 
	{NumericArray[l1, "Integer64"], NumericArray[l2, "Integer64"]};

conformLists[l1_, l2_] := (Message[LowLevelBinarySearch::wrtps]; ThrowFailure["wrtps"]);
