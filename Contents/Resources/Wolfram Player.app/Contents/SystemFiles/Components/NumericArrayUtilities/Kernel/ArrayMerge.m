(* ::Package:: *)

(**********************************************************************
Low Level Array Merge Function
**********************************************************************)

Package["NumericArrayUtilities`"]
PackageImport["GeneralUtilities`"]

(*********************************************************************)
PackageExport["LowLevelArrayMerge"]
PackageExport["LowLevelKPosArrayMerge"]

SetUsage[LowLevelArrayMerge, "
LowLevelArrayMerge[m1$, m2$] \
merges the two sorted arrays m1$ and m2$ returning the resulting \
array.
"]

SetUsage[LowLevelKPosArrayMerge, "
LowLevelKPosArrayMerge[m1$, m2$] \
merges the two sorted arrays m1$ and m2$ returning the index positions \
of m1$ and m2$ elements in the merged array.
"]

(* merging arrays *)

DeclareLibraryFunction[
        mTensorArrayMerge, 
        "array_merge_MTensor",
	{
	   {_, 1, "Constant"},
           {_, 1, "Constant"},
           Integer
	},
	{_, 1}
]

DeclareLibraryFunction[
        mNumericArrayMerge, 
	"array_merge_MNumericArray",
	{
	   {"NumericArray", "Constant"}, 
           {"NumericArray", "Constant"},
           Integer
	},
	{"NumericArray"}
]

LowLevelArrayMerge::wrtps =
"Mixed types of arguments (MTensor and NumericArray).";  

Options[LowLevelArrayMerge] = {"KeepDuplicates" -> False};

LowLevelArrayMerge[input1_, input2_, opts:OptionsPattern[]] :=
CatchFailure @ Module[
     {dupl}
     ,
     If [Length[input2] == 0, Return[input1]];
     If [Length[input1] == 0, Return[input2]];
     If [(Head[input1] === NumericArray && Head[input2] =!= NumericArray)
         || (Head[input1] =!= NumericArray && Head[input2] === NumericArray)
         ,
         ReturnFailed["wrtps"]
     ];
     
     dupl = OptionValue["KeepDuplicates"];
     If [Head[input1] === NumericArray && Head[input2] === NumericArray,
	 mNumericArrayMerge[input1, input2, Boole@dupl]
	,
	 mTensorArrayMerge[input1, input2, Boole@dupl]
     ]
]

(* Key positions of merged arrays *)

DeclareLibraryFunction[
        mTensorKPosArrayMerge, 
        "kpos_merge_MTensor",
	{
	   {_, 1, "Constant"},
           {_, 1, "Constant"},
           Integer
	},
	{Integer, 1}
]

DeclareLibraryFunction[
        mNumericKPosArrayMerge, 
	"kpos_merge_MNumericArray",
	{
	   {"NumericArray", "Constant"}, 
           {"NumericArray", "Constant"},
           Integer
	},
	{Integer, 1}
]

LowLevelKPosArrayMerge::wrtps =
"Mixed types of arguments (MTensor and NumericArray).";  

Options[LowLevelKPosArrayMerge] = {"KeepDuplicates" -> False};

LowLevelKPosArrayMerge[input1_, input2_, opts:OptionsPattern[]] :=
CatchFailure @ Module[
     {dupl, pos, len1 = Length[input1], len2 = Length[input2]}
     ,
     If [len2 == 0, Return[Range[len1]]];
     If [len1 == 0, Return[Range[len2]]];
     If [(Head[input1] === NumericArray && Head[input2] =!= NumericArray)
         || (Head[input1] =!= NumericArray && Head[input2] === NumericArray)
         ,
         ReturnFailed["wrtps"]
     ];
     
     dupl = OptionValue["KeepDuplicates"];
     If [Head[input1] === NumericArray && Head[input2] === NumericArray,
	 pos = mNumericKPosArrayMerge[input1, input2, Boole@dupl]
	,
	 pos = mTensorKPosArrayMerge[input1, input2, Boole@dupl]
     ];
     {pos[[;;len1]], pos[[len1+1;;]]}
]

