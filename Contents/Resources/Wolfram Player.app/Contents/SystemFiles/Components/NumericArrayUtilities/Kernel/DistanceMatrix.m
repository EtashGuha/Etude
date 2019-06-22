(* ::Package:: *)

(**********************************************************************
Distance Matrix Functions
**********************************************************************)

Package["NumericArrayUtilities`"]
PackageImport["GeneralUtilities`"]

(*********************************************************************)
PackageExport["LowLevelDistanceMatrix"]
PackageExport["LowLevelDistanceMatrixMethod"]
PackageExport["LowLevelDistanceMatrixKey"]

SetUsage[LowLevelDistanceMatrix, "
LowLevelDistanceMatrix[(u1$, u2$,...}, {v1$, v2$,...}, meth$] \
gives the matrix of distances between each pair of numerical elements \
ui$, vj$ using the method string meth$. \
LowLevelDistanceMatrix[(u1$, u2$,...}, meth$] \
gives the matrix of distances between each pair of numerical elements \
ui$, uj$ using the method string meth$. 

Options:
- Parallelization: If False, suppress parallelization
"]
SetUsage[LowLevelDistanceMatrixMethod, "
LowLevelDistanceMatrixMethods[meth$] returns the position (integer) \
corresponding to the string meth$ of the method used by LowLevelDistanceMatrix..
"]
SetUsage[LowLevelDistanceMatrixKey, "
LowLevelDistanceMatrixKey[i$] returns a string being the method name \
corresponding to the internal method i$ which is also the position of \
the method in an association.
"]

DeclareLibraryFunction[mTensorDistanceMatrix2Arg, 
	"distance_matrix_MTensor2Arg",
	{
		{Real, 2, "Constant"},
		{Real, 2, "Constant"},
                Integer,
                "Boolean"  
	},
	{Real, 2}
]

DeclareLibraryFunction[mTensorDistanceMatrix1Arg, 
	"distance_matrix_MTensor1Arg",
	{
		{Real, 2, "Constant"},
                Integer,
                "Boolean" 
	},
	{Real, 2}
]

DeclareLibraryFunction[mNumericArrayDistanceMatrix2Arg, 
	"distance_matrix_MNumericArray2Arg",
	{
		{"NumericArray", "Constant"}, 
		{"NumericArray", "Constant"}, 
                Integer,
                "Boolean"  
	},
	{"NumericArray"}
]

DeclareLibraryFunction[mNumericArrayDistanceMatrix1Arg, 
	"distance_matrix_MNumericArray1Arg",
	{
		{"NumericArray", "Constant"}, 
		Integer,
		"Boolean"  
	},
	{"NumericArray"}
]

$extractLLDMMethod = <|
    "BrayCurtisDistance" -> 1,
	"CanberraDistance" -> 2,
	"EuclideanDistance" -> 3,
	"SquaredEuclideanDistance" -> 4,
	"NormalizedSquaredEuclideanDistance" -> 5,
	"ManhattanDistance" -> 6,
	"ChessboardDistance" -> 7,
	"CosineDistance" -> 8,
	"CorrelationDistance" -> 9,
	"JaccardDissimilarity" -> 10,
	"MatchingDissimilarity" -> 11,
	"RussellRaoDissimilarity" -> 12,
	"SokalSneathDissimilarity" -> 13,
	"RogersTanimotoDissimilarity" -> 14,
	"DiceDissimilarity" -> 15,
	"YuleDissimilarity" -> 16
|>;

LowLevelDistanceMatrixMethod[meth_String] := $extractLLDMMethod[meth];
LowLevelDistanceMatrixKey[i_Integer] := Keys[$extractLLDMMethod][[i]];

Options[LowLevelDistanceMatrix] = {
  Parallelization -> True
}
  
LowLevelDistanceMatrix::notmeth = "`` is an invalid method.";
LowLevelDistanceMatrix[input1_, input2_, meth_String,
                       opts:OptionsPattern[]] :=
        CatchFailure @ Module[
                {runOMP, imeth},
        runOMP = OptionValue[Parallelization];
	imeth = $extractLLDMMethod[meth];
	If[MissingQ[imeth], ReturnFailed["notmeth", meth]];
	If [Head[input1] === NumericArray && Head[input2] === NumericArray,
		mNumericArrayDistanceMatrix2Arg[input1, input2, imeth, runOMP]
	,
		mTensorDistanceMatrix2Arg[input1, input2, imeth, runOMP]
	]
]

LowLevelDistanceMatrix[input1_, meth_String, opts:OptionsPattern[]] :=
        CatchFailure @ Module[
                {runOMP, imeth},
        runOMP = OptionValue[Parallelization];
	imeth = $extractLLDMMethod[meth];
	If[MissingQ[imeth], ReturnFailed["notmeth", meth]];
	If [Head[input1] === NumericArray,
		mNumericArrayDistanceMatrix1Arg[input1, imeth, runOMP]
	,
		mTensorDistanceMatrix1Arg[input1, imeth, runOMP]
	]
]
