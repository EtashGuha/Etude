Package["DAALLink`"]

PackageImport["GeneralUtilities`"]

PackageScope["numericTableMLE"] (* numeric Table ManagedLibraryExpression *)

(*----------------------------------------------------------------------------*)
(****** Load Library Functions ******)

naToHomogenousDense := naToHomogenousDense = LibraryFunctionLoad[$DAALLinkLib, "WL_NAToHomogenousDense",
	{
		Integer,
		{"NumericArray", "Constant"}
	},
	"Void"
]

numericTableDims := numericTableDims = LibraryFunctionLoad[$DAALLinkLib, "WL_NumericTableDims",
	{
		Integer
	},
	{Integer, 1}
]

numericTableToNA := numericTableToNA =  LibraryFunctionLoad[$DAALLinkLib, "WL_NumericTableToNA",
	{
		Integer
	},
	"NumericArray"
]

numericTableSetCat := numericTableSetCat = LibraryFunctionLoad[$DAALLinkLib, "WL_SetCategoricalIndices",
	{
		Integer,
		{Integer, 1}
	},
	Null
]

(*----------------------------------------------------------------------------*)
PackageExport["DAALNumericTable"]

(* This is a utility function defined in GeneralUtilities, which makes a nicely
formatted display box *)
DefineCustomBoxes[DAALNumericTable, 
	e:DAALNumericTable[mle_, array_, type_, ___] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		DAALNumericTable, e, None, 
		{
			BoxForm`SummaryItem[{"ID: ", getMLEID[mle]}],
			BoxForm`SummaryItem[{"Dimensions: ", safeLibraryInvoke[numericTableDims, getMLEID[mle]]}],
			BoxForm`SummaryItem[{"Type: ", type}]
		},
		{},
		StandardForm
	]
]];

getMLEID[DAALNumericTable[mle_, ___]] := ManagedLibraryExpressionID[mle];

DAALNumericTable /: Dimensions[x_DAALNumericTable] := DAALNumericTableDimensions[x]

(*----------------------------------------------------------------------------*)
PackageExport["DAALNumericTableCreate"]

SetUsage[DAALNumericTableCreate,
"DAALNumericTableCreate[matrix$] creates a Real32 DAALNumericTable[$$] object from a matrix matrix$.\
in$ can be any object convertible to a Real32 NumericArray. If matrix$ is a Real32 NumericArray, \
the memory is shared with the resulting DAALNumericTable[$$], and no copy is made. \
DAALNumericTable[$$] also keeps a reference to matrix$, preventing deallocation for the lifetime of \
DAALNumericTable[$$]. All other input types are copied.
The following options are available:
| NominalVariables | {} | specify that certain features should be interpreted as \
nominal, where the nominal values are encoded as integers starting from 0. List of feature positions.  | 
"
]

Options[DAALNumericTableCreate] = {
	NominalVariables -> {}
};

DAALNumericTableCreate::invdim = 
	"A matrix was expected as input, but a rank-`` tensor was given.";

DAALNumericTableCreate[in_, opts:OptionsPattern[]] := CatchFailure @ Module[
	{
		mle, dims = Dimensions[in],
		cat, data
	},
	data = toNumericArray[in];
	If[FailureQ[data], Return[$Failed]];
	If[Length[dims] =!= 2, ThrowFailure["invdim", Length[dims]]];
	(* nominals are zero indexed in DAAL, but 1-indexed in WL *)
	cat = OptionValue[NominalVariables] - 1;

	mle = CreateManagedLibraryExpression["DAALNumericTable", numericTableMLE];
	safeLibraryInvoke[naToHomogenousDense, getMLEID[mle], data];
	safeLibraryInvoke[numericTableSetCat, getMLEID[mle], cat];
	
	System`Private`SetNoEntry @ DAALNumericTable[mle, data, "Real32"]
]

toNumericArray[in_ /; NumericArrayQ[in]] := If[
	NumericArrayType[in] === "Real32",
		in, 
		NumericArray[in, "Real32"]
]

toNumericArray[in_] := NumericArray[in, "Real32"]

(*----------------------------------------------------------------------------*)
PackageExport["DAALNumericTableToNumericArray"]

SetUsage[DAALNumericTableToNumericArray,
"DAALNumericTableToNumericArray[DAALNumericTable[$$]] converts a DAALNumericTable[$$] to \
a Real32 NumericArray.
"
]

DAALNumericTableToNumericArray[x_DAALNumericTable] := CatchFailure @ 
	safeLibraryInvoke[numericTableToNA, getMLEID[x]];

(*----------------------------------------------------------------------------*)
PackageExport["DAALNumericTableDimensions"]

DAALNumericTableDimensions[x_DAALNumericTable] := CatchFailure @ 
	safeLibraryInvoke[numericTableDims, getMLEID[x]];
