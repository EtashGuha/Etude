(*******************************************************************************

LGBM: Dataset-level functions

*******************************************************************************)

Package["LightGBMLink`"]

PackageImport["Developer`"]
PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
(* Load Library Functions *)

lgbmDatasetGetNumData = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetGetNumData",
	{
		Integer (* Dataset handle key *)
	},
	Integer		(* num of data *)
];

lgbmDatasetGetNumFeature = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetGetNumFeature",
	{
		Integer	(* Dataset handle key *)
	},
	Integer		(* num of data *)
];

lgbmDatasetCreateFromFile = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetCreateFromFile", 
	{
		Integer,   		(* Dataset output handle key *)
		"UTF8String", 	(* file name *)
		"UTF8String", 	(* additional parameters (string) *)
		Integer  		(* 0 if not used; or the handle key of a dataset to
                 			align bin mapper *)
	}, 
	Integer	(* success/failure (0/-1) *)
];

lgbmDatasetCreateFromMatrix = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetCreateFromMat", 
	{
		Integer,   				(* Dataset output handle key *)
		{"NumericArray", "Shared"}, (* input matrix *)
		"UTF8String", 			(* additional parameters (string) *)
		Integer  				(* 0 if not used; or the handle key of a dataset 
									to align bin mapper *)
	},
	Integer  (* success/failure (0/-1) *)
];

lgbmDatasetCreateFromSparse = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetCreateFromCSR",
	{
		Integer, 		(* Dataset output handle key *)      
		{LibraryDataType[SparseArray, Real], "Constant"}, (* sparse array: *)
		"UTF8String", 	(* additional parameters (string) *)
		Integer  		(* 0 if not used; or the handle key of a dataset to
                     		align bin mapper  *)
	}, 
	Integer  (* success/failure (0/-1) *)
];

lgbmDatasetGetSubset = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetGetSubset",
	{
		Integer, 		(* existing dataset handle key *)
		Integer, 		(* subset handle key *)
		{Integer, _}, 	(* row indices used in subset *)
		"UTF8String" 	(* additional parameters (string) *)
	}, 
	Integer  (* success/failure (0/-1) *)
];

lgbmDatasetSaveBinary = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetSaveBinary",
	{
		Integer, 		(* existing dataset handle key *)
		"UTF8String" 	(* binary file name *)
	}, 
	Integer  (* success/failure (0/-1) *)
];

lgbmDatasetSetField = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetSetField",
	{
		Integer, 					(* existing dataset handle key *)
		"UTF8String", 				(* field name *)
		{"NumericArray", "Constant"} 	(* field data *)
	},
	Integer  (* success/failure (0/-1) *)
];

lgbmDatasetGetField = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_DatasetGetField",
	{
		Integer,        (* existing dataset handle key *)
		"UTF8String" 	(* field name *)
	}, 
	"NumericArray" (* field data *)
];

lgbmDatasetSetFeatureNames = LibraryFunctionLoad[$lightGBMLinkLib,
        "WL_DatasetSetFeatureNames",
		{
			Integer, 		(* existing dataset handle key *)
			"UTF8String" 	(* feature names *)
		}, 
		Integer  (* success/failure (0/-1) *)
];


(*----------------------------------------------------------------------------*)
PackageExport["LGBMDataset"]

DefineCustomBoxes[LGBMDataset,
	sym_LGBMDataset ? System`Private`NoEntryQ :> LGBMDatasetBox[sym]
];

LGBMDatasetBox[sym:LGBMDataset[id_Integer]] := Scope[
	numdata = LGBMInvoke[lgbmDatasetGetNumData, sym];
	numfeature = LGBMInvoke[lgbmDatasetGetNumFeature, sym];
	BoxForm`ArrangeSummaryBox[
		LGBMDataset,
		sym,
		None,
		{makeItem["ndata", numdata], 
		 makeItem["nfeature", numfeature]},
		{},
		StandardForm
	]
];

makeItem[name_, value_] := BoxForm`MakeSummaryItem[
	{Pane[name <> ": ", {48, Automatic}], value}, StandardForm
];


(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetCreate"]
PackageScope["iLGBMDatasetCreate"]

SetUsage[LGBMDatasetCreate, 
"LGBMDatasetCreate[data$] creates an LGBMDatasetObject from a packed \
array, a Real32/Real64 NumericArray, or a sparse array data$ of dimensions \
{batch$, featureSize$} with corresponding labels label$. 
LGBMDatasetCreate[data$, labels$] creates a dataset with labels. The following options can be specified:
	'Weight' -> {} : the weights of samples in the data.
	'CategoricalIndices' -> {} : the list of indices of categorical features.
	'MaxBinNumber' -> 255 : the maximum number of discrete bins for each feature.
"]

Options[LGBMDatasetCreate] = {
	"Weight"-> {},
	"MaxBinNumber" -> 255,
	"CategoricalIndices" -> {}
};

LGBMDatasetCreate[data__, opts:OptionsPattern[]] := CatchFailure @ Module[
	{
		weight, catInd, lgbmParam, 
		lgbmParamStr, dataArray, lgbmDat, maxbin
	},
	{weight, maxbin, catInd} = 
		OptionValue[LGBMDatasetCreate, {"Weight", "MaxBinNumber", "CategoricalIndices"}];
	iLGBMDatasetCreate[data, weight, maxbin, catInd]
]

(* internal version which throws errors *)	
iLGBMDatasetCreate::invdim = "Data must be a matrix.";
	
iLGBMDatasetCreate[data_, weight_, maxbin_, catInd_] := CatchFailure @ Module[
	{
		lgbmParam, lgbmParamStr, dataArray, lgbmDat
	},
	(* Create parameter string *)
	lgbmParam = <|"max_bin" -> maxbin,
                      "verbosity" -> -1,
                      "min_data" -> 1,
                      "min_data_in_bin" -> 1 |>;
	If[catInd =!= {}, 
		(* -1 is to convert to 0-indexed positions *)
		lgbmParam["categorical_column"] = (catInd - 1)
	]; 
	lgbmParamStr = AssocToString[lgbmParam];

	(* handle different data types *)
	lgbmDat = Switch[data,
		_SparseArray,
			LGBMDatasetCreateFromSparse[data, lgbmParamStr],
		_,
			dataArray = ConformDataToNumArr32[data];
			LGBMDatasetCreateFromMatrix[dataArray, lgbmParamStr]
	];
	iLGBMDatasetSetField[lgbmDat, "weight", weight];
	lgbmDat
];

iLGBMDatasetCreate[data_, labels_, weight_, maxbin_, catInd_] := Module[
	{res},
	res = iLGBMDatasetCreate[data, weight, maxbin, catInd];
	iLGBMDatasetSetField[res, "label", labels];
	res
]

(*----------------------------------------------------------------------------*)
ConformDataToNumArr32::incorType = "Cannot convert data to NumericArray of type Real32.";
PackageScope["ConformDataToNumArr32"]

ConformDataToNumArr32[data_NumericArray] := If[NumericArrayType[data] === "Real32",
	data,
	NumericArray[data, "Real32"]
]

ConformDataToNumArr32[data_List] :=  Module[
	{rdata}
	,
	rdata = Quiet @ NumericArray[data, "Real32"];
	If[!NumericArrayQ[rdata], ThrowFailure[ConformDataToNumArr32::incorType]];
	rdata
];

ConformDataToNumArr32::invType = "Data type must be list or NumericArray."
ConformDataToNumArr32[_] := ThrowFailure[ConformDataToNumArr32::invType]

(* Convert Association of Parameters to an appropriate String *)
PackageScope["AssocToString"]
$rules = {"{" | "}" | " " -> "", "->" -> "="};
AssocToString[params_] := 
	StringRiffle[StringReplace[ToString[#], $rules] & /@ Normal[params]];


(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetCreateFromSparse"]

SetUsage[LGBMDatasetCreateFromSparse, 
"LGBMDatasetCreateFromSparse[data$, param$] creates an LGBMDataset from a \
sparse matrix data$ with parameters param$ provided as a string. \
The default value of param$ is an empty string.
"];
	
LGBMDatasetCreateFromSparse[data_, param_String] := CatchFailure @ Module[
	{handle},
	If[Length[Dimensions[data]] =!= 2, ReturnFailed["incorDim"]];
	(* Create sparse matrix handle *)
	handle = CreateManagedLibraryExpression["LGBMDataset", LGBMDataset];
	LGBMInvoke[lgbmDatasetCreateFromSparse, handle, data, param, 0];
	System`Private`SetNoEntry @ handle
];

(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetCreateFromMatrix"]

SetUsage[LGBMDatasetCreateFromMatrix, 
"LGBMDatasetCreateFromMatrix[data$, param$] creates an LGBMDataset from a \
dense Real32 NumericArray matrix data$ with parameters param$ provided as a string. \
The default value of param$ is an empty string.
"];

LGBMDatasetCreateFromMatrix[data_, param_String] := CatchFailure @ Module[
	{handle},
	If[Length[Dimensions[data]] =!= 2, ReturnFailed["incorDim"]];
	(* Create Dataset handle *)
	handle = System`Private`SetNoEntry @ CreateManagedLibraryExpression["LGBMDataset", LGBMDataset];
	LGBMInvoke[lgbmDatasetCreateFromMatrix, handle, data, param, 0];
	handle
];


(****************************************************************************)	
(***************Getting Information about the Dataset ***********************)
(****************************************************************************)

PackageExport["LGBMDatasetGetDimensions"]

SetUsage[LGBMDatasetGetDimensions, 
" LGBMDatasetGetDimensions[dataset$] returns the number of samples and number of \
features of LGBMDataset dataset$.
"];

LGBMDatasetGetDimensions[obj_LGBMDataset] := CatchFailure @ Module[
	{ndata, nftr},
	ndata = LGBMInvoke[lgbmDatasetGetNumData, obj];
	nftr = LGBMInvoke[lgbmDatasetGetNumFeature, obj];
	{ndata, nftr} 
];

(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetGetLabel"]

SetUsage[LGBMDatasetGetLabel, "
	LGBMDatasetGetLabel[dataset$] returns the labels associated with LGBMDataset dataset$.
"];

LGBMDatasetGetLabel[obj_LGBMDataset]:= CatchFailure @ LGBMDatasetGetField[obj, "label"];

(****************************************************************************)
PackageExport["LGBMDatasetGetWeight"]

SetUsage[LGBMDatasetGetWeight, 
"LGBMDatasetGetWeight[dataset$] returns the weights associated with the LGBMDataset dataset$.
"];

LGBMDatasetGetWeight[obj_LGBMDataset]:= CatchFailure @ LGBMDatasetGetField[obj, "weight"];	

(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetSetField"]

SetUsage[LGBMDatasetSetField, 
"LGBMDatasetSetField[dataset$, fieldname$, fielddata$] sets fielddata$ \
to a content in an LGBMDataset dataset$ corresponding to field name fieldname$.
"]

iLGBMDatasetSetField::invna32 = "Can't convert input to a Real32 NumericArray.";

LGBMDatasetSetField[dataset_LGBMDataset, fieldname_String, fielddata_] := 
	CatchFailure @ iLGBMDatasetSetField[dataset, fieldname, fielddata]

iLGBMDatasetSetField[dataset_LGBMDataset, fieldname_String, fielddata_NumericArray] := 
	LGBMInvoke[lgbmDatasetSetField, dataset, fieldname, fielddata]

iLGBMDatasetSetField[dataset_LGBMDataset, fieldname_String, fielddata_List] := 
	iLGBMDatasetSetField[dataset, fieldname, ConformDataToNumArr32[fielddata]]
(* For the empty list case: do nothing? or return error? *)
iLGBMDatasetSetField[dataset_LGBMDataset, fieldname_String, {}] := Null

iLGBMDatasetSetField::invtype = "Invalid type for field data.";
iLGBMDatasetSetField[dataset_LGBMDataset, fieldname_String, _] := 
	ThrowFailure[iLGBMDatasetSetField::invtype]

(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetGetField"]
PackageScope["iLGBMDatasetGetField"]

SetUsage[LGBMDatasetGetField, 
"LGBMDatasetGetField[dataset$, fieldname$] get an info vector from an \
LGBMDataset dataset$ corresponding to the field name fieldname$.
"];

LGBMDatasetGetField::dnexist = "The dataset does not exist.";

LGBMDatasetGetField[dataset_, fieldname_String] := 
	CatchFailure @ iLGBMDatasetGetField[dataset, fieldname]

iLGBMDatasetGetField[dataset_, fieldname_String] :=  Module [
	{},
	(* Check that dataset exists *)
	If[Not @ ManagedLibraryExpressionQ[dataset],
		ReturnFailed[LGBMDatasetGetField::dnexist]
	];
	LGBMInvoke[lgbmDatasetGetField, dataset, fieldname] 
]



(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetGetSubset"]

SetUsage[LGBMDatasetGetSubset, 
"LGBMDatasetGetSubset[dataset$, rowInd$, param$] creates a subset of existing \
LGBMDataset dataset$ using row indices rowInd$ \
with parameters param$. The default value of param$ is an empty string.
"];

LGBMDatasetGetSubset::dnexist = "The dataset does not exist.";

LGBMDatasetGetSubset[dataset_, rowInd_, param_String:""] := CatchFailure @ Module [
	{handle} ,
	(* Check that dataset exists *)
	If[Not @ ManagedLibraryExpressionQ[dataset],
		ReturnFailed[LGBMDatasetGetSubset::dnexist]
	];
	handle = CreateManagedLibraryExpression["LGBMDataset", LGBMDataset];
	LGBMInvoke[lgbmDatasetGetSubset, dataset, handle, rowInd-1, param]; 
	System`Private`SetNoEntry @ handle 
];

(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetCreateFromFile"]

SetUsage[LGBMDatasetCreateFromFile, 
"LGBMDatasetCreateFromFile[fileName$, param$] creates an LGBMDataset directly \
from file fileName$ with parameters param$ (with an empty string as the dafault value).
"];

LGBMDatasetCreateFromFile::nffil = "File not found.";

LGBMDatasetCreateFromFile[fileName_String, param_String:""] := CatchFailure @ Module[
	{handle},
	(* Check that file exists *)
	If[Not @ FileExistsQ[fileName],
		ReturnFailed[LGBMDatasetCreateFromFile::nffil]
	];
	(* Create matrix handle *)
	handle = System`Private`SetNoEntry @ 
		CreateManagedLibraryExpression["LGBMDataset", LGBMDataset];
	LGBMInvoke[lgbmDatasetCreateFromFile, handle, fileName, param, 0];
	handle
]

(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetSaveBinary"]

SetUsage[LGBMDatasetSaveBinary, "
	LGBMDatasetSaveBinary[dataset$, filename$] saves dataset$ \
	to the binary file filename$.
"]

LGBMDatasetSaveBinary::dnexist = "The dataset does not exist.";

LGBMDatasetSaveBinary[dataset_, filename_String] := CatchFailure @ Module [
	{}
	,
	(* Check that dataset exists *)
	If[Not@ManagedLibraryExpressionQ[dataset],
		ReturnFailed[LGBMDatasetSaveBinary::dnexist]];
		LGBMInvoke[lgbmDatasetSaveBinary,dataset, filename] 
]

(*----------------------------------------------------------------------------*)
PackageExport["LGBMDatasetSetFeatureNames"]

SetUsage[LGBMDatasetSetFeatureNames, 
"LGBMDatasetSetFeatureNames[dataset$, fnames$] sets feature names fnames$ to a dataset$.
"]       

LGBMDatasetSetFeatureNames::dnexist = "The dataset does not exist.";
                                    
(* set feature names to a dataset *)
LGBMDatasetSetFeatureNames[dataset_, fnames_String] := CatchFailure @ Module [
	{},
	(* Check that dataset exists *)
	If[Not @ ManagedLibraryExpressionQ[dataset],
		ReturnFailed[LGBMDatasetSetFeatureNames::dnexist]
	];
	LGBMInvoke[lgbmDatasetSetFeatureNames, dataset, fnames]
]

