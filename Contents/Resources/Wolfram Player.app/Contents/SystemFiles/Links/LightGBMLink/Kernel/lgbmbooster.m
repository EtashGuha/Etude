Package["LightGBMLink`"]
PackageImport["Developer`"]
PackageImport["GeneralUtilities`"]


(****************************************************************************)
(********************* Load Library Functions *******************************)
(****************************************************************************)	
	
lgbmBoosterCreate = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterCreate", 
	{
		Integer,   				(* booster output handle key *)
		Integer, 				(* existing dataset handle key *)
		"UTF8String" 			(* additional parameters (string) *)
	},
	Integer  (* success/failure (0/-1) *)
];	

lgbmBoosterGetNumClasses = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterGetNumClasses", 
	{
		Integer					(* booster handle key *)
	},
	Integer  (* Number of Classes *)
];	

lgbmBoosterAddValidData = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterAddValidData",
	{
		Integer, 				(* booster handle key *)
		Integer 				(* dataset handle key *)
	},
	Integer  (* success/failure (0/-1) *)
];

lgbmBoosterUpdateOneIter = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterUpdateOneIter",
	{
		Integer 				(* booster handle key *)
	},
	Integer  (* finished/unfinished (1/0) *)
];

lgbmBoosterGetCurrentIteration = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterGetCurrentIteration",
	{
		Integer 				(* booster handle key *)
	},
	Integer  (* Counts *)
];

lgbmBoosterGetEvalCounts = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterGetEvalCounts",
	{
		Integer 				(* booster handle key *)
	},
	Integer  (* Counts *)
];

lgbmBoosterGetEvalNames = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterGetEvalNames",
	{
		Integer 				(* booster handle key *)
	},
	Integer  (* Counts *)
];

lgbmBoosterSaveModelToFile = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterSaveModel",
	{
		Integer, 				(* booster handle key *)
		Integer, 				(* num iterations *)
		"UTF8String" 			(* file name *)
	},
	Integer  (* success/failure (0/-1) *)
];

lgbmBoosterSaveModelToString = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterSaveModelToString",
	{
		Integer, 			(* booster handle key *)
		Integer				(* num iterations *)
	},
	"UTF8String"  				(* string *)
];


lgbmBoosterDumpModelToJson = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterDumpModel",
	{
		Integer, 				(* booster handle key *)
		Integer 				(* num iterations *)
	},
	"UTF8String" 				 (* string in JSON format *)
];

lgbmBoosterLoadModelFromString = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterLoadModelFromString",
	{
		Integer, 				(* booster output handle key *)
		"UTF8String" 			(* model string *)
	},
	Integer  					(* success/failure (0/-1) *)
];

lgbmBoosterResetParameter = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterResetParameter",
	{
		Integer, 				(* booster input handle key *)
		"UTF8String" 			(* parameter string *)
	},
	Integer  					(* success/failure (0/-1) *)
];


(*----------------------------------------------------------------------------*)
PackageExport["LGBMBooster"]

DefineCustomBoxes[LGBMBooster,
	sym_LGBMBooster ? System`Private`NoEntryQ :> LGBMBoosterBox[sym]
];

LGBMBoosterBox[sym:LGBMBooster[id_Integer]] := Scope[
	BoxForm`ArrangeSummaryBox[
		LGBMBooster,
		sym,
		None,
		{makeItem["ID", id]},
		{},
		StandardForm
	]
];
makeItem[name_, value_] := 
	BoxForm`MakeSummaryItem[{Pane[name <> ": ", {20, Automatic}], value}, StandardForm];

(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterObject"]

DefineCustomBoxes[LGBMBoosterObject,
	exec_LGBMBoosterObject ? System`Private`NoEntryQ :> LGBMBoosterObjectBox[exec]
];

LGBMBoosterObjectBox[exec:LGBMBoosterObject[outputdata_Association]] := Scope[
	numDataset = outputdata["DatasetNumber"];
	numClass = outputdata["ClassNumber"];
	BoxForm`ArrangeSummaryBox[
		LGBMBoosterObject,
		exec,
		None,
		{makeBOItem["ClassNumber", numClass], makeBOItem["DatasetNumber", numDataset]},
		{},
		StandardForm
	]
];

makeBOItem[name_, value_] := 
	BoxForm`MakeSummaryItem[{Pane[name <> ": ", {83, Automatic}], value}, StandardForm];	
	

LGBMBoosterObject[outputdata_]["Properties"] := Drop[$boosterprop, 1];
LGBMBoosterObject[outputdata_][query_] := outputdata[query];
LGBMBoosterObject[outputdata_][queries_List] := outputdata[#] &/@ queries;

LGBMBoosterObject /: Normal[LGBMBoosterObject[outputdata_]] := outputdata

(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterCreate"]

SetUsage[LGBMBoosterCreate, "LGBMBoosterCreate[datasetobject$] creates an LGBMBoosterObject from an LGBMDataset datasetobject$. \ 
The following options can be specified:
	'CategoricalIndices' -> {} : the list of indices of categorical features.
	'Objective' -> 'Regression' : the objective relevant for the data. 
	'ClassNumber' -> 1 : the number of classes available for the labels.
	'LossFunction' -> 'MeanSquaredError': the loss function used for training a model on the data.
	'BoostingMethod' -> 'Gradient' : type of gradient boosting used for training. 
	'LeavesNumber' -> 63 : the number of leaves in a tree that is trained on the data.
	'LearningRate' -> 0.1 : the learning rate used in gradient descent. also known as 'shrinakge rate'.
	'MaxBinNumber' -> 255 : the maximum number of discrete bins for each feature.
	'MaxDepth' -> -1 : the maximum depth for each tree that is trained. -1 means no limit. 
	'LeafSize' -> 50 : the minimum number of data points in one leaf of the tree.
	'FeatureFraction' -> 1.0 : the fraction of features randomly selected at each iteration if less than 1. 
	'BaggingFraction' -> 1.0 : the fraction of data samples randomly selected at each iteration if less than 1.
	'BaggingFrequency' -> 0 : frequency for performing bagging, if set to an integer k$, will perform beagging every k$ iterations.
	'MinGainToSplit' -> 0 : minimal gain to perform split.
	'L1Regularization' -> 0 : L1 regularization coefficient. 
	'L2Regularization' -> 0 : L2 regularization coefficient.
	'ThreadNumber' -> $ProcessorCount : the number of threads to use.
"]

General::DatasetTypeError = "`1` should be an LGBMDataset.";
General::BoosterTypeError = "`1` should be an LGBMBoosterObject.";
General::bnexist = "The booster object does not exist.";
LGBMBoosterCreate::mismatchCatInd = "The categorical indices of the dataset and booster should match.";
LGBMBoosterCreate::mismatchMaxBin = "The dataset and booster should have the same number of maximum bins.";
LGBMBoosterCreate::mismatchClassObjective = "The number of classes must be set to 1 for non multi-class training.";
LGBMBoosterCreate::mismatchMultiClass = "The number of classes must be greater than 2 for multi-class training.";

Options[LGBMBoosterCreate] = $BoosterDataOpts;

LGBMBoosterCreate[trainset_LGBMDataset, opts:OptionsPattern[]] := CatchFailure @ Module[
	{
		wlParam, boosterParam, boosterParamStr, dim, booster, 
		numClasses, paramVals1, paramVals2, trainName, 
		attribute, bestIteration
	}
	,
	(* parse parameters into both wl and lgbm forms. The wl form is used in the BoosterObject for 
		informational purposes. *)
	paramVals1 = OptionValue[LGBMBoosterCreate, Take[Keys[$OptionNamesWLToLGBM], 4]];
	paramVals2 = OptionValue[LGBMBoosterCreate, Drop[Keys[$OptionNamesWLToLGBM], 4]];
	wlParam = AssociationThread[
		Keys[$OptionNamesWLToLGBM], 
		Join[paramVals1, paramVals2]
	];
	paramVals1 = If[ContainsAny[Keys @ $OptionValuesWLToLGBM, {#}], 
		$OptionValuesWLToLGBM[#],
		#
	]& /@ paramVals1;
	boosterParam =
		AssociationThread[Values[$OptionNamesWLToLGBM], Join[paramVals1, paramVals2]];	

	boosterParam["verbosity"] = -1;

	If[OptionValue["CategoricalIndices"] =!= {}, 
		(* -1 is to convert to 0-indexed positions *)
		boosterParam["categorical_column"] = (OptionValue["CategoricalIndices"] - 1)
	]; 

	(* Create parameter string *)
	boosterParamStr = AssocToString[boosterParam];

	(* Construct dataset -- various cases *)
	dim = LGBMDatasetGetDimensions[trainset];
	If[Length[dim] =!= 2, ReturnFailed["incorDim"]];
	booster = LGBMBoosterCreateFromDataset[trainset, boosterParamStr];
	numClasses  = LGBMInvoke[lgbmBoosterGetNumClasses, booster];
	trainName = "training";
	attribute = <||>;
	bestIteration = -1;
	
	System`Private`SetNoEntry @ LGBMBoosterObject @ Association[
		"LGBMBooster" -> booster, 
		"LGBMDataset" -> trainset,
		"Parameters" -> wlParam,
		"ValidationSets" -> {},
		"DatasetNumber" -> 1, 
		"ClassNumber"-> numClasses,
		"InitialPredictor" -> None,
		"TrainingName" -> trainName,
		"Attribute" -> attribute,
		"BestIteration" -> bestIteration
	]
];

$boosterprop = { 
	"Properties", "LGBMBooster", "LGBMDataset", "Parameters", "ValidationSets", "DatasetNumber", 
	"ClassNumber", "InitialPredictor", "TrainingName", "Attribute", "BestIteration"
};

(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterCreateFromDataset"]

SetUsage[LGBMBoosterCreateFromDataset, "LGBMBoosterCreateFromDataset[dataset$, param$] creates a Booster from an LGBMDataset dataset$ \ 
with parameters param$ provided as a string. The default value of $param is an empty string.
"]

LGBMBoosterCreateFromDataset[dataset_LGBMDataset, param_String] := CatchFailure @ Module[
	{handle},
	If[Not @ ManagedLibraryExpressionQ[dataset],
		ReturnFailed["bnexist"]
	];
	(* Create Booster handle *)
	handle = System`Private`SetNoEntry @ CreateManagedLibraryExpression["LGBMBooster", LGBMBooster];
	LGBMInvoke[lgbmBoosterCreate, handle, dataset, param];
	handle 
];	

(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterUpdate"]
SetUsage[LGBMBoosterUpdate, "LGBMBoosterUpdate updates an existing LGBMBoosterObject boosterObj$ by one iteration in order to train the booster on the \
data associated with boosterObj$.
"];

LGBMBoosterUpdate[boosterObj_]:= CatchFailure @ Module[
	{booster, result}
	,
	(* Check Types *)
	If[Head[boosterObj] =!= LGBMBoosterObject, ReturnFailed["BoosterTypeError"]];
	booster = boosterObj["LGBMBooster"];
	If[Not @ ManagedLibraryExpressionQ[booster],
		ReturnFailed["bnexist"]
	];
	result = LGBMInvoke[lgbmBoosterUpdateOneIter, booster];
	result
];


(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterSaveModelToFile"]

SetUsage[LGBMBoosterSaveModelToFile, "LGBMBoosterSaveModelToFile[boosterObj$, filename$] saves an existing LGBMBoosterObject boosterObj$ to a file \
filename$.
"];

Options[LGBMBoosterSaveModelToFile] = Options[LGBMBoosterDumpModelToJson] = 
	Options[LGBMBoosterSaveModelToString] = {"NumIterations" -> -1};	

LGBMBoosterSaveModelToFile[boosterObj_LGBMBoosterObject, filename_String, opts:OptionsPattern[]]:= CatchFailure @ Module[
	{numiter, booster},
	numiter = OptionValue["NumIterations"];
	If[numiter <= 0, 
		numiter = boosterObj["BestIteration"];
	];
	booster = boosterObj["LGBMBooster"];
	If[Not @ ManagedLibraryExpressionQ[booster],
		ReturnFailed["bnexist"]
	];
	LGBMInvoke[lgbmBoosterSaveModelToFile, booster, numiter, filename]
];	

(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterSaveModelToString"]

SetUsage[LGBMBoosterSaveModelToString, "LGBMBoosterSaveModelToString[boosterObj$] saves an existing LGBMBoosterObject boosterObj$ to a string.
"];

LGBMBoosterSaveModelToString[boosterObj_LGBMBoosterObject, opts:OptionsPattern[]]:= CatchFailure @ Module[
	{numiter, booster},
	numiter = OptionValue["NumIterations"];
	If[numiter <= 0, 
		numiter = boosterObj["BestIteration"];
	];
	booster = boosterObj["LGBMBooster"];
	If[!ManagedLibraryExpressionQ[booster],
		ReturnFailed["bnexist"]
	];
	LGBMInvoke[lgbmBoosterSaveModelToString, booster, numiter]
];	


(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterDumpModelToJson"]

SetUsage[LGBMBoosterDumpModelToJson, "LGBMBoosterDumpModelToJson[boosterObj$] saves an existing LGBMBoosterObject boosterObj$ to a JSON expression.
"];

LGBMBoosterDumpModelToJson[boosterObj_LGBMBoosterObject, opts:OptionsPattern[]]:= CatchFailure @ Module[
	{numiter, booster},
	numiter = OptionValue["NumIterations"];
	If[numiter <= 0, 
		numiter = boosterObj["BestIteration"];
	];
	booster = boosterObj["LGBMBooster"];
	If[Not@ManagedLibraryExpressionQ[booster],
		ReturnFailed["bnexist"]
	];
	LGBMInvoke[lgbmBoosterDumpModelToJson, booster, numiter]
];

(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterGetNumClasses"]

SetUsage[LGBMBoosterGetNumClasses, "LGBMBoosterGetNumClasses[boosterObj$] gives the number of classes available for the data used to create \ 
the LGBMBoosterObject boosterObj$.
"];

LGBMBoosterGetNumClasses[boosterObj_LGBMBoosterObject]:= CatchFailure @ Module[
	{booster},
	booster = boosterObj["LGBMBooster"];
	If[Not @ ManagedLibraryExpressionQ[booster],
		ReturnFailed["bnexist"]
	];
	LGBMInvoke[lgbmBoosterGetNumClasses, booster]
];	


(*----------------------------------------------------------------------------*)
PackageExport["LGBMBoosterGetCurrentIteration"]
SetUsage[LGBMBoosterGetCurrentIteration, "LGBMBoosterGetCurrentIteration[boosterObj$] gets the current iteration of the LGBMBoosterObject boosterObj$ \ 
while training with a given number of iterations.
"];

LGBMBoosterGetCurrentIteration[boosterObj_LGBMBoosterObject] := CatchFailure @ Module[
	{booster},
	booster = boosterObj["LGBMBooster"];
	If[Not @ ManagedLibraryExpressionQ[booster],
		ReturnFailed["bnexist"]
	];
	LGBMInvoke[lgbmBoosterGetCurrentIteration, booster]
];	

(*----------------------------------------------------------------------------*)
(* At present, it only creates an LGBMBooster from the string, but not an LGBMBoosterObject. It will be good to do that in the future. *)
PackageExport["LGBMBoosterLoadModelFromString"]

SetUsage[LGBMBoosterLoadModelFromString, "LGBMBoosterLoadModelFromString[modelstr$] loads an existing LGBMBoosterObject saved as the string \ 
modelstr$ as an LGBMBooster.
"];

LGBMBoosterLoadModelFromString[modelstr_String]:= CatchFailure @ Module[
	{handle}
	,
	(* Create Booster handle *)
	handle = System`Private`SetNoEntry @ CreateManagedLibraryExpression["LGBMBooster", LGBMBooster];
	LGBMInvoke[lgbmBoosterLoadModelFromString, handle, modelstr];
	handle
];	

(*******************************************************************************************************************)
(******************************Incomplete -- can be done later ****************************************************)
(*******************************************************************************************************************)

(*

PackageExport["LGBMBoosterAddValidationDataset"]	
LGBMBoosterAddValidationDataset::mismatch = "The validation data should have the same predictor as that for the booster.";	

LGBMBoosterAddValidationDataset[boosterObj_LGBMBoosterObject, datasetObj_LGBMDatasetObject, datasetName_String]:= CatchFailure @ Module[
	{datasetObjNew, booster, boosterObjNew}
	,
	If[boosterObj["InitialPredictor"] =!= datasetObj["Predictor"], ReturnFailed["mismatch"]];
	datasetObjNew = SetReference[datasetObj, boosterObj["LGBMDatasetObject"]];
	
	(* Add validation data to booster *)
	(* Check that booster handle exists *)
	booster = boosterObj["LGBMBooster"];	
	If[Not@ManagedLibraryExpressionQ[booster],ReturnFailed["bnexist"]];
	
	
	handle = System`Private`SetNoEntry @ CreateManagedLibraryExpression["LGBMDataset", LGBMDataset];
	LGBMInvoke[lgbmDatasetGetSubset, dataset, handle, rowInd-1, param]; 
	boosterObjNew = LGBMInvoke[lgbmBoosterAddValidationDataset, boosterObj["LGBMBooster"], datasetObj["LGBMDataset"] ];
	
	(* Update properties of BoosterObject *)
	boosterObjNew["ValidationSets"] = Append[boosterObj["ValidationSets"], datasetObj];
	boosterObjNew["DatasetNumber"] = boosterObj["DatasetNumber"] + 1;
	boosterObj["ValidationSetNames"] = Append[boosterObj["ValidationSetNames"], datasetName];
	
	boosterObjNew 	
];

PackageExport["LGBMBoosterAddValidData"]
LGBMBoosterAddValidData[booster_LGBMBooster, validdataset_LGBMDataset]:= CatchFailure @ Module[
	{},
	If[Not@ManagedLibraryExpressionQ[booster],ReturnFailed["bnexist"]];
	If[Not@ManagedLibraryExpressionQ[validdataset],ReturnFailed["dnexist"]];
	LGBMInvoke[lgbmBoosterAddValidData, booster, validdataset]
];	

LGBMBoosterValidationEvaluate[boosterObj_, dataset_, dataName_String, opts:OptionsPattern[]]:= CatchFailure @ Module[
	{dataID, boosterObjNew}
	,
	If[Head[boosterObj] =!= LGBMBoosterObject, ReturnFailed["BoosterTypeError"]];
	If[Head[dataset] =!= LGBMDatasetObject, ReturnFailed["DatasetTypeError"]];
	Which[
		dataset === boosterObj["Trainingset"], 
			dataID = 0;,
		ContainsAny[boosterObj["ValidationSets"], {dataset}],
			dataID = First[Flatten[ Position[ boosterObj["ValidationSets"], dataset] ]];,
		True,	
			dataID = -1;
	];
	boosterObjNew = boosterObj;
	If[dataID == -1, 
		boosterObjNew = LGBMBoosterAddValidationDataset[boosterObj, dataset, dataName];
		dataID = boosterObjNew["NumDataset"] - 1;
	];	
	iBoosterEvaluate[boosterObjNew, dataName, dataID]
];

General::wrongval = "DataID should be smaller than the number of datasets.";

iBoosterEvaluate[boosterObj_LGBMBoosterObject, dataName_String, dataID_Integer]:= CatchFailure @ Module[
	{evalInfo, evalNum, res, evalRes, evalNames, higherBetterEvals, result}
	,
	If[dataID >= boosterObj["NumDataset"], ReturnFailed["wrongval"]];
	evalInfo = LGBMEvalInformation[boosterObj];
	evalNum = evalInfo["EvaluationNumber"];
	evalNames = evalInfo["EvaluationNames"];
	(*higherBetterEvals = evalInfo["HigherBetterEval"];*)
	result = {};
	If[evalNum > 0,
		res = NumericArray[ConstantArray[0, evalNum], "Real32"];
		evalRes = lgbmBoosterGetEvaluation[boosterObj, dataID];
		result = MapThread[{dataName, #1, #2, #3}&, {evalNames, evalRes, higherBetterEvals}];
	];
	result
];	


PackageExport["LGBMBoosterEvalInformation"]
LGBMBoosterEvalInformation::NameError = "The list `1` should be of the same length as number of evaluations.";

LGBMBoosterEvalInformation[obj_LGBMBoosterObject]:= CatchFailure @ Module[
	{evalNum, evalNames, maxNumChar},
	evalNames = {};
	evalNum = LGBMInvoke[lgbmBoosterGetEvalCounts, obj["LGBMBooster"]];
	maxNumChar = 255;
	If[evalNum > 0,
		evalNames = LGBMInvoke[lgbmBoosterGetEvalNames, obj["LGBMBooster"], maxNumChar];
	];
	If[Length[evalNames] =!= evalNum, ReturnFailed["NameError", evalNames]];
	
	<| "EvaluationNumber" -> evalNum, "EvaluationNames" -> evalNames |>
];

*)
