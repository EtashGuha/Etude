Package["LightGBMLink`"]
PackageImport["Developer`"]
PackageImport["GeneralUtilities`"]

(****************************************************************************)
(********************* Load Library Functions *******************************)
(****************************************************************************)	
	
lgbmBoosterPredictForSparse = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterPredictForCSR", 
	{
		Integer,   				(* booster input handle key *)
		{LibraryDataType[SparseArray, Real], "Shared"}, (* sparse array *)
		Integer, 				(* predict Type *)
		Integer, 				(* iteration used for prediction *)
                "UTF8String"                            (* aux parameters *)
	},
	"NumericArray"					(* predictions and numrows *)
];	

lgbmBoosterPredictForMatrix = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterPredictForMat", 
	{
		Integer,   				(* booster input handle key *)
		{"NumericArray", "Shared"}, (* data matrix in the form of numeric array *)
		Integer, 				(* predict Type *)
		Integer,				(* iteration used for prediction *)
                "UTF8String"                            (* aux parameters *)
	},
	"NumericArray" 
];	

lgbmBoosterCalcNumPredict = LibraryFunctionLoad[$lightGBMLinkLib, 
	"WL_BoosterCalcNumPredict", 
	{
		Integer,   				(* booster input handle key *)
		Integer, 				(* number of rows of the data matrix *)
		Integer, 				(* predict Type *)
		Integer 				(* iteration used for prediction *)
	},
	Integer 					(* number of predictions *)
];	

(*****************************************************************************************************)
(*******************************Options and Parameters*************************************************)
PackageScope["$OptionNamesWLToLGBM"]
$OptionNamesWLToLGBM = <|
	"Objective" -> "objective",
	"ClassNumber" -> "num_class",
	"LossFunction" -> "metric",
	"BoostingMethod" -> "boosting_type",  
	"LeavesNumber" -> "num_leaves", 
	"LearningRate" -> "learning_rate", 
	"MaxBinNumber" -> "max_bin",
	"MaxDepth" -> "max_depth",
	"LeafSize" -> "min_data_per_leaf",
	"FeatureFraction" -> "feature_fraction", 
	"BaggingFraction" -> "bagging_fraction", 
	"BaggingFrequency" -> "bagging_freq",
	"MinGainToSplit" -> "min_gain_to_split",
	"L1Regularization" -> "lambda_l1",
	"L2Regularization" -> "lambda_l2",
	"ThreadNumber" -> "num_threads"
|>; 

PackageScope["$OptionValuesWLToLGBM"]
$OptionValuesWLToLGBM = <|
	"Gradient" -> "gbdt", 
	"DART" -> "dart", 
	"GradientOneSideSampling" -> "goss",
	"MeanSquaredError" -> "l2", 
	"MeanAbsoluteError" -> "l1", 
	"HuberLoss" -> "huber", 
	"PoissonRegression" -> "poisson",
	"AUC" -> "auc",
	"BinaryLogLoss" -> "binary_logloss", 
	"BinaryError" -> "binary_error", 
	"MultiLogLoss" -> "multi_logloss", 
	"MultiError" -> "multi_error",
	"Regression" -> "regression",
	"Binary" -> "binary",
	"MultiClass" -> "multiclass"
|>;

(************************************************************************
		Main WL Interface Function -- LGBMBoosterTrain
*************************************************************************)
ClearAll[$BoosterDataOpts]
PackageScope["$BoosterDataOpts"]
$BoosterDataOpts = {
	"CategoricalIndices" -> {}, 
	"Objective" -> "Regression",
	"ClassNumber" -> 1,
	"LossFunction" -> "MeanSquaredError",
	"BoostingMethod" -> "Gradient",
	"LeavesNumber" -> 63,
	"LearningRate" -> 0.1,
	"MaxBinNumber" -> 255,
	"MaxDepth" -> -1,
	"FeatureFraction" -> 1.0,
	"LeafSize" -> 50,
	"BaggingFraction" -> 1.0,
	"BaggingFrequency" -> 0,
	"MinGainToSplit" -> 0,
	"L1Regularization" -> 0,
	"L2Regularization" -> 0, 
	"ThreadNumber" -> $ProcessorCount
};

PackageScope["$ValidationOpts"]
$ValidationOpts = {
	ValidationSet -> {},
	"PrintEvaluationMetric" -> False
};	 

PackageExport["LGBMBoosterTrain"]

SetUsage[LGBMBoosterTrain, "LGBMBoosterTrain[datasetObj$] trains an LGBMBoosterObject on the traing data associated with LGBMDataset datasetobj$. \ 
The following options can be specified:
	'MaxIterations' -> 100 : the number of boosting rounds. 
	'CategoricalIndices' -> {} : the list of indices of categorical features.
	'Objective' -> 'Regression' : the objective relevant for the data. 
	'ClassNumber' -> 1 : the number of classes available for the labels.
	'LossFunction' -> : the loss function used for training a model on the data.
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
	'ValidationSet' -> {} : the data to use for validation.
	'PrintEvaluationMetric' -> False : whether to print the evaluation metrics for validation data.  
"]

LGBMBoosterTrain::CatIndError = "Categorical indices `1` do not match that of the LGBMDataset. ";
LGBMBoosterTrain::notsaved = "FinalModel could not be saved properly.";

Options[LGBMBoosterTrain] = Join[{MaxIterations-> 100}, $BoosterDataOpts, $ValidationOpts];	

LGBMBoosterTrain[trainset_LGBMDataset, opts:OptionsPattern[]]:= CatchFailure @ Module[
	{
		boosterObj, catIndices, 
	 	wlParam, res, maxIter, numIter, boosterAssoc, newBooster
	}
	,
	(* parse options *)
	{maxIter, catIndices} = 
		OptionValue[LGBMBoosterTrain, {MaxIterations, "CategoricalIndices"}];
	wlParam = AssociationThread[
		Keys[$OptionNamesWLToLGBM], 
		OptionValue[LGBMBoosterTrain, Keys[$OptionNamesWLToLGBM]]
	];
	
	(* Create Booster *)
	boosterObj = 
		LGBMBoosterCreate[trainset, "CategoricalIndices" -> catIndices, FilterOptions[wlParam]];
	If[FailureQ[boosterObj], Return[boosterObj]];
	
	(* Set some properties of the Booster Object *)
	boosterAssoc = Normal[boosterObj];
	boosterAssoc["TrainingName"] = "training";
	boosterAssoc["BestIteration"] = -1;
	newBooster = System`Private`SetNoEntry @ LGBMBoosterObject[boosterAssoc];
	
	(* Train booster for numBoostRounds iterations *)
	Do[
		res = LGBMBoosterUpdate[boosterObj];
		If[FailureQ[res], Return[res]];
		(*If[res == 1, Break[]];*)
		,
		{maxIter}
	];
	numIter = LGBMBoosterGetCurrentIteration[boosterObj];
	If[FailureQ[numIter], Return[numIter]];
	
	<| "LGBMBoosterObject" -> boosterObj, "NumIterations" -> numIter, "TrainingData" -> trainset |>
];

$predictType = <|"Normal" -> 0, "RawScore" -> 1, "LeafIndex" -> 2|>;

(**********************************************************************************)
PackageExport["LGBMBoosterPredict"]

SetUsage[LGBMBoosterPredict, "LGBMBoosterPredict[boosterObj$, data$] predicts values or probabilities of labels for the testdata data$ \
using a trained LGBMBoosterObject boosterObj$.
The following options can be specified:
	'NumIteration' -> 100 : the iteration used for prediction. 
	'RawScore' -> False : whether to predict the raw score.
	'PredictLeaf' -> False : whether to predict the leaf indices.
	'Reshape' -> True : whether to reshape predictions to a matrix of dimensions - Dimensions[data$].
"];	

Options[LGBMBoosterPredict] = {
	"NumIteration" -> 100,
	"RawScore" -> False,
	"PredictLeaf" -> False,
	"Reshape" -> True
};

LGBMBoosterPredict[boosterObj_LGBMBoosterObject, data_, OptionsPattern[]]:= CatchFailure @ Module[
	{
		numIter, numClass, numTotalIter, predictType, 
		testdata, predictions, nrow, 
	 	rawscore, pleaf, reshape
	},
	If[Length[Dimensions[data]] =!= 2, 
		ReturnFailed["incorDim"]
	];
	nrow = First @ Dimensions[data];
		
	{numIter, rawscore, pleaf, reshape} = 
		OptionValue[LGBMBoosterPredict, {"NumIteration", "RawScore", "PredictLeaf", "Reshape"}];
	If[numIter <= 0, 
		numIter = boosterObj["BestIteration"];
	];
	numClass = LGBMBoosterGetNumClasses[boosterObj];
	numTotalIter = LGBMBoosterGetCurrentIteration[boosterObj];
	If[numIter > numTotalIter, numIter = numTotalIter];
	
	(* determine predict type *)
	predictType = $predictType["Normal"];
	If[rawscore === True, predictType =  $predictType["RawScore"]];
	If[pleaf === True, predictType =  $predictType["LeafIndex"]];
	
	(* prediction with empty auxiliary params *)
	predictions = If[Head[data] === SparseArray,
		LGBMBoosterPredictForSparse[boosterObj, data, predictType, numIter, ""],
		testdata = ConformDataToNumArr32[data];
		LGBMBoosterPredictForMatrix[boosterObj, testdata, predictType, numIter, ""]
	];
	If[pleaf === True, predictions = NumericArray[predictions, "Integer32"]];
	If[(reshape === True) && (Length[predictions] =!= nrow),
		If[Mod[Length[predictions], nrow] == 0, 
			predictions = ArrayReshape[predictions, {nrow, Length[predictions]/nrow}];
			,
			ReturnFailed["lenError"]
		];
	];
	Normal[predictions]
];	

General::incorDim = "Data must be a matrix.";
General::lenError = "Wrong length of prediction results.";

(**********************************************************************************)
PackageExport["LGBMBoosterPredictForMatrix"]

LGBMBoosterPredictForMatrix[boosterObj_LGBMBoosterObject, data_NumericArray, predictType_Integer, numIter_Integer, params_String]:= Module[
	{numPred, booster, result},
	If[Length[Dimensions[data]] =!= 2, ReturnFailed["incorDim"]];
	booster = boosterObj["LGBMBooster"];
	numPred = LGBMInvoke[
		lgbmBoosterCalcNumPredict, 
		booster, 
		First @ Dimensions[data], 
		predictType, 
		numIter
	];
	result = LGBMInvoke[lgbmBoosterPredictForMatrix, booster, data, predictType, numIter, params];
	If[Length[result] =!= numPred, ReturnFailed["lenError"], result]
];	

(**********************************************************************************)
PackageExport["LGBMBoosterPredictForSparse"]

LGBMBoosterPredictForSparse[boosterObj_LGBMBoosterObject, data_SparseArray, predictType_Integer, numIter_Integer, params_String]:= Module[
	{numPred, booster, result}
	,
	If[Length[Dimensions[data]] =!= 2, ReturnFailed["incorDim"]];
	booster = boosterObj["LGBMBooster"];
	numPred = LGBMInvoke[lgbmBoosterCalcNumPredict, booster, First@Dimensions[data], predictType, numIter];
	result = LGBMInvoke[lgbmBoosterPredictForSparse, booster, data, predictType, numIter, params];
	If[Length[result] =!= numPred, ReturnFailed["lenError"], result]
];
