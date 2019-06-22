Package["DAALLink`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
trainRFClassifier := trainRFClassifier = LibraryFunctionLoad[$DAALLinkLib, "WL_TrainRandomForestClassifier",
	{
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		True|False
	},
	Real
]

testRFClassifier := testRFClassifier = LibraryFunctionLoad[$DAALLinkLib, "WL_TestRandomForestClassifier",
	{
		Integer,
		Integer,
		Integer
	},
	{Integer, 1}
]

trainRFRegressor := trainRFRegressor = LibraryFunctionLoad[$DAALLinkLib, "WL_TrainRandomForestPredictor",
	{
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		Integer,
		True|False
	},
	Real
]

testRFRegressor := testRFRegressor = LibraryFunctionLoad[$DAALLinkLib, "WL_TestRandomForestPredictor",
	{
		Integer,
		Integer
	},
	{Real, 1}
]

exportRFClassifier := exportRFClassifier = LibraryFunctionLoad[$DAALLinkLib, "WL_RandomForestVectorExport",
	{
		Integer,
		"Boolean" (* True for Classify False for Predict*)
	},
	Null
]

rfGetVec := rfGetVec = LibraryFunctionLoad[$DAALLinkLib, "WL_RandomForestGetVector",
	{
		Integer
	},
	{Real, 1}
]

(*----------------------------------------------------------------------------*)
PackageExport["DAALRandomForestPredictor"]

(* This is a utility function defined in GeneralUtilities, which makes a nicely
formatted display box *)
DefineCustomBoxes[DAALRandomForestPredictor, 
	e:DAALRandomForestPredictor[mle_, oobe_, task_, class_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		DAALRandomForestPredictor, e, None, 
		{
			BoxForm`SummaryItem[{"ID: ", getMLEID[mle]}],
			BoxForm`SummaryItem[{"Error: ", oobe}],
			BoxForm`SummaryItem[{"Task: ", task}]
		},
		{},
		StandardForm
	]
]];

getMLEID[DAALRandomForestPredictor[mle_, ___]] := ManagedLibraryExpressionID[mle];
DAALGetClassNumber[DAALRandomForestPredictor[___, class_]] := class
DAALGetTaskType[DAALRandomForestPredictor[___, task_, _]] := task
DAALRandomForestPredictor[_, oobe_, __]["Error"] := oobe


(*----------------------------------------------------------------------------*)
PackageExport["DAALRandomForestTrain"]

SetUsage[DAALRandomForestTrain,"
DAALRandomForestTrain[classify$, features$, target$] trains a random forest classifier \
when classify$ is True, and a regressor when classify$ is False, given \
a set of examples features$ of type DAALNumericTable[$$], and a list of \
targets target$. For classification, target$ is a list of integers in the \
 range 0 to (Number Classes - 1). 
The following options are available:
| 'CategoricalFeatureIndices' | {} | Categorical feature positions |
| 'NTrees' | 100 | The number of trees in the forest. |
| 'ObservationsPerTreeFraction' | 1 | Fraction of the training set S used to \
form the bootstrap set for a single tree training, 0 < observationsPerTreeFraction \
<= 1. The observations are sampled randomly with replacement. |
| 'FeaturesPerNode' | 0 | The number of features tried as possible splits per \
node. If the parameter is set to 0, the library uses the square root of the \
number of features for classification and (the number of features)/3 for regression. |
| 'MaxTreeDepth' | 0 | Maximal tree depth. Default is 0 (unlimited). |
| 'MinObservationsInLeafNodes' | 1 | Minimum number of observations in the leaf node. |
| 'ImpurityThreshold' | 0 | The threshold value used as stopping criteria: \
if the impurity value in the node is smaller than the threshold, \
the node is not split anymore. |
"
]

Options[DAALRandomForestTrain] = {
	"NTrees" -> 100,
	"ObservationsPerTreeFraction" -> 1,
	"FeaturesPerNode" -> 0,
	"MaxTreeDepth" -> 0,
	"MinObservationsInLeafNodes" -> 1,
	"ImpurityThreshold" -> 0
};

(* classification case *)
DAALRandomForestTrain[True, x_DAALNumericTable, y_List, opts:OptionsPattern[]] := 
CatchFailure @ Scope[
	UnpackOptions[nTrees, observationsPerTreeFraction,
		featuresPerNode, maxTreeDepth, minObservationsInLeafNodes, impurityThreshold
	];
	classNum = Max[Max[y] + 1, 2];
	y2 = DAALNumericTableCreate[ArrayReshape[y, {Length[y], 1}]];
	mle = CreateManagedLibraryExpression["DAALRFClassifier", daalRFClassifierMLE];
	(* remove CatchFailure once intel OOB bug fixed *)
	error = CatchFailure @ safeLibraryInvoke[trainRFClassifier, getMLEID[mle], getMLEID[x], getMLEID[y2],
		classNum, nTrees, observationsPerTreeFraction, featuresPerNode,
		maxTreeDepth, minObservationsInLeafNodes, impurityThreshold, True
	];
	(* hack to fix DAAL OOB bug: compute, if failed, disable OOB *)
	If[FailureQ[error],
		error = safeLibraryInvoke[trainRFClassifier, getMLEID[mle], getMLEID[x], getMLEID[y2],
			classNum, nTrees, observationsPerTreeFraction, featuresPerNode,
			maxTreeDepth, minObservationsInLeafNodes, impurityThreshold, False (* disable OOB *)
		];
	];
	(* for small datasets, might not be an OOB if all trees are stumps *)
	If[error < 0, error = None];
	DAALRandomForestPredictor[mle, error, "Classification", classNum]
]

(* regression case *)
DAALRandomForestTrain[False, x_DAALNumericTable, y_List, opts:OptionsPattern[]] := 
CatchFailure @ Scope[
	UnpackOptions[nTrees, observationsPerTreeFraction,
		featuresPerNode, maxTreeDepth, minObservationsInLeafNodes, impurityThreshold
	];
	y2 = DAALNumericTableCreate[ArrayReshape[y, {Length[y], 1}]];
	mle = CreateManagedLibraryExpression["DAALRFRegressor", daalRFRegressorMLE];
	(* remove CatchFailure once intel OOB bug fixed *)
	error = CatchFailure @ safeLibraryInvoke[trainRFRegressor, getMLEID[mle], getMLEID[x], getMLEID[y2],
		nTrees, observationsPerTreeFraction, featuresPerNode,
		maxTreeDepth, minObservationsInLeafNodes, impurityThreshold, True
	];
	(* hack to fix DAAL OOB bug: compute, if failed, disable OOB *)
	If[FailureQ[error],
		mle = CreateManagedLibraryExpression["DAALRFRegressor", daalRFRegressorMLE];
		error = safeLibraryInvoke[trainRFRegressor, getMLEID[mle], getMLEID[x], getMLEID[y2],
			nTrees, observationsPerTreeFraction, featuresPerNode,
			maxTreeDepth, minObservationsInLeafNodes, impurityThreshold, False (* disable OOB *)
		];
	];
	(* for small datasets, might not be an OOB if all trees are stumps *)
	If[error < 0, error = None];
	DAALRandomForestPredictor[mle, error, "Regression", None]
]

(*----------------------------------------------------------------------------*)
PackageExport["DAALRandomForestEvaluate"]

DAALRandomForestEvaluate[predictor_DAALRandomForestPredictor, x_DAALNumericTable] := 
CatchFailure @ If[DAALGetTaskType[predictor] === "Classification", 
	safeLibraryInvoke[testRFClassifier, getMLEID[predictor], getMLEID[x], DAALGetClassNumber[predictor]]
	,
	safeLibraryInvoke[testRFRegressor, getMLEID[predictor], getMLEID[x]]
]

(*----------------------------------------------------------------------------*)
PackageExport["DAALRandomForestExport"]

DAALRandomForestExport[predictor_DAALRandomForestPredictor, symbol_] := CatchFailure @ Scope[
	safeLibraryInvoke[exportRFClassifier, 
		getMLEID[predictor], 
		<|Classify -> True, Predict -> False|>[symbol]
		];
	convertForest[
		<| (*the vectors below are already in the ML format but are all concatenated*)
			"FeatureIndices" -> IntegerPart @ safeLibraryInvoke[rfGetVec, 0],
			"NumericalThresholds" ->  safeLibraryInvoke[rfGetVec, 1],
			"Children" -> IntegerPart @ safeLibraryInvoke[rfGetVec, 2],
			"LeafValues" ->  safeLibraryInvoke[rfGetVec, 3],
			"NodeCounts" -> IntegerPart @ safeLibraryInvoke[rfGetVec, 4],
			"LeafCounts" -> IntegerPart @ safeLibraryInvoke[rfGetVec, 5]
		|>,
		symbol
	]
]

(*----------------------------------------------------------------------------*)

(* this function splits the vectors corresponding to a forest according to the nodecounts vector 

e.g. "FeatureIndices" -> {1,2,4,3,5,6}, "NodeCounts" -> {3, 3} mean that there are two trees in the forest and 
the result will be 
{
	<|"FeatureIndices" -> {1, 2, 4}, ...|>,
	<|"FeatureIndices" -> {3, 5, 6}, ...|>
}

*)
convertForest[forest_, symbol_] := Module[
	{forest2, nodecounts, leafcounts},	
	forest2 = forest[[{"FeatureIndices", "NumericalThresholds", "Children", "LeafValues"}]];
	nodecounts = forest["NodeCounts"];
	leafcounts = forest["LeafCounts"];
	forest2 = Association/@Transpose[Map[splitVectors[#, nodecounts, leafcounts]&, Normal@forest2]];
	If[symbol === Classify, 
		forest2[[All, "LeafValues"]] = forest2[[All, "LeafValues"]] + 1(* put back 1-start indexing *)
	];
	forest2 = Append[#, {"NominalSplits" -> {}, "RootIndex" -> 1, "NominalDimension" -> 0}] & /@ forest2;
	forest2
];


splitVectors[key:("FeatureIndices"|"NumericalThresholds")->value_, nodecounts_, leafcounts_] := 
	Thread[Rule[key, splitf[value, 0, nodecounts]]]
	
splitVectors["Children" -> value_, nodecounts_, leafcounts_] := 
	Thread[Rule["Children", splitf[Partition[value, 2], 0, nodecounts]]]
	
splitVectors["LeafValues" -> value_, nodecounts_, leafcounts_] := 
	Thread[Rule["LeafValues", splitf[value, 0, leafcounts]]]
	
		

splitf[vector_, initial_, {}] := {}
splitf[vector_, initial_, splittings_] := Join[{
	vector[[initial + 1 ;; initial + First[splittings]]]}, 
	splitf[vector,  First[splittings] + initial , Rest[splittings]]
]


