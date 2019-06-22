Package["NumericArrayUtilities`"]

(******************************************************************************)
PackageExport["FindNextThreshold"]
PackageExport["UpdateClusteringVector"]
PackageExport["DecisionTreeEvaluation"]

SetUsage[FindNextThreshold, "
Computes the best splitting (continuous or categorical).\
Parallel computation on every active leaves.\
"
]

SetUsage[UpdateClusteringVector, "
It updates the clustering vector by splitting each cluster in two according to \
the threshold found before.
"
]

DeclareLibraryFunction[findIndexThresholdClassify, 
	"find_next_indexthreshold_Classify",
	{
		{Integer, 2,"Constant"}, 
		{Real, 2,"Constant"}, 
		{Integer, 1, "Constant"}, 
		{Integer, 1, "Constant"}, 
		Integer,
		{Integer, 1, "Constant"},
		Integer,
		Real,
		Integer
	}
	,
	{Real, 2, Automatic}
]

DeclareLibraryFunction[findIndexThresholdPredict, 
	"find_next_indexthreshold_Predict",
	{
		{Integer, 2,"Constant"}, 
		{Real, 2,"Constant"}, 
		{Real, 1, "Constant"}, 
		{Integer, 1, "Constant"}, 
		{Integer, 1, "Constant"},
		Integer,
		Real,
		Integer
	}
	,
	{Real, 2, Automatic}
]


DeclareLibraryFunction[updatePartitioningVector, 
	"update_clustering_vector",
	{
		{Integer, 1,"Shared"}, 
		{Integer, 2,"Constant"}, 
		{Real, 2,"Constant"}, 
		{Integer, 2,"Constant"}, 
		{Integer, 1,"Constant"}, 
		{Real, 2,"Constant"}, 
		Integer, 
		Integer, 
		{Integer, 1, "Constant"}
	}
	,
	Integer
]

DeclareLibraryFunction[decisionTreeEvaluator, 
	"tree_evaluation",
	{
		{"NumericArray", "Constant"}, 
		{"NumericArray", "Constant"}, 
		{"NumericArray", "Constant"}, 
		{"NumericArray", "Constant"}, 
		{Integer, _,"Constant"}, 
		{Real, _,"Constant"}, 
		Integer, 
		Integer, 
		Integer
	}
	,
	{Integer, 1, Automatic}
]

UpdateClusteringVector[
	partitionedData_, datacategoricalT_, datacontinuousT_, parents_, 
	indeces_, criteria_, nCat_, nCon_] := PreemptProtect@Module[
	{partitioneddDataScaled, roots, arrayComponent, arrayComponentsPs, rule, rootRule},			
	roots = parents[[All, 1]]/2;		
	arrayComponent = ArrayComponents[roots];
	rootRule = 	Thread[Rule[roots, arrayComponent]];	
	arrayComponentsPs = ArrayComponents[parents, 2, rootRule];
	rule = Association@Thread[Rule[Flatten[arrayComponentsPs], Flatten[parents]]];
	partitioneddDataScaled = Developer`ToPackedArray@Lookup[Association[rootRule], partitionedData, -1];		
	updatePartitioningVector[
			partitioneddDataScaled, datacategoricalT, datacontinuousT, 
			arrayComponentsPs, indeces - 1, criteria, nCat, nCon, arrayComponent
	];						
	Lookup[rule, Round[partitioneddDataScaled], -1]

]

DecisionTreeEvaluation[tree_, xCat_, xCon_] := PreemptProtect@Module[
	{featureIndeces, nominalSplit, numericalThreshold, children, dimCat, 
		root, dimCon, n},
	n = Max[Length[xCat], Length[xCon]];
	featureIndeces = tree[["FeatureIndices"]];
	If[featureIndeces === {-1}, Return[ConstantArray[-1, n]]];
	nominalSplit = tree[["NominalSplits"]];
	numericalThreshold = tree[["NumericalThresholds"]];
	children = tree[["Children"]];
	root = tree[["RootIndex"]];	
	dimCat = tree[["NominalDimension"]];
	dimCon = Length[First[xCon, {}]];	
	decisionTreeEvaluator[
		featureIndeces,
		numericalThreshold,
		nominalSplit,
		children,
		xCat,
		xCon,
		dimCat,
		dimCon,
		root
	]
]


FindNextThreshold[datacategoricalT_, datacontinuousT_, labels_, 
	partinioningData_, nclasses_, parents_, variableSampleSize_, smoothing_, seed_] := PreemptProtect@Module[
	{array, arrayComponents, rule},
	arrayComponents = ArrayComponents[parents];	
	rule = Thread[Rule[parents, arrayComponents]];		
	array = Switch[nclasses,
		0,
			findIndexThresholdPredict[
				datacategoricalT ,datacontinuousT, labels, Lookup[Association[rule], partinioningData, -1], 
				arrayComponents, variableSampleSize, smoothing, seed
			],
		_Integer,
			findIndexThresholdClassify[
				datacategoricalT ,datacontinuousT, labels, Lookup[Association[rule], partinioningData, -1],
				nclasses, arrayComponents, variableSampleSize, smoothing, seed
			],
		_,
			AbortMachineLearning[]
	];
	array[[All, 2]] = Round[array[[All, 2]]];
	MapThread[
		returnIndexThreshold[
			#1, datacategoricalT, datacontinuousT, partinioningData, 
			labels, #2, nclasses, smoothing
		]&, 
		{array, parents}
	]	
]


returnIndexThreshold[array_, datacategoricalT_, datacontinuousT_, partinioningData_, 
	labels_, parent_, nclasses_, smoothing_] := Module[
	{index, threshold, dimCategorical,dimContinuous, carray, selectedLabels, res, leafValue, mean, stdv},
	index = Last[array];
	threshold = array[[1]];
	dimCategorical = Length[datacategoricalT];
	dimContinuous = Length[datacontinuousT];
	If[index == dimCategorical+dimContinuous+1, 
		selectedLabels = Pick[labels, partinioningData, parent];
			
		leafValue = Switch[nclasses,
			0,
				(*this stdv has been regularized by adding a point (mean + smoothing) to the sample list*)
				mean = Mean[selectedLabels];
				stdv = Sqrt[Total[Append[(selectedLabels - mean)^2, smoothing]]/(Length[selectedLabels] + smoothing)];
				{mean, stdv + $MinMachineNumber},
			_,
				carray = ConstantArray[0, nclasses];
				(carray[[#[[1]]]] = #[[2]])&/@Tally[selectedLabels];
				carray + smoothing
		];
		Return[{0, leafValue, 0, 0, 0, parent}]		
	];
	
	If[index <= dimCategorical,
		res = {index, 0, Round[threshold], 2 parent ,  2 parent +1, parent},
		res = {index, threshold, 0,  2 parent ,  2 parent +1, parent};
	];
	res
]

