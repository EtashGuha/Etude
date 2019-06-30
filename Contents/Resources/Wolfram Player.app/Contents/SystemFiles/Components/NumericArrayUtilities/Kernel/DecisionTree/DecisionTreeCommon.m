Package["NumericArrayUtilities`"]

(******************************************************************************)

PackageExport["UpdateClusteringVector"]
PackageExport["DecisionTreeEvaluation"]


SetUsage[UpdateClusteringVector, "
It updates the clustering vector by splitting each cluster in two according to \
the threshold found before.
"
]

SetUsage[DecisionTreeEvaluation, "
It travels the tree down to the leaves in parallel. It returns the negative index of the leaf in the LeafValues vector.
"
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
		{"RawArray", "Constant"}, 
		{"RawArray", "Constant"}, 
		{"RawArray", "Constant"}, 
		{"RawArray", "Constant"}, 
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
	indeces_, criteria_, nCat_, nCon_] := Module[
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

