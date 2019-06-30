Package["NumericArrayUtilities`"]

(******************************************************************************)
PackageExport["FindNextUnsupervisedThreshold"]


SetUsage[FindNextUnsupervisedThreshold, "
Computes the best splitting (continuous or categorical).\
Parallel computation on every active leaves.\
"
]



DeclareLibraryFunction[findIndexThresholdLearnDistribution, 
	"find_next_indexthreshold",
	{
		{Integer, 2,"Constant"}, 
		{Real, 2,"Constant"}, 
		{Integer, 1, "Constant"}, 
		{Integer, 1, "Constant"}, 
		Integer,
		{Real, 3, "Constant"},
		{Integer, 2, "Constant"},
		Real
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



FindNextUnsupervisedThreshold[datacategoricalT_, datacontinuousT_, partinioningData_, parents_, variableSampleSize_,
	contLimits_, catLimits_, smoothing_] := Module[
	{array, max, differences, arrayComponents, rule},	
	max = Max[parents] - 1;	

	
	arrayComponents = ArrayComponents[parents];
	rule = Thread[Rule[parents, arrayComponents]];
	differences = -Apply[Subtract, contLimits, {2}];
	array = findIndexThresholdLearnDistribution[
				datacategoricalT,datacontinuousT, Lookup[Association[rule], partinioningData, -1]
				(*ArrayComponents[partinioningData, 1, rule]*), arrayComponents, 
				variableSampleSize, contLimits, catLimits, smoothing
			];	

	array[[All, 2]] = Round[array[[All, 2]]];								
	MapThread[
		returnIndexThreshold[#1, datacategoricalT, datacontinuousT, partinioningData, #2, #3, #4]&, 
		{array, parents, differences, catLimits}
	]
]



returnIndexThreshold[array_, datacategoricalT_, datacontinuousT_, partinioningData_, parent_,
	diff_, catLimits_] := Module[
	{index, threshold, dimCategorical,dimContinuous, n, volumeCon, volumeCat, volume, pdf, res},
	n = Count[partinioningData, parent];
	index = Last[array];
	threshold = array[[1]];
	dimCategorical = Length[datacategoricalT];
	dimContinuous = Length[datacontinuousT];
	If[index == dimCategorical+dimContinuous+1, 
		volumeCon = Times@@diff;	
		volumeCat = Times@@catLimits;	
		volume = N@volumeCon*volumeCat;	
		If[volume == 0., pdf = 0, pdf = (n+1)/volume];	
			Return[{0, pdf, 0, 0, 0, n, parent}]
	];
	
	If[index <= dimCategorical,
		res = {index, 0, Round[threshold], 2 parent ,  2 parent +1, n, parent},
		res = {index, threshold, 0,  2 parent ,  2 parent +1, n, parent};
	];
	res
]
