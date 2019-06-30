Inputs:
	$Input: ChannelT[$$Channels, TensorT[$$InputDimensions]]

Output: ChannelT[$$Channels, TensorT[$OutputDimensions]]

Parameters:
	$OutputDimensions: SizeListT[2]
	$ReflectionProbabilities: Defaulting[ListT[2,IntervalScalarT[0,1]], {0,0}]
	$$Channels: SizeT
	$$InputDimensions: SizeListT[2]

PosArgCount: 1

PostInferenceFunction: Function[
	If[Min[$$InputDimensions - $OutputDimensions] < 0, 
		FailValidation["output dimensions `` must be smaller than input dimensions ``.", $OutputDimensions, $$InputDimensions]
	];
]

HasTrainingBehaviorQ: Function[True]

Writer: Function @ Scope[
	input = GetInput["Input"];
	before = #$InputDimensions;
	after = #OutputDimensions;
	rprob = #ReflectionProbabilities;
	If[!$TMode,
		output = SowNode["slice", input, "begin" -> Join[{None, None}, Floor@((before - after)/2)], "end" -> Join[{None, None}, before - Ceiling@((before - after)/2)]];
	,
		cropLimit = SowFixedArray["Crop", toNumericArray[before - after + 1]];
		rcrop = SowNode["floor", SowBHad[SowUniformRandom[{2}], cropLimit]];
		If[Max[rprob] > 0, 
			rprob = SowFixedArray["RProb", toNumericArray[rprob - 0.5]];
			reflect = SowNode["round", SowBPlus[SowUniformRandom[{2}], rprob]];
		,
			reflect = SowZeroArray[{2}];
		];
		output = SowNode["ImageAugment", {input, SowBlockGrad @ rcrop, SowBlockGrad @ reflect}, "crop_size" -> after];
		(* TODO: handle blocking of gradient within ImageAugment layer directly *)
	];
	SetOutput["Output", output];
]

Tests: {
	{{3, 3}, "Input" -> {3, 5, 5}} -> "3*3*3_SaAKJtscRoM_Ewpx3fakyaI=1.259962e+1",
	{{7, 3}, "Input" -> {3, 12, 7}} -> "3*7*3_f2NwucDOdbs_NueQhQKu450=3.109027e+1"
}
