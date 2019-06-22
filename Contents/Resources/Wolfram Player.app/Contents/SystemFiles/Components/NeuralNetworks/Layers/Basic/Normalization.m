Input: TensorT[$$Dimensions, AtomT]
Output: TensorT[$$Dimensions]

Parameters:
	$AggregationLevels: LevelSpecT[2;;All, True]
	$ScalingLevels: LevelSpecT["Complement", False, {"Same", "Complement"}]
	$Epsilon: Defaulting[IntervalScalarT[10^-14, Infinity], 0.001]
	$$Dimensions: SizeListT[]
	$$StatsDimensions: ComputedType[SizeListT[],
		toScalingParameterDimensions[First @ $ScalingLevels, $$Dimensions, First @ $AggregationLevels]
	, {$$Dimensions, $ScalingLevels, $AggregationLevels}]

Arrays:
	$Scaling: Nullable @ TensorT[$$StatsDimensions, RealT]
	$Biases: Nullable @ TensorT[$$StatsDimensions, RealT]

AllowDynamicDimensions: True

MinArgCount: 0
MaxArgCount: 2

ArgumentRewriter: RenameDeprecatedOption[{"Gamma" -> "Scaling", "Beta" -> "Biases"}]

toScalingLevelSet[scalingLevels_, rank_, aggregationLevels_] := CheckLevelSpec[
	NormalizationLayer,
	HandleShapeException[
		NormalizationLayer,
		Switch[scalingLevels,
			"Complement", Complement[Range[rank], ToLevelSet[aggregationLevels, rank]],
			"Same", ToLevelSet[aggregationLevels, rank],
			_, ToLevelSet[scalingLevels, rank]
		]
	],
	False, {"Same", "Complement"}, "ScalingLevels"
];

toScalingParameterDimensions[scalingLevels_, dims_, aggregationLevels_] := Module[{actualScalingLevels, res},
	actualScalingLevels = toScalingLevelSet[scalingLevels, Length[dims], aggregationLevels];
	If[Length[actualScalingLevels] === 0, {},
		res = Part[dims, actualScalingLevels];
		If[MemberQ[res, LengthVar[_]],
			FailValidation[NormalizationLayer, "The levels on which are applied Scaling and Biases cannot be dynamic."]
		,
			res
		]
	]
];

Writer: Function @ Scope[

	input = GetInput["Input"];
	inputDims = GetInputDims["Input"];
	rank = Length[inputDims];

	aggregationLevels = ToLevelSet[First @ #AggregationLevels, rank];
	scalingLevels = toScalingLevelSet[First @ #ScalingLevels, rank, First @ #AggregationLevels];

	mxnetInstanceNormQ = And[
		aggregationLevels ===  Range[2, rank],
		scalingLevels === {1},
		#Scaling =!= None,
		#Biases =!= None
	];

	If[mxnetInstanceNormQ,
		(* MXNet native implementation of InstanceNorm *)
		(* TODO: make MXNet support Real64 for InstanceNorm, and remove casts *)
		{input, scaling, biases} = SowCast[{input, #Scaling, #Biases}, $DType, "Real32"];
		normalized = SowNode["InstanceNorm", {input, scaling, biases}, "eps" -> #Epsilon];
		normalized = SowCast[normalized, "Real32", $DType];
	,
		normalized = input;

		If[Length[aggregationLevels] > 0,

			lenNode = GetDynamicLengthNode @ First[inputDims, None];

			(* Compute mean and standard deviation *)
			mean = computeAverage[input, aggregationLevels, inputDims, lenNode];
			diff = SowBMinus[input, insertDims[mean, aggregationLevels, inputDims]];
			squaredDiff = SowSquare[diff];
			stdev = SowSqrt @ SowPlusScalar[ (* Note: SowMaxScalar could be used here *)
				computeAverage[squaredDiff, aggregationLevels, inputDims, lenNode],
				#Epsilon
			];

			(* Normalize *)
			normalized = SowBDivide[diff, insertDims[stdev, aggregationLevels, inputDims]];
		];

		(* Learned rescaling *)
		If[#Scaling =!= None,
			normalized = SowBHad[normalized, insertDimsComplementary[#Scaling, scalingLevels, inputDims]];
		];
		If[#Biases =!= None,
			normalized = SowBPlus[normalized, insertDimsComplementary[#Biases, scalingLevels, inputDims]];
		];
	];

	SetOutput["Output", normalized];
]

insertDims[node_, levels_, inputDims_] := Module[{newDims},
	newDims = Prepend[Rest[inputDims], -1]; (* Will induce length *)
	newDims[[levels]] = 1;
	newDims = Prepend[newDims, 0]; (* Batch dimension *)
	SowUReshape[node, newDims]
];
insertDimsComplementary[node_, levels_, inputDims_] := Module[{newDims},
	newDims = inputDims /. LengthVar[_] :> 1;
	newDims[[Complement[Range[Length[inputDims]], levels]]] = 1;
	newDims = Prepend[newDims, 1]; (* Batch dimension  *)
	SowUReshape[node, newDims]
];


computeAverage[node_, levelsToAggregate_, inputDims_, lenNode_] := Module[{input, mean, count},

	input = node;
	mustHandleVaryingDimension = (lenNode =!= None && MemberQ[aggregationLevels, 1]);
	rank = Length[inputDims];

	If[mustHandleVaryingDimension,
		(* Apply the mask, whose value depends on the type of aggregation *)
		input = SowSeqMaskBatchwise[input, lenNode, 0.];
	];

	mean = SowNode["sum_axis", input, "axis" -> aggregationLevels];

	If[!mustHandleVaryingDimension,
		count = Times @@ inputDims[[aggregationLevels]];
		mean = SowDivideScalar[mean, count];
	,
		If[Length @ aggregationLevels > 1,
			count = Times @@ inputDims[[DeleteCases[aggregationLevels, 1]]];
			mean = SowDivideScalar[mean, count];
		];
		reshapedLenNode = If[rank === Length[aggregationLevels], lenNode,
			SowReshape[lenNode, Table[1, rank - Length[aggregationLevels]]]];
		mean = SowNode["broadcast_div", {mean, reshapedLenNode}];
	];

	mean
];


initMethod = Hold["Scaling" -> NormalDistribution[0, 1], "Biases" -> NormalDistribution[0, 1]];

Tests: {

	(* InstanceNormalization
		1st= channels
		2nd and 3rd= image dimensions
	*)
	{"Input" -> {1, 1, 1}} -> "1*1*1_f+wff5A6deg_VCrp1AZ3zRg=9.887545e-1",
	{"Input" -> {3, 5, 5}} -> "3*5*5_APdpcBWCLAg_QaBgMRoJ+Pg=9.757319e+1",
	{"Biases" -> {0.2, 0.1, -0.2}, "Scaling" -> {0.01, 0.12, 0.2}, "Epsilon" -> 0.01, "Input" -> {3, 6, 6}} -> "3*6*6_ckTE4SWzCb8_csA2bbqNQ2c=2.020558e+1",
	{"Input" -> {3, 7, 5, 2}, initMethod} -> "3*7*5*2_Anqjqar2VrQ_GFEWQ1//WAU=2.695219e+2",
	{"Input" -> {3, 7, 5, 2}, "Scaling" -> 2, "Biases" -> -1} -> "3*7*5*2_LJkbY65Z6gk_EXtssXhA6K8=4.018403e+2",

	(* LayerNorm
		1st= sequence length
		2nd= features, normalized one by one
	*)
	{2;;, "Same", "Input" -> {"Varying", 50}, "Biases" -> None, "Scaling" -> None} -> "3*50_bekYJ1l14rQ_WxHpzGk5woo=1.323065e+2",
	{2;;, "Same", "Input" -> {"Varying", 50}, "Biases" -> None, "Scaling" -> Table[1,50]} -> "3*50_bekYJ1l14rQ_WxHpzGk5woo=1.323065e+2",
	{2;;, "Same", "Input" -> {"Varying", 50}, "Biases" -> Table[0,50], "Scaling" -> Table[1,50]} -> "3*50_bekYJ1l14rQ_WxHpzGk5woo=1.323065e+2",
	{2;;, "Same", "Input" -> {"Varying", 50}, "Biases" -> Table[0,50], "Scaling" -> None} -> "3*50_bekYJ1l14rQ_WxHpzGk5woo=1.323065e+2",
	{2;;, "Same", "Input" -> {"Varying", 5, 4, 3}, "Biases" -> None, "Scaling" -> None} -> "3*5*4*3_C8K4HeZMisM_M5T5Dy0pv4o=1.577502e+2",
	{-1, "Same", "Input" -> {"Varying", 50}, "Biases" -> None, "Scaling" -> None} -> "3*50_bekYJ1l14rQ_WxHpzGk5woo=1.323065e+2",
	{1, "Same", "Input" -> {30, 50}, "Biases" -> None, "Scaling" -> None} -> "30*50_N6oQJR8vi3w_NGnZIMQRC1c=1.291475e+3",
	{1, "Input" -> {"Varying", 50}, "Biases" -> None,  "Scaling" -> None} -> "3*50_Oxukxt3WF9E_JolYsLuxoa8=1.292062e+2",
	{{2, 4}, "Same", "Input" -> {"Varying", 5, 4, 3}, "Biases" -> None, "Scaling" -> None} -> "3*5*4*3_AWl8mtO5vd0_aQbdjwX3IPc=1.594429e+2",
	{1 ;; 2, "Same", "Input" -> {2, 3}, "Scaling" -> None, "Biases" -> None} -> "2*3_cKunr7/isoY_XFxqKFPGsv0=4.991813e+0",
	{1 ;; 2, "Same", "Input" -> {2, 3}, "Scaling" -> 1, "Biases" -> 0} -> "2*3_cKunr7/isoY_XFxqKFPGsv0=4.991813e+0",
	{2;;, "Same", initMethod, "Input" -> {"Varying", 50}} -> "3*50_aBrbWpgv24k_QUch4rCHbm4=1.693649e+2",
	{2;;, "Same", initMethod, "Input" -> {"Varying", 5, 4, 3}} -> "3*5*4*3_Yp/8USqV5RE_Q00Ug1YmOzY=1.950148e+2",
	{-1, "Same", initMethod, "Input" -> {"Varying", 50}} -> "3*50_aBrbWpgv24k_QUch4rCHbm4=1.693649e+2",
	{1, "Same", initMethod, "Input" -> {30, 50}} -> "30*50_AhYLCWcskB4_Y3M6VP7OCyo=1.584874e+3",
	{{2, 4}, "Same", initMethod, "Input" -> {"Varying", 5, 4, 3}} -> "3*5*4*3_NNRRmc7g3rU_fa6kvY1kpOY=1.868377e+2",
	{2;;, "Same", "Input" -> {"Varying", 50}, "Scaling" -> Range[50], "Biases" -> -Range[50]} -> "3*50_Z8PKbChmblE_JoPKUxCHvLQ=4.589259e+3",
	{1, "Same", "Input" -> {30, 50}, "Biases" -> Table[0,30], "Scaling" -> Table[1,30]} -> "30*50_N6oQJR8vi3w_NGnZIMQRC1c=1.291475e+3",
	{{2, 4}, "Same", "Input" -> {"Varying", 5, 4, 3}, "Scaling" -> Table[0.1 *i * (-1)^i * {1,-2,3}, {i,5}], "Biases" -> Table[0.1 *i * (-1)^(i+1) * {1,-2,3}, {i,5}]} -> "3*5*4*3_dLVeKGmGelA_KbeK67knUj4=1.294306e+2",
	{1 ;; 2, "Same", initMethod, "Input" -> {2, 3}} -> "2*3_A+EhT7Fl5Jk_K1IV8pn+s5k=4.398448e+0",

	(* In between *)
	{3;;, 2;;3, initMethod, "Input" -> {"Varying", 2, 3, 4}} -> "3*2*3*4_QYHMNA4EjSQ_CDx8XUkJ1pw=7.736093e+1",
	{2;;3, 3;;, initMethod, "Input" -> {"Varying", 5, 4, 3}} -> "3*5*4*3_GbN4OPSck/c_IhTbamrMyKw=1.930594e+2",

	(* Corner cases : handled *)
	{{}, {1}, "Input" -> {2, 3}} -> "2*3_c4sP0RXQKR4_TWWZqhMPQPc=8.469628e+0",
	{{}, All, "Input" -> {2, 3}} -> "2*3_VyAOG9jA63Y_GE3y5h8yo8A=6.663319e+0",
	{1, "ScalingLevels" -> 1, "Input" -> {3}} -> "3_E/113PX48e8_ASTnjrcYwQA=5.401979e+0",
	{1, "ScalingLevels" -> 1, "Input" -> {3, "Integer"}} -> "3_QcATnvxP3QI_CusLrcuTsEU=2.859565e+0",


	(* Corner cases : not implemented yet *)
	{{}, "Same", "Input" -> {2, 3}} -> "Validation failed for NormalizationLayer: ScalingLevels specification was not a non-zero integer, a span, All, \"Same\", \"Complement\", or a non-empty list of integers or spans.",
	{2 ;; All, {}, "Input" -> {2, 3}} -> "Value of {} given for the scaling levels (second argument) was invalid: level specification was not a non-zero integer, a span, All, \"Same\", \"Complement\", or a non-empty list of integers or spans.",
	
	(* Invalid configuration *)
	{"Input" -> {"Varying", 5, 5}} -> "Validation failed for NormalizationLayer: The levels on which are applied Scaling and Biases cannot be dynamic.",
	{"Input" -> {3}} -> "Validation failed for NormalizationLayer: level specification Span[2, All] exceeds rank of input (1)",
	{0} -> "Value of 0 given for the aggregation levels (first argument) was invalid: level specification was not a non-zero integer, a span, All, or a list of integers or spans.",
	{1, "Same", "Input" -> {2, 3}, "Scaling" -> {1, 2, 3}} -> "Inferred inconsistent value for stats dimensions."
}
