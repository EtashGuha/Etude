Inputs:
	$Input: TensorT[$$InputDimensions, SwitchedType[$TargetForm,
		"Binary" -> 				RealT, 
		"Probabilities"|"Index" -> 	VectorT[$$Classes],
		RealTensorT
	]]
	$Target: TensorT[$$InputDimensions, SwitchedType[$TargetForm,
		"Binary" -> 				RealT,
		"Index" -> 					IndexIntegerT[$$Classes], 
		"Probabilities" -> 			VectorT[$$Classes],
		AnyTensorT
	]]

Outputs:
	$Loss: ScalarT

AllowDynamicDimensions: True

PostInferenceFunction: Function[
	If[$$Classes === 1,
		FailValidation["input to CrossEntropyLossLayer should be a vector of at least two elements."]
	]
]

FinalCheck: Function[
	If[Head[$$Classes] === LengthVar,
		FailValidation["the number of classes cannot be dynamic."]
	]
]

Parameters:
	$TargetForm: EnumT[{"Binary", "Index", "Probabilities"}]
	$$InputDimensions: SizeListT[]
	$$Classes: SwitchedType[$TargetForm,
		"Binary" -> None,
		_String ->	SizeT,
		Nullable[SizeT]
	]

MinArgCount: 1

Writer: Function @ Scope[
	input = GetInputMetaNode["Input"];
	target = GetInputMetaNode["Target"];

	constInputRank = Count[#$InputDimensions, _Integer];
	meanf = errf = If[constInputRank > 0, SowFlatMean, Identity];

	fused = False;
	If[#TargetForm =!= "Binary",
		{prev, prevNode} = GetPreviousLayer["Input", True]; 
		If[prev["Type"] === "Softmax", 
			level = prev["Parameters", "Level"];
			If[level < 0, level = GetInputRank["Input"] + 1 + level];
			input = SowMetaSoftmax[prevNode, level, True]; 
			fused = True];
	];

	{input, target, postfMean, postfSum, postfNoAgg} = MakeVariableLengthLossData[input, target];

	If[#TargetForm === "Binary",
		loss = SowMinus @ SowPlus[
			SowHad[target, SowLog @ SowPlusEps @ input], 
			SowHad[SowOneMinus @ target, SowLog @ SowPlusEps @ SowOneMinus @ input]
		];
	,
		If[!fused, input = SowSafeLog[input]];
		If[#TargetForm === "Index",
			target = SowMinusScalar[target, 1]; (* convert to 0-index *)
			loss = SowNode["pick", {input, target}, "axis" -> -1, "keepdims" -> False];
		,
			loss = SowHad[input, target];
			loss = SowSumAxis[loss, -1]; (* <- sum over the class dims *)
		];
		meanf = meanf /* SowMinus;
		(* ^ negate it when its smallest *)
	];

	(* loss comes out of the above section as being (b, ...) or (b*t, ...). the meanf
	will get rid of the ..., the postfMean will average over the masked ts *)

	SetOutput["Loss", postfMean @ meanf @ loss];

	getCounts = Function[
		level = If[#3, -2, -1];
		inpExp = SowInsertDim[#1, level]; (* batch * ... * num_classes * 1 * (n) *)
		tgtExp = SowInsertDim[#2, level]; (* batch * ... * num_classes * 1 * (1) *)
		notInp = SowNot[inpExp];
		notTgt = SowNot[tgtExp];
		
		tp = SowAnd[inpExp, tgtExp]; (* batch * ... * num_classes * 1 * (n) *)
		fp = SowAnd[inpExp, notTgt];
		fn = SowAnd[notInp, tgtExp];
		tn = SowAnd[notInp, notTgt];

		sumDims = Range[constInputRank];
		tp = SowSumAxis[tp, sumDims]; (* batch * num_classes * 1 * (n) *)
		fp = SowSumAxis[fp, sumDims];
		fn = SowSumAxis[fn, sumDims];
		tn = SowSumAxis[tn, sumDims];

		SowJoin[tp, fp, fn, tn, level] (* batch * num_classes * 4 * (n) *)
	];

	If[ShouldSetMetricQ["Counts"],	
		Switch[#TargetForm,
			"Index",
				input1Hot = SowOneHot[SowNode["argmax", input, "axis" -> "-1"], #$Classes];
				(* ^ input is an n-dimmensional array of log-probabilities *)
				target1Hot = SowOneHot[target, #$Classes],
			"Binary",
				input1Hot = SowNode["round", input]; 
				(* ^ input is a single probability *)
				target1Hot = SowNode["round", target],
			"Probabilities",
				input1Hot = SowOneHot[SowNode["argmax", input, "axis" -> "-1"], #$Classes]; 
				(* ^ input is an n-dimmensional array of log-probabilities *)
				target1Hot = SowOneHot[SowNode["argmax", target, "axis" -> "-1"], #$Classes];
		];	
		(* ^ for the counts we always want input at target to be 1-hot vectors describing the class *)

		counts = getCounts[input1Hot, target1Hot, False];
		SetMetric["Counts", postfSum @ counts];
	];

	If[ShouldSetMetricQ["ROCCounts"],	
		Switch[#TargetForm,
			"Index",
				inp = SowExp[input]; 
				tgt = SowOneHot[target, #$Classes],
			"Binary",
				inp = input; 
				tgt = SowNode["round", target],
			"Probabilities",
				inp = SowExp[input]; 
				tgt = SowOneHot[SowNode["argmax", target, "axis" -> "-1"], #$Classes]
		];
		(* ^ for the counts we always want target to be a 1-hot vectors describing the class, 
			while input must be a vector of probabilties for each class  *)

		inp = SowInsertDim[inp, -1]; (* batch * ... * num_classes * 1 *)
		tgt = SowInsertDim[tgt, -1]; 	
		
		cutoffs = SowSigmoid @ SowNode["_arange",  {}, 
			"start" -> -7, "stop" -> 7., 
			"step" -> N @ 1/10, "dtype" -> $DTypeMXName
		];
		(* TODO: explain broadcasting dims *)
		bin = SowNode["broadcast_greater_equal", {inp, cutoffs}]; (* batch * num_classes * n *)
		counts = getCounts[bin, tgt, True];
		SetMetric["ROCCounts", postfSum @ counts];
	];

	If[ShouldSetMetricQ["Entropy"],
		entropy = Switch[#TargetForm,
			"Binary",
				SowMinus @ SowPlus[
					SowHad[input, SowLog @ SowPlusEps @ input], 
					SowHad[SowOneMinus @ input, SowLog @ SowPlusEps @ SowOneMinus @ input]
				],
			_,
				inpProb = SowExp[input]; 
				inpLogProb = input;
				SowSumAxis[SowHad[inpProb, inpLogProb], -1]
		];			

		SetMetric["Entropy", postfMean @ meanf @ entropy];
	];


	If[!$WithinOperatorQ && #TargetForm =!= "Binary",
		Cases[		
			$MeasurementPaths,
			Append[Append[$path, "ErrorRate"], k_Integer] :> createTopK[k, #TargetForm]
		];
	];


	Switch[#TargetForm,
		"Index", 
			input = SowNode["argmax", input, "axis" -> "-1"];
			target = target,
		"Binary", 
			input = SowNode["round", input]; 
			target = SowNode["round", target],
		"Probabilities",
			input = SowNode["argmax", input, "axis" -> "-1"];
			target = SowNode["argmax", target, "axis" -> "-1"]
	];
	(* ^ for the pairs and error rate want input and target indices describing the class *)

	If[ShouldSetMetricQ["Pairs"],		
		pairs = postfNoAgg @ SowNode["concat", {SowInsertDim[target, -1], SowInsertDim[input, -1]}, "num_args" -> 2, "dim" -> -1];
		SetMetric["Pairs", 
			SowUReshape[pairs, 0, -1, 2], 
			With[n = If[#TargetForm =!= "Binary", #$Classes, 2], 
				MXNetLink`NDSparseCountsArray[#, {n, n}]&]
		];
	];	

	If[ShouldSetMetricQ["ErrorRate"],	
		SetMetric["ErrorRate", postfMean @ errf @ SowNode["_not_equal", {input, target}]];
	];		
]

IsLoss: True

SummaryFunction: Function[
	If[StringQ[#TargetForm], 
		HoldForm[CrossEntropyLossLayer[#TargetForm]],
		CrossEntropyLossLayer
	]
]

Tests: {
	{"Index", "Input" -> 3} -> "_A+O/esckMuo_Al4TO/DUBiY=1.050349e+0",
	{"Probabilities", "Input" -> 3} -> "_Czm72Ft0JbQ_RQJYQ7V7KQA=1.239247e+0",
	{"Binary", "Input" -> {}} -> "_P2V/LF49EVU_KGjv+XzHNTw=7.051201e-1",

    (* these are higher-dimension tests *)
	{"Index", "Input" -> {3, 5}} -> "_XhH+2GaKfXw_To8bRotBq3g=1.528176e+0",
	{"Probabilities", "Input" -> {3, 5}} -> "_LBHD28h3HPE_FdGoWdDbUjY=3.118586e+0",
	{"Binary", "Input" -> 3} -> "_fdXQ2C0BeMs_IXvz0l6CXZo=6.615562e-1",
	{"Binary", "Input" -> {3, 5}} -> "_A+O/esckMuo_fRRnF06idDc=1.066194e+0",

	{"Probabilities", "Input" -> {2, 3, 5}} -> "_QXO9IkoiJu4_bk3ynK1caS4=2.234891e+0",
	{"Index", "Input" -> {5, 3, 7}} -> "_Ub1LJNSHJ+8_CoNuJdPrMNs=1.001477e+0",
	{"Probabilities", "Input" -> {2, 3, 2, 5}} -> "_NSBlg4XV3uo_BlgyGfRbV0o=2.062849e+0",
	{"Index", "Input" -> {2, 3, 4, 7}} -> "_Czm72Ft0JbQ_VbV5FFeS8D0=1.191704e+0",

	(* varying cases, but Prob and Index don't allow num of classes to vary *)
	{"Binary", "Input" -> "Varying"} -> "_fdXQ2C0BeMs_L7kb8lkO/ws=6.615562e-1",
	{"Binary", "Input" -> {"Varying", 3}} -> "_Ub1LJNSHJ+8_PT0ntwHGzCk=1.040417e+0",
	{"Probabilities", "Input" -> {"Varying", 3}} -> "_GDrCke6xaME_ffkhzR+jVR0=2.007059e+0",
	{"Index", "Input" -> {"Varying", 3}} -> "_NSBlg4XV3uo_PrQWDoa+A/Y=2.075394e+0"
}

createTopK[k_, targetForm_] := Scope[
	inpTopK = SowNode["topk", input, "k" -> k];
	If[targetForm === "Probabilities",
		tgtTopK = SowNode["argmax", target, "axis" -> "-1"],
		tgtTopK = target
	];
	tgtTopK = SowInsertDim[tgtTopK, -1];
	SetMetric["ErrorRate" -> k, postfMean @ errf @ SowOneMinus @ SowSumAxis[SowNode["broadcast_equal", {inpTopK, tgtTopK}], -1]];
];