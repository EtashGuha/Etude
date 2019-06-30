Inputs: 
	$Key: TensorWithMinRankT[2] (* TODO: handle integers *)
	$Value: TensorWithMinRankT[2]
	$Query: TensorT[$$QueryShape, TensorT[$$QueryChannels, RealT]]

Outputs:
	$Output: TensorT[$$QueryShape, TensorT[$$ValueChannels, RealT]]

ReshapeParams: {"$QueryShape", "$QueryChannels", "$KeyChannels", "$ValueChannels"}

Upgraders: {
	"12.0.0" -> DropParam["$ScoreChannels"]
}

Parameters:
	$ScoringNet: NormalizedT[
		NetT[<|"Input" -> TensorT[$$KeyChannels, RealT], "Query" -> TensorT[$$QueryChannels, RealT]|>, <|"Output" -> TensorT[{}, RealT]|>],
		toScoringNet, "Bilinear"
	]
	(* Possible extension: masks for Images -> ForwardForward / BackwardBackward / ... *)
	$Mask: Defaulting[EnumT[{None, "Causal"(*, "Backward" , "Forward", "ZeroDiagonal"*)}], None]
	$ScoreRescaling: Defaulting[EnumT[{"LengthSqrt", None}], None]

	$$InputPorts: Defaulting[EnumT[{"InputQuery", "KeyValueQuery"}], "KeyValueQuery"]
	$$InputShape: SwitchedType[$Mask,
		_String -> SizeListT[1],
		_ -> SizeListT[],
		SizeListT[]
	]
	$$QueryShape: SwitchedType[$Mask,
		_String -> $$InputShape,
		_ -> SizeListT[],
		SizeListT[]
	]
	(* Scalar or vector for multi-dimensional *)
	$$QueryChannels: SizeListT[1]
	$$KeyChannels: SizeListT[1]
	$$ValueChannels: SizeListT[1]

PosArgCount: 1

AllowDynamicDimensions: True

HasDynamicPorts: True

RuntimeInferenceRules: Function[
	Append[
		Switch[#Parameters["$InputPorts"],
		"InputQuery", {
			NetPath["Inputs", "Input"] -> TensorT[$$InputShape, TensorT[$$KeyChannels, RealT]],
			$$ValueChannels -> $$KeyChannels
		},
		"KeyValueQuery", {
			NetPath["Inputs", "Key"] -> TensorT[$$InputShape, TensorT[$$KeyChannels, RealT]],
			NetPath["Inputs", "Value"] -> TensorT[$$InputShape, TensorT[$$ValueChannels, RealT]]
		}]
	, If[#Parameters["Mask"] === None, Nothing,
			$$QueryShape -> $$InputShape (* Should we support something like EitherT[{$$InputShape, {}}] ? *)
		]
	]
]

(* Hack: the easiest way of handling differing inputs is to modify the prototype network definition
when we observe $InputPorts -> "InputQuery" *)

ArgumentRewriter: checkForInputPorts

checkForInputPorts[args_List] := (
	If[!FreeQ[args, "$InputPorts" -> "InputQuery"],
		SetCurrentLayerInputs[
			<|"Input" -> TensorWithMinRankT[2], "Query" -> TensorWithMinRankT[1]|>
		];
	];
	args
);

toScoringNet["Bilinear"] := NetGraph[
	<|"Linear" -> LinearLayer["Biases" -> None, "Output" -> $Raw[VectorT[]]], "Dot" -> DotLayer[]|>, 
	{{NetPort["Query"], "Linear"} -> "Dot"},
	"Query" -> $Raw[VectorT[]],
	"Input" -> $Raw[VectorT[]]
];

toScoringNet["Dot"] := NetGraph[{DotLayer[]},
	{{NetPort["Query"], NetPort["Input"]} -> 1},
	"Query" -> $Raw[VectorT[]],
	"Input" -> $Raw[VectorT[]]
];

toScoringNet[net_] := net;

$lfNameToFunc = <|"LengthSqrt" -> Sqrt|>;

toUnaryFunc[name_] := toUnaryFunc[name] = ToUnaryElementwiseFunction @ $lfNameToFunc @ name;

MinArgCount: 0

Writer: Function @ Scope[

	ilen = First @ #$InputShape;
	varLengthQ = !IntegerQ[ilen];

	ilenNode = GetDynamicLengthNode[ilen];

	querySequenceQ = Length[#$QueryShape] > 0;

	causalMaskQ = MatchQ[#Mask, "Causal"];
	scoringScaleQ = #ScoreRescaling =!= None;
	outputRank = Length[#$ValueChannels];

	Switch[#$InputPorts,
		"InputQuery",
			keys = values = GetInput["Input"];
		,
		"KeyValueQuery",
			keys = GetInput["Key"]; 
			values = GetInput["Value"];
	];									

	queries = GetInput["Query"];						(* batch, ilen, isize *)

	(* Note from Tali after discussin with Jerome: 
	* basically, if both q and i are dynamic, we have two choices about how to proceed:
	1) batch-flatten both q and i into batch dim by broadcasting across the other (an outer product)
	2) unroll over one

	Jerome chose option 2. However, there are some mitigations. For example, the batch_dot op is effectively
	already an outer product of dot products. if it is the only operation, we can avoid the memory-expensive
	broadcast step, which makes option 1 more attractive. this is the optimizableQ check, below.
	*)

	(* flatXXX is the form of port XXX in which all query/input dimensions other than the channel dimension
	have been flattened together, to make (batch, qflat, qchannels) *)

	extraQueryRank = Length[#$QueryShape];
	extraInputRank = Length[#$InputShape];

	isize = Times @@ Rest[#$InputShape];
	ilen *= isize;
	If[varLengthQ, ilenNode = SowTimesScalar[ilenNode, isize]];
	(* ^ this compensates for the fact that we have flattened the dynamic dim with other dims *)

	flatKeys = SowFlatten[keys, extraInputRank-1, 1];			(* (batch, ilen, kchan *)
	flatValues = SowFlatten[values, extraInputRank-1, 1];		(* (batch, ilen, vchan *)

	flatQueries = Switch[extraQueryRank, 
		0, SowInsertDim[queries, 1], (* TODO: maybe make SowFlatten work on -1 etc by inserting dims ? *)
		1, queries,
		_,SowFlatten[queries, extraQueryRank-1, 1]
	];		(* (batch, qlen, qchan *)

	arange = SowNode["_arange", {}, "start" -> "1", "infer_range" -> True, "dtype" -> $DTypeMXName];
	If[scoringScaleQ,
		(* arange has trouble inferring through the score rescaling code, so in that case
		we must 'hint' it by adding it to a zero vector of the right size *)
		zeros = SowNode["_zeros", {}, "shape" -> "(1,)", "dtype" -> $DTypeMXName];
		queryShapeNode = SowNode["broadcast_like", {zeros, flatQueries}, "lhs_axes" -> "(0,)", "rhs_axes" -> "(1,)"];
		rangeQueryIndices = SowPlus[arange, queryShapeNode];
	,
		rangeQueryIndices = arange;
	];

	numQueries = Times @@ #$QueryShape;

	(***** Fast Path *****)
	If[!$ForceAttentionUnrolling && optimizableQ[#ScoringNet],
		(* Optimized path, for special cases of ScoringNet *)
		keyTransform = keyTransformBeforeDot[#ScoringNet];
		(* the scoring net often has e.g. linear transforms that apply to the key and query before they are combined.
		we can handle these if they are immediately before the dot product. these are pre-applied via batch folding. *)

		If[keyTransform =!= Null,
			name = First @ Keys @ #ScoringNet["Nodes"];
			flatKeys = SowFlatMap[
				SowInnerNet[<|"Input" -> #|>, {"Parameters", "ScoringNet", "Nodes", name}, keyTransform]["Output"]&,
				flatKeys
			]; (* (batch, ilen, kchan', where kchan' is output chan dim of key transform *)
		];

		scores = SowGEMM2[flatQueries, flatKeys, False, True];	(* batch, qlen, ilen *)
	,
		flatScores = If[numQueries === 1,
			(* Only one query *)
			SowInsertDim[
				Lookup[
					SowSubNet["ScoringNet",
						<|"Input" -> SowFlatten[flatKeys]
						, "Query" ->
							SowFlatten[
								SowNode[
									"broadcast_like", {flatQueries, flatKeys}
									, "lhs_axes" -> "(1,)", "rhs_axes" -> "(1,)"
								]
							]
						|>
					], "Output"
				], 0
			]
		,
			(* flatKeys is (batch, ilen, kchan) *)
			(* flatQueries is (batch, qlen, qchan *)
			transposedQueries = SowTranspose01[flatQueries];      (* (qlen, batch, qchan) *)
			transposedQueries = SowInsertDim[transposedQueries, 2];     (* (qlen, batch, 1, qchan) *)

			flatFlatKeys = SowFlatten[flatKeys]; (* (batch * ilen, kchan *)

			(* inside foreach:
				query is ( batch, 1, qchan )
				input is ( batch * ilen, kchan )
				output is ( batch * ilen ) *)
			Block[{$DisableMXForEach = $DisableMXForEach || $ForceAttentionUnrolling}, 
				First @ SowForEach[
					Function[query,
						Lookup[
							SowSubNet["ScoringNet",
								<|"Input" -> flatFlatKeys
								, "Query" ->
									SowFlatten[
										SowNode[
											"broadcast_like", {query, flatKeys}
											, "lhs_axes" -> "(1,)", "rhs_axes" -> "(1,)"
										]
									]
								|>
							]
							, {"Output"}
						]
					], 
					numQueries,
					{transposedQueries}
				]
			]
		];  (* flatScores : (qlen, batch * ilen) *) 

		scores = SowNode[
			"reshape_like", {flatScores, flatKeys}, 
			"lhs_begin" -> 1, "lhs_end" -> 2, "rhs_begin" -> 0, "rhs_end" -> 2
		];
		scores = SowTranspose01[scores]; (* (batch, qlen, ilen) *) 
	];

	If[scoringScaleQ,
		scores = If[varLengthQ || causalMaskQ,
			scoreNorm = SowInsertDimList[
				SowScalarFunction[toUnaryFunc[#ScoreRescaling], If[causalMaskQ, rangeQueryIndices, ilenNode]], 
				If[causalMaskQ, {0, 2}, {1, 2}]
			];
			SowBDivide[scores, scoreNorm]
		,
			SowDivideScalar[scores, $lfNameToFunc[#ScoreRescaling][ilen]]
		];
	];
	(* Mask *)
	If[causalMaskQ,
		scores = SowTranspose[scores, {1, 2, 0}];								(* qlen, ilen, batch *)
		scores = SowSeqMaskBatchwise[scores, rangeQueryIndices, "-1e37"];
		scores = SowTranspose[scores, {2, 0, 1} ];								(* batch, qlen, ilen *)
	];
	If[varLengthQ && !causalMaskQ,
		scores = SowSwapAxis[scores, 1, 2];										(* batch, ilen, qlen *)
		scores = SowSeqMaskBatchwise[scores, ilenNode, "-1e37"];
		scores = SowSwapAxis[scores, 1, 2];										(* batch [*osize], qlen, ilen *)
	];
	scores = SowSoftmax[scores, -1];	
	
	If[ShouldSetMetricQ["AttentionWeights"],
		scr = scores;	
		If[extraQueryRank > 1,
			scr = SowUnflatten[scr, queries, extraQueryRank-1, 1, 1];
		];
		(* ^ Reshape to get back the orig query dims *)
		If[extraInputRank > 1,
			scr = SowUnflatten[scr, keys, extraInputRank - 1, 2 + extraQueryRank-1, 1];
		];
		(* ^ Reshape to get back the orig input dims *)
		
		SetMetric["AttentionWeights", scr];
	];	

	(* scores is (batch, qlen, ilen)
	   values is (batch, ilen, vchan)
	   output is (batch, qlen, vchan) *)
	outs = SowGEMM2[scores, flatValues];

	If[extraQueryRank === 0,
		outs = SowReshape[outs, #$ValueChannels];
	,
		outs = SowUnflatten[outs, queries, extraQueryRank-1, 1];
		(* queries = (batch, q1, ..., qn, qchan) *)
		(* output = (batch, q1, ..., qn, values) *)
	];

	SetOutput["Output", outs];
]

(* returns True for "Dot" and "Bilinear" *)
optimizableQ[scoringNet_] := Or[
	scoringNet[["Type"]] === "Graph" && Length @ scoringNet[["Nodes"]] === 1 && scoringNet[["Nodes",1,"Type"]] === "Dot",
	keyTransformBeforeDot[scoringNet] =!= Null
];

(* returns 
	- Null, if NOT in the "Bilinear" case
	- the subnet that is applied to the "Input" port before doing the Dot, if the net can be decomposed like this
*)
keyTransformBeforeDot[scoringNet_] := Cached[iKeyTransformBeforeDot, scoringNet];
iKeyTransformBeforeDot[scoringNet_] := If[
	And[
		Length @ scoringNet[["Nodes"]] === 2,
		scoringNet[["Nodes",2,"Type"]] === "Dot",
		MemberQ[scoringNet[["Edges"]], NetPath["Nodes", First @ Keys @ scoringNet[["Nodes"]], "Inputs", "Input"] -> NetPath["Inputs", "Input"]]
	], 
	scoringNet[["Nodes", 1]]
];

netDotAfterInputAndTargetTransform := NetGraph[
	{LinearLayer[4], LinearLayer[4], DotLayer[]},
	{NetPort["Input"] -> 1, NetPort["Query"] -> 2, 1 -> 3, 2 -> 3},
	"Input" -> 2, "Query" -> 3
];
netDotAfterInputTransform := NetGraph[
	{ConstantTimesLayer["Scaling" -> {2, -1, 1.5}], DotLayer[]},
	{NetPort["Input"] -> 1 -> 2, NetPort["Query"] -> 2},
	"Input" -> 3, "Query" -> 3
];
netAdditiveScoring := NetGraph[
	{LinearLayer[10,"Biases"->None], LinearLayer[10,"Biases"->None], ThreadingLayer[Tanh[#1 + #2]&], LinearLayer[{},"Biases"->None]},
	{NetPort["Input"] -> 1 -> 3, NetPort["Query"] -> 2 -> 3 -> 4},
	"Input" -> 5, "Query" -> 3
];
netAdditiveScoring2 := NetGraph[
	{CatenateLayer[], LinearLayer[10,"Biases"->None], Tanh, LinearLayer[{},"Biases"->None]},
	{{NetPort["Input"], NetPort["Query"]} -> 1 -> 2 -> 3 -> 4},
	"Input" -> 5, "Query" -> 3
];

dotInputRank2 := NetGraph[{FlattenLayer[], FlattenLayer[], DotLayer[]},{NetPort["Input"]->1, NetPort["Query"]->2, {1,2}->3}, "Input"->{Automatic, Automatic}, "Query"->{Automatic, Automatic}];
multiDot := NetGraph[{5, 5, ThreadingLayer[Plus], Tanh, 5},{NetPort["Input"]->1, NetPort["Query"]->2, {1,2}->3->4->5}];
multiDotOutputRank2 := NetGraph[{4, 4, ThreadingLayer[Plus], Tanh, {5,4}},{NetPort["Input"]->1, NetPort["Query"]->2, {1,2}->3->4->5}];

(* see the comment LayerTests.m about testing of layers that employ SowSeqMask *)

Tests: {
	(* Optimized & defaults ones *)
	{"Dot", (*"Key" -> {3,3},*) "Value" -> {3,5}, "Query" -> {2,3}} -> "2*5_V85yDrvIUAA_A3StYN2DVy0=5.061416e+0",
	{"Dot", (*"Key" -> {"Varying",3},*) "Value" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_QH9zzxOWjuo_W57u2b5Pfgw=7.565303e+0",
	{"Bilinear", "Key" -> {2,5}, "Value" -> {2,3}, "Query" -> {3,4}} -> "3*3_CnR0ikaupLY_GjFm54kWGVQ=4.440291e+0",
	{"Bilinear", "Key" -> {Automatic,5}, "Value" -> {"Varying",3}, "Query" -> {"Varying",4}} -> "3*3_WDrp4JHo/Yk_bXFSBOaj5Ww=3.783087e+0",
	{Hold @ netAdditiveScoring, "Key" -> {Automatic,5}, "Value" -> {3,4}, "Query" -> {2,3}} -> "2*4_T+erSyjqYWA_LDHVLCsfiqM=4.136163e+0",
	{Hold @ netAdditiveScoring2, "Key" -> {"Varying",5}, "Value" -> {"Varying",4}, "Query" -> {"Varying",3}} -> "3*4_DPr1heJfc78_bo1o7NBRrw8=5.948393e+0",

	(* Custom "ScoringNet" *)
	{Hold @ netDotAfterInputAndTargetTransform, "$InputPorts" -> "InputQuery", "Input" -> {3,2}, "Query" -> {2,3}} -> "2*2_U8wyxc/7WfE_ZDDZOz35jdc=1.508825e+0",
	{Hold @ netDotAfterInputAndTargetTransform, "$InputPorts" -> "InputQuery", "Input" -> {"Varying",2}, "Query" -> {"Varying",3}} -> "3*2_CayEn8FGUvY_AOKU6n79to8=2.280666e+0",
	{Hold @ netDotAfterInputTransform, "$InputPorts" -> "InputQuery", "Input" -> {"Varying",3}, "Query" -> {"Varying",3}} -> "3*3_EMsSyckseW8_IRVoIeen/GA=3.151370e+0",
	{Hold @ netAdditiveScoring, "$InputPorts" -> "InputQuery", "Input" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_eF2ZuO7rVmw_XOa5Cx0FTxY=6.758457e+0",
	{Hold @ netAdditiveScoring2, "$InputPorts" -> "InputQuery", "Input" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_MhlrJehMaew_bRL08XF7ZDE=6.070540e+0",
	
	(* {"Input", "Query"} ports (historical setting) *)
	{"Dot", "$InputPorts" -> "InputQuery", "Input" -> {2,5}, "Query" -> {3,5}} -> "3*5_GUq3wE5WHyg_XN2SsxcfCrk=6.016190e+0",
	{"Dot", "$InputPorts" -> "InputQuery", "Input" -> {"Varying",13}, "Query" -> {"Varying",13}} -> "3*13_MhNY/Z2P2pA_OSqphQvIQDU=1.732208e+1",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Input" -> {7,4}, "Query" -> {3,5}} -> "3*4_BFKzfjQiEIY_dkc1KH/rOyA=5.280171e+0",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Input" -> {"Varying",2}, "Query" -> {"Varying",7}} -> "3*2_CNqgeiahgx0_dMm11nUEowo=2.071213e+0",

	(* Option "ScoringScale" *)
	{"Dot", "$InputPorts" -> "InputQuery", "ScoreRescaling" -> "LengthSqrt", "Input" -> {5,3}, "Query" -> {2,3}} -> "2*3_K6KsKbcrJl8_c2x8g1n2q6g=2.604345e+0",
	{"Dot", "ScoreRescaling" -> "LengthSqrt", "Value" -> {5,5}, "Query" -> {2,3}} -> "2*5_Tznr1KS3qSM_In14WXmUiTY=4.328105e+0",
	{"Dot", "ScoreRescaling" -> "LengthSqrt", "Value" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_CornAgjIgv8_GSTy7k6SScw=7.522131e+0",
	{"Dot", "ScoreRescaling" -> "LengthSqrt", "Value" -> {"Varying", 5}, "Query" -> {2, 3}} -> "2*5_ZLshcmrObzM_d0haPbq+L2M=5.024913e+0",
	{"Bilinear", "ScoreRescaling" -> "LengthSqrt", "Key" -> {5,5}, "Value" -> {Automatic,3}, "Query" -> {3,4}} -> "3*3_ctlcBLcdPo4_D9YFtcduDHE=3.874887e+0",
	{"Bilinear", "ScoreRescaling" -> "LengthSqrt", "Key" -> {"Varying",5}, "Value" -> {"Varying",3}, "Query" -> {"Varying",4}} -> "3*3_Jb5XknEv4vk_faQpcl+TOBY=3.991437e+0",
	{"Bilinear", "ScoreRescaling" -> "LengthSqrt", "Key" -> {"Varying", 6}, "Value" -> {"Varying", 5}, "Query" -> {4, 3}} -> "4*5_XG8sOrcwzUE_bMTc7CI9YbM=9.730099e+0",
	{Hold @ netAdditiveScoring, "$InputPorts" -> "InputQuery", "ScoreRescaling" -> "LengthSqrt", "Input" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_SDVHLA0+Dso_O4se5k2I57k=6.713597e+0",
	{Hold @ netAdditiveScoring2, "ScoreRescaling" -> "LengthSqrt", "Key" -> {7,5}, "Value" -> {7,4}, "Query" -> {2,3}} -> "2*4_aSo+FwYjYoc_cQRvglQAzLE=3.977322e+0",
	{Hold @ netAdditiveScoring, "ScoreRescaling" -> "LengthSqrt", "Key" -> {"Varying",5}, "Value" -> {Automatic,4}, "Query" -> {"Varying",3}}  -> "3*4_TLVeKhT3X0Y_JrTcC1Oj78Y=5.932735e+0",
	{Hold @ netAdditiveScoring2, "ScoreRescaling" -> "LengthSqrt", "Value" -> {"Varying", 5}, "Query" -> {2, 3}} -> "2*5_UO4ty0ilg8k_NCDUuJdRuQE=4.739860e+0",

	(* Causal masking *)
	{"Dot", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "Input" -> {5, 6}, "Query" -> {5, 6}} -> "5*6_RauVvqD2ovY_UrstGr+kpy8=1.314408e+1",
	{"Dot", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "Input" -> {"Varying", 5}, "Query" -> {"Varying", 5}} -> "3*5_P1bwfa7sH8I_Lr6hoPIQc44=5.838590e+0",
	{"Dot", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Input" -> {6, 5}, "Query" -> {6, 5}} -> "6*5_fThkvTyfMfU_QPEo3wbNKI0=1.250078e+1",
	{"Dot", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Input" -> {"Varying", 5}, "Query" -> {"Varying", 5}} -> "3*5_Hc6iOZEA+Og_CplhaULesqU=5.833833e+0",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "Input" -> {5, 5}, "Query" -> {5, 3}} -> "5*5_AusNbDSJhXg_GzzcfwfkTBQ=9.866275e+0",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "Input" -> {"Varying", 5}, "Query" -> {"Varying", 3}} -> "3*5_ZsuACSmFpd4_GxtyxEkhoBQ=5.642976e+0",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Input" -> {5, 5}, "Query" -> {5, 3}} -> "5*5_GAkOuKeDD8I_CzymKA6sqe8=9.955441e+0",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Input" -> {"Varying", 5}, "Query" -> {"Varying", 3}} -> "3*5_Ak7fTWkR2es_GBGzJ+5DnlI=5.693743e+0",
	{"Dot", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Value" -> {6, 5}, "Query" -> {6, 5}} -> "6*5_DmPRFc/XG/o_DfL35Em6mic=1.247542e+1",
	{"Dot", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Value" -> {"Varying", 5}, "Query" -> {"Varying", 5}} -> "3*5_SwYKfpGWKmc_cIebz1nzzbc=6.994309e+0",
	{"Bilinear", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Key" -> {Automatic, 4}, "Value" -> {5, 5}, "Query" -> {5, 3}} -> "5*5_A8u2rwLGh4Q_PDrjg5A4QKk=1.160705e+1",
	{"Bilinear", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Key" -> {"Varying", 4}, "Value" -> {"Varying", 5}, "Query" -> {"Varying", 3}} -> "3*5_Jz7NT5+Xa3M_cMKRr8bWST4=6.227284e+0",
	{Hold @ netAdditiveScoring, "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "Input" -> {5,5}, "Query" -> {5,3}} -> "5*5_SV/FGQX6eB0_cuiAP6NGTos=1.070261e+1",
	{Hold @ netAdditiveScoring2, "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "Input" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_BWUhY+eTQqA_G+H7CoUsddE=5.729235e+0",
	{Hold @ netAdditiveScoring, "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Input" -> {5,5}, "Query" -> {5,3}} -> "5*5_Q7+UbsvxlgU_Ktnsy8wgESY=1.064684e+1",
	{Hold @ netAdditiveScoring2, "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Input" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_UO7oRLhvS5U_HqGPgCTRLuc=5.754768e+0",
	{Hold @ netAdditiveScoring, "Mask" -> "Causal", "Value" -> {5, 4}, "Query" -> {5, 3}} -> "5*4_b7iVhlKLX6g_cdTLt1SeGWA=8.487166e+0",
	{Hold @ netAdditiveScoring2, "Mask" -> "Causal", "Value" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_AaiG2PZlaZc_IeAe47VE1tc=6.868546e+0",
	{Hold @ netAdditiveScoring, "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Value" -> {5, 4}, "Query" -> {5, 3}} -> "5*4_M/SWKr4r+BQ_S2clV6sZgBs=8.854272e+0",
	{Hold @ netAdditiveScoring2, "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Value" -> {"Varying",5}, "Query" -> {"Varying",3}} -> "3*5_X7W8bIVYBpM_AA1cMkTdwgc=6.904272e+0",
	{"Mask" -> "Causal", "$InputPorts" -> "InputQuery", "ScoreRescaling" -> "LengthSqrt", "Input" -> {"Varying", 5}, "Query" -> {5, 5}} -> "5*5_JT6RcrRjNWg_cG3G63mcPig=9.932647e+0",
	{"Mask" -> "Causal", "$InputPorts" -> "InputQuery", "ScoreRescaling" -> "LengthSqrt", "Input" -> {5, 5}, "Query" -> {"Varying", 5}} -> "5*5_JT6RcrRjNWg_cG3G63mcPig=9.932647e+0",
	{"Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Key" -> {"Varying", 4}, "Value" -> {"Varying", 4}, "Query" -> {5, 3}} -> "5*4_fyUZZJP3Z+M_FNzjXSPncRg=9.613337e+0",
	{"Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Key" -> {5, 4}, "Value" -> {5, 4}, "Query" -> {"Varying", 3}} -> "5*4_fyUZZJP3Z+M_FNzjXSPncRg=9.613337e+0",

	(* Processing arrays of rank > 2 *)
	{"Dot", "$InputPorts" -> "InputQuery", "Query" -> {2, 4}, "Input" -> {5, 3, 2, 4}} -> "2*4_Za2uLab8Pg4_J1zLggAcBEM=3.844315e+0",
	{"Dot", "$InputPorts" -> "InputQuery", "Query" -> {2, 3, 4}, "Input" -> {5, 4}} -> "2*3*4_S5LSBuE9ZV8_Tj3vo0UvpXM=1.042138e+1",
	{"Dot", "$InputPorts" -> "InputQuery", "Query" -> {"Varying", 3, 2, 2, 4}, "Input" -> {"Varying", 3, 2, 4}} -> "3*3*2*2*4_S0CfXWJxNWU_QsOb6TxnZos=7.096527e+1",
	{"Dot", "Query" -> {2, 3, 4}, "Key" -> {5, 6, 4}, "Value" -> {5, 6, 3}} -> "2*3*3_LlTBiG8gaTw_TA3z832+POk=8.369591e+0",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Query" -> {2, 4}, "Input" -> {5, 3, 2, 5}} -> "2*5_bTtMWenuv1E_I4mKBmgBcxg=4.218785e+0",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Query" -> {2, 3, 4}, "Input" -> {5, 5}} -> "2*3*5_K5SQ1b2E1bg_Gm16D3LChkw=1.250797e+1",
	{"Bilinear", "$InputPorts" -> "InputQuery", "ScoreRescaling" -> "LengthSqrt", "Query" -> {"Varying", 3, 2, 2, 4}, "Input" -> {"Varying", 3, 2, 5}} -> "3*3*2*2*5_B4EsvoNucxU_FQiNesWEAVs=8.064258e+1",
	{"Bilinear", "ScoreRescaling" -> "LengthSqrt", "Query" -> {"Varying", 3, 4}, "Key" -> {"Varying", 6, 2}, "Value" -> {"Varying", 6, 3}} -> "3*3*3_RyBxu0sms5Q_Fpsue2Y/BZQ=1.244879e+1",
	{Hold @ netAdditiveScoring, "$InputPorts" -> "InputQuery", "Query" -> {2, 3}, "Input" -> {5, 6, 5}} -> "2*5_GL+BSUgPKkY_KZ8v61+lCYc=4.926056e+0",
	{Hold @ netAdditiveScoring2, "ScoreRescaling" -> "LengthSqrt", "Query" -> {2, 3, 3}, "Value" -> {5, 6, 5}} -> "2*3*5_CxLakTb39HI_GMgE4GY3yGU=1.432295e+1",
	{Hold @ netAdditiveScoring, "ScoreRescaling" -> "LengthSqrt", "Query" -> {"Varying", 3, 3}, "Value" -> {"Varying", 6, 5}} -> "3*3*5_R2jSnsLAalo_d1XjR9SWOI8=1.972460e+1",
	{Hold @ netAdditiveScoring2, "ScoreRescaling" -> "LengthSqrt", "Query" -> {"Varying", 3, 3}, "Value" -> {"Varying", 6, 5}} -> "3*3*5_PyVQLToIRXo_BItHT0ZPZBk=2.046021e+1",

	(* Query is a single vector *)
	{"Dot", "$InputPorts" -> "InputQuery", "Query" -> {4}, "Input" -> {"Varying", 4}} -> "4_C9Ks0lJkpN4_BEB+RP1JMbk=1.767433e+0",
	{"Bilinear", "$InputPorts" -> "InputQuery", "Query" -> {4}, "ScoreRescaling" -> "LengthSqrt", "Input" -> {5, 2, 3}} -> "3_QZOOZQXQFpU_aDjs7JDH8cY=1.373257e+0",
	{Hold @ netAdditiveScoring, "$InputPorts" -> "InputQuery", "Query" -> {3}, "Input" -> {5, 2, 5}} -> "5_X+8u2mwVEkg_Df+A1P4t9eY=2.231389e+0",
	{Hold @ netAdditiveScoring, "ScoreRescaling" -> "LengthSqrt", "Query" -> {3}, "Value" -> {"Varying", 2, 3}} -> "3_HfI9L4cgAso_TAQJhz7xR8k=1.300341e+0",
	{Hold @ netAdditiveScoring2, "$InputPorts" -> "InputQuery", "Query" -> {3}, "Input" -> {5, 2, 5}} -> "5_K9APRbWLDGY_Y7z/pPXFfpA=2.379796e+0",
	{Hold @ netAdditiveScoring2, "ScoreRescaling" -> "LengthSqrt", "Query" -> {3}, "Value" -> {"Varying", 2, 3}} -> "3_aQBqDPB4VhM_YOmkogzaNKE=1.284718e+0",
	
	(* TODO: Mask option does not import if the query is a single vector (let the user does his stuff)
	{"Dot", "Mask" -> "Causal", "Query" -> {4}, "Input" -> {"Varying", 4}} -> "4_C9Ks0lJkpN4_BEB+RP1JMbk=1.767433e+0",
	{"Bilinear", "Mask" -> "Causal", "Query" -> {4}, "ScoreRescaling" -> "LengthSqrt", "Input" -> {5, 2, 3}} -> "3_QZOOZQXQFpU_aDjs7JDH8cY=1.373257e+0",
	{Hold @ netAdditiveScoring, "Mask" -> "Causal", "ScoreRescaling" -> "LengthSqrt", "Query" -> {3}, "Value" -> {"Varying", 2, 3}} -> "3_HfI9L4cgAso_TAQJhz7xR8k=1.300341e+0",
	*)

(*
	(* Multi-dimensional attention *)
	{Hold @ multiDot, "$InputPorts" -> "InputQuery", "Input" -> {4, 5}, "Query" -> {6, 3}} -> "6*5_ZlXppB+RpFc_D95ib1ghVRI=1.332442e+1",
	{Hold @ multiDot, "Key" -> {Automatic, 4}, "Value" -> {4, 5}, "Query" -> {6, 3}} -> "6*5_XgWI3pVgEmE_RL8Y99i968A=1.370204e+1",
	{Hold @ multiDot, "Key" -> {Automatic, Automatic, Automatic, 4}, "Value" -> {4, 2, 3, 5}, "Query" -> {6, 2, 3}} -> "6*2*5_JyAqR1fyfWk_c2FN3f6qupw=2.797078e+1",
	{Hold @ multiDot, "ScoreRescaling" -> "LengthSqrt", "Key" -> {Automatic, Automatic, 4}, "Value" -> {4, 2, 5}, "Query" -> {6}} -> "5_cCpeFsOun2o_WeHKME/X18o=2.480318e+0",
	{Hold @ multiDot, "ScoreRescaling" -> "LengthSqrt", "Mask" -> "Causal", "Key" -> {Automatic, 4}, "Value" -> {4, 5}, "Query" -> {4, 6}} -> "4*5_XntJ9gSfFGw_AsOkRFyMLtU=9.352415e+0",


	(* Multi-dimensional attention with rank > 1 *)
	{Hold[multiDotOutputRank2], "Value" -> {7, 5, 4}, "Key" -> {7, 3}, "Query" -> {2}} -> "5*4_ZscQ/ld3h6A_ACzx2TwqTEQ=9.271690e+0",
	{Hold[multiDotOutputRank2], "Value" -> {"Varying", 5, 4}, "Key" -> {"Varying", 3}, "Query" -> {"Varying", 2}} -> "3*5*4_XBzcOKXcAqY_fYOR6CG37GQ=2.960827e+1",
	*)
	(* Cases of errors: Mask for any arrays -- not implemented (ex: we should propose the 4 possible scanning directions for 2D shapes, not only Backward/Forward) *)
	{"Dot", "$InputPorts" -> "InputQuery", "Mask" -> "Causal", "Input" -> {5, 4, 2}, "Query" -> {5, 4, 2}} -> "Inferred inconsistent value for query channels.",

	(* Cases of errors: channels of rank > 1 -- not implemented *)
	{Hold @ dotInputRank2} -> "Inferred inconsistent value for key channels.",
	{"Dot", "Value" -> {4, 4, 4}, "Key" -> {4, 4}, "Query" -> {4, 4}} -> "Inferred inconsistent value for value channels.",

	(* Cases of errors: Input with no length -- sick case not supported *)
	{"Dot", "$InputPorts" -> "InputQuery", "Input" -> {5}, "Query" -> {5}} -> "Specification {5} is not compatible with port \"Input\", which must be an array of rank \[GreaterEqual] 2.",
	{"Dot", "Value" -> {5}, "Query" -> {5, 4}} -> "Specification {5} is not compatible with port \"Value\", which must be an array of rank \[GreaterEqual] 2.",

	(* Cases of errors: Invalid configurations *)
	{"Dot", "$InputPorts" -> "InputQuery", "Input" -> {3, 4}, "Query" -> {4, 3}} -> "Type inconsistency in DotLayer: cannot take dot product of arrays with shapes 3 and 4.",
	{"Dot", "Key" -> {3, 4}, "Value" -> {3, 3}, "Query" -> {4, 3}} -> "Type inconsistency in DotLayer: cannot take dot product of arrays with shapes 3 and 4.",
	{"Bilinear", "Value" -> {3, 3}, "Query" -> {4, 3}} -> "Cannot initialize net: unspecified or partially specified shape for array \"Weights\".",
	{"$InputPorts" -> "InputQuery", "Mask" -> "Causal", "Input" -> {6, 5}, "Query" -> {5, 5}} -> "Value of query shape ({5}) is inconsistent with value of input shape ({6}).",
	{"Mask" -> "Causal", "Value" -> {6, 4}, "Query" -> {5, 3}} -> "Value of query shape ({5}) is inconsistent with value of input shape ({6}).",
	{"Bilinear", "Key" -> {Automatic, Automatic, Automatic, 4}, "Value" -> {4, 2, 5},"Query" -> {6}} -> "Inferred inconsistent ranks for input \"Value\" (a rank-3 array versus a rank-4 array).",

	(* Cases of errors: Wrong argument type *)
	{$Failed} -> "$Failed is not a layer, a net, or a valid specification for one.",
	{"Dot", "Input" -> {"Varying", 2}} -> "\"Input\" is not a known parameter for AttentionLayer. Allowed parameters include: \"ScoringNet\", \"Mask\", \"ScoreRescaling\", \"Key\", \"Value\", and \"Query\".",
	{"Dot", "$InputPorts" -> "InputQuery", "Key" -> {"Varying", 2}} -> "\"Key\" is not a known parameter for AttentionLayer. Allowed parameters include: \"ScoringNet\", \"Mask\", \"ScoreRescaling\", \"Input\", and \"Query\".",
	{"Dot", "$InputPorts" -> "gnark"} -> "The value of $InputPorts -> gnark should be either \"InputQuery\" or \"KeyValueQuery\".",
	
	{"Dot", "Mask" -> "gnark"} -> "The value of Mask -> gnark should be either None or \"Causal\".",

	(* Previously (when supporting multi-dimensional): "Value of value channels ({5}) is inconsistent with value of key channels ({3}).",*)
	{Hold @ multiDot, "$InputPorts" -> "InputQuery", "Input" -> {4, 3}} -> "Output port \"Output\" of net specified for the scoring net (first argument) takes a length-5 vector of real numbers, but a port producing a number is required."
}
