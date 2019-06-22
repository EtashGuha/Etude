Input: AnyTensorT

Output: AnyTensorT

Parameters:
	$Function: EnumT[{Mean, Min, Max, Total, Times}]
	$Levels: LevelSpecT[2;;All, True]

(* TODO: support Variance and StandardDeviation via SowFlatVariance *)

ShapeFunction: Map1[AggregationShape[#, First @ $Levels]&]

RankFunction: Map1[AggregationRank[#, First @ $Levels]&]

TypeFunction: Function[
	DefaultedType @ Switch[$Function,
		Min|Max, #,
		Mean, {RealT},
		_, ReplaceAll[#, IndexIntegerT[_Integer] -> IndexIntegerT[Infinity]]
	]
]

AllowDynamicDimensions: True

MinArgCount: 1

Writer: Function @ Scope[

	inputDims = GetInputDims["Input"];
	function = #Function;
	
	levelsToAggregate = ToLevelSet[First @ #Levels, Length[inputDims]];
	
	input = GetInput["Input"];
	rank = Length[inputDims];
	lenNode = GetDynamicLengthNode @ First @ inputDims;

	If[Length[levelsToAggregate] == 0, out = input,

		mustHandleVaryingDimension = (lenNode =!= None && MemberQ[levelsToAggregate, 1]);

		If[mustHandleVaryingDimension,
			(* Apply the mask, whose value depends on the type of aggregation *)
			input = SowSeqMaskBatchwise[input, lenNode, toMaskValue[function]];
		];

		out = SowNode[
			toSymbolType[function],
			input,
			"axis" -> levelsToAggregate
		];
		If[function === Mean,
			If[!mustHandleVaryingDimension,
				count = Times @@ inputDims[[levelsToAggregate]];
				out = SowDivideScalar[out, count];
			,
				If[Length @ levelsToAggregate > 1,
					count = Times @@ inputDims[[DeleteCases[levelsToAggregate, 1]]];
					out = SowDivideScalar[out, count];
				];
				reshapedLenNode = If[rank === Length[levelsToAggregate], lenNode,
					SowReshape[lenNode, Table[1, rank - Length[levelsToAggregate]]]];
				out = SowNode["broadcast_div", {out, reshapedLenNode}];
			];
		];
	];
	SetOutput["Output", out];
]

toSymbolType = <|
	Max -> "max_axis",
	Min -> "min_axis",
	Total -> "sum_axis",
	Mean -> "sum_axis",
	Times -> "prod"
|>;

toMaskValue = <|
	Max ->"-1e37",
	Min -> "+1e37",
	Total -> "0.0",
	Mean -> "0.0",
	Times -> "1.0"
|>;

Tests: {
	{Max, All, "Input" -> 4} -> "_IWqOWEDnRsY_Su2OBskfoT0=7.695246e-1",
	{Max, 1, "Input" -> 4} -> "_IWqOWEDnRsY_Su2OBskfoT0=7.695246e-1",
	{Max, {1, 3}, "Input" -> {2, 3, 4}} -> "3_NHzKAqS8WzE_BWFfB1red6w=2.533165e+0",
	{Max, 2, "Input" -> {2, 3, 4}} -> "2*4_I4AaMPCaDdI_UeVR1YhS60c=4.946831e+0",
	{Max, 1 ;; 3, "Input" -> {2, 3, 4}} -> "_a+6OH6f4oJI_Nri4ch5a/s0=9.264554e-1",
	{Max, 2, "Input" -> {2, 3, 4, 5}} -> "2*4*5_ZfjJvQTQUyA_Hvv7F0eicPE=2.871128e+1",
	{Max, 2 ;; 4, "Input" -> {2, 3, 4, 5}} -> "2_Tb+A8CskU/E_V6JFiXvji+E=1.939947e+0",
	{Max, All, "Input" -> {2, 3, 4, 5}} -> "_GPuLbfiRyzQ_Hn241y9oU3E=9.870309e-1",
	{Min, 2, "Input" -> {2, 3, 4, 5}} -> "2*4*5_KVdgVK3dUtA_Mh0tRBXZw1o=7.815091e+0",
	{Min, 2 ;; 4, "Input" -> {2, 3, 4, 5}} -> "2_GqoHJ9qF/kA_DbipFFm7GD4=1.019921e-1",
	{Min, All, "Input" -> {2, 3, 4, 5}} -> "_MXCqZlxDpmk_H8VSI6gnbYE=5.032360e-2",
	{Total, 2, "Input" -> {2, 3, 4, 5}} -> "2*4*5_W/VeLg8W3tI_CIhDl0KDlng=5.521994e+1",
	{Total, 2 ;; 4, "Input" -> {2, 3, 4, 5}} -> "2_TBBSEw0EsCs_V8N1alSIt4w=5.521994e+1",
	{Total, All, "Input" -> {2, 3, 4, 5}} -> "_EQTJ9fJWDKU_DQgy7R++flY=5.521994e+1",
	{Mean, 2, "Input" -> {2, 3, 4, 5}} -> "2*4*5_CSxuhvkFQ1A_JugqrYQktJI=1.840665e+1",
	{Mean, 2 ;; 4, "Input" -> {2, 3, 4, 5}} -> "2_COK82RpC65E_P42WvM4e1sQ=9.203323e-1",
	{Mean, All, "Input" -> {2, 3, 4, 5}} -> "_CQg8sA0YP2o_QJY9TRBT71A=4.601661e-1",
	{Times, 1, "Input" -> 4} -> "_fz/kclNRCu8_YZOtXT5IFgo=1.335252e-2",
	{Times, All, "Input" -> {2,3}} -> "_UuOZJcPxKlE_ahaRjch5ncs=1.471308e-4",
	{Min, 2, "Input" -> {2, 3, 4, 5, Restricted["Integer", 4]}} -> "2*4*5_YOPpBW7VSo8_GL+hupc9SG4=6.000000e+1",
	(* Varying dimension, but no aggregation on the varying dimension *)
	{Min, {2, 3, 4}, "Input"-> {"Varying", 2, 2, 2}} -> "3_M5vdAy8tD28_Wt53IVPHiJ8=4.035749e-1",
	{Mean, 2, "Input"-> {"Varying", 2, 2}} -> "3*2_KrE9RktIUQs_D4FAZ2JrIiQ=2.615769e+0",
	(* Varying dimension, with aggregation on the varying dimension *)
	{Total, 1, "Input"-> {"Varying", 5}} -> "5_PKPPoQhOYH8_DPDY3uzyuvY=6.320056e+0",	
	{Total, 1, "Input"-> {"Varying"}} -> "_bt+kDbKWQgU_Jl+ATQs3ShI=9.048177e-1",
	{Max, 1, "Input"-> {"Varying", 3, 2}} -> "3*2_SRJlQDnC1C4_MQRYSDvVBXE=3.944532e+0",
	{Min, 1, "Input"-> {"Varying", 2, 3}} -> "2*3_aEUH50S32p0_AE2BQpCL3ow=1.255258e+0",
	{Min, 1, "Input"-> {"Varying"}} -> "_akQbzL2Wru4_cta11WTG9UE=1.119578e-1",
	{Times, 1, "Input"-> {"Varying", 2, 3}} -> "2*3_XapFZmlcl0Y_IpuB91P5+zs=4.401449e-1",
	{Times, 1, "Input"-> {"Varying"}} -> "_MjWJ21mnkbc_SFjlx4BNgmw=1.735165e-2",
	{Mean, 1, "Input"-> {"Varying", 5}} -> "5_e2y16OBnVwY_OKqBkZaVg5E=2.106685e+0",
	{Mean, 1, "Input"-> {"Varying"}} -> "_dUqbYdb+LVU_FpATOHyzJqg=3.016059e-1",
	{Total, {1, 2}, "Input"-> {"Varying", 5}} -> "_TxHHs45rlEs_CcBwAwKJtuQ=6.320056e+0",
	{Times, {1, 2}, "Input"-> {"Varying", 2, 3}} -> "3_d2Ybp4bk53U_WKJvDqD+3Zw=8.577791e-3",
	{Min, {1, 3, 4}, "Input"-> {"Varying", 4, 3, 2}} -> "4_b+PcK/RnWLo_Cqnki2jzqR4=2.545180e-1",
	{Mean, {1, 2}, "Input"-> {"Varying", 5}} -> "_ITE5ulEqMKI_NOhHHaaEvcU=4.213370e-1",
	{Mean, {1, 3}, "Input"-> {"Varying", 2, 3, 4}} -> "2*4_e8reMTum1Cw_RZGPow4IJjE=3.806305e+0",
	{Max, {}, "Input" -> {2, 3}} -> "2*3_XNJtjBjJ1rk_Luzvstmun4A=1.911142e+0",
	(* Cases of error *)
	{x, "Input" -> {2, 3, 4, 5}} -> "Value given for the function (first argument) should be either Mean, Min, Max, Total, or Times, but was x instead.",
	{Max, {0, 2}, "Input" -> {2, 3, 4, 5}} -> "Value of {0, 2} given for the levels (second argument) was invalid: level specification was not a non-zero integer, a span, All, or a list of integers or spans."
}

Upgraders: {
	"11.3.2" -> DropAllHiddenParams
}	