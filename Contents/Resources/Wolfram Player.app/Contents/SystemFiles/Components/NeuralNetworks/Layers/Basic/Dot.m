Inputs: 
	$Multiport: RealTensorT (* TODO: handle integers *)

Output: RealTensorT

ShapeFunction: List[DotShape[#]]&

RankFunction: List[DotRank[#]]&

Writer: Function @ Scope[
	inputs = GetInput[All]; dims = GetInputDims[All];

	If[MatchQ[dims, {{_, _}, {_, _}}],
		output = SowGEMM2[First @ inputs, Last @ inputs];
		,
		res = Fold[sowBatchDot, Thread[inputs -> dims]];
		output = SowReshape[res[[1]], res[[2]]];
	];
	
	SetOutput["Output", output];
]

sowBatchDot[a_ -> {m_, n_}, b_ -> {n_, k_}] := 
	SowGEMM2[a, b] -> {m, k};

sowBatchDot[a_ -> adims_, b_ -> bdims_] := 
	SowGEMM2[flattenFront[a, adims], flattenBack[b, bdims]] -> Join[Most[adims], Rest[bdims]];

flattenFront[a_, dims_] := SowNode["reshape", a, "shape" -> {0, -1, Last[dims]}];
flattenBack[a_, dims_] := SowNode["reshape", a, "shape" -> {0, First[dims], -1}];

MXNet:
	Name: "batch_dot"

Tests: {
	{"Input" -> {3, 3}} -> "_YMJo2dfNev4_flD+uz32U+w=3.167709e-1",
	{"Input" -> {{2, 3}, 3}} -> "2_QTLgdUUrczU_cGTSnF9S108=8.569903e-1",
	{"Input" -> {3, {3, 2}}} -> "2_WASBZ9NBC8E_JHfv5hpgJWI=6.331198e-1",
	{"Inputs" -> {{3, 4}, {4, 3}}} -> "3*3_NbT+zsJwRxM_VIILzLx+gMc=7.009739e+0",
	{"Inputs" -> {{3, 4}, {4, 3}, {3}}} -> "3_R18MJIvS0OU_fzG8dLdMML8=2.648691e+0",
	{"Inputs" -> {{3, 4, 5}, {5, 4, 3}}} -> "3*4*4*3_P1h8d5JsmX4_UUVs+ubHF3M=1.526926e+2"
	(* TODO {"Inputs" -> {{3, 4, 5, "Integer"}, {5, 4, 3, "Integer"}}} -> "3*4*4*3_PTSqiY2I3nw_N4cxuU+bzrs=1.011380e+11" *)
}

Upgraders: {
	"11.3.2" -> UpgradeToMultiport
}

