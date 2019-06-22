Input: ChannelT[SizeT, AnyTensorT]

Output: AnyTensorT

Parameters:
	$Specification: ValidatedParameterT[checkPartSpec, Automatic]

ShapeFunction: Map1[PartShape[#1, ToList @ StripVP @ $Specification]&]

RankFunction: Map1[PartRank[#1, ToList @ StripVP @ $Specification]&]

TypeFunction: Identity

AllowDynamicDimensions: True

checkPartSpec[s_List] := Map[checkPartSpec, s];

checkPartSpec[s:Span[i:(_Integer|All), j:(_Integer|All)]] := Which[
	j === 0 || i === 0, 
		FailValidation[PartLayer, "part specification must be non-zero."],
	TrueQ[i > j && Positive[i*j]],
		FailValidation[PartLayer, "span is empty.", s],
	True,
		s /. All -> -1
];

checkPartSpec[i_Integer] := If[i === 0, 
	FailValidation[PartLayer, "part specification must be non-zero."],
	i
];

checkPartSpec[All] := 1 ;; -1;

checkPartSpec[i_] := FailValidation[PartLayer, "`` is an invalid part specification.", i];

(* eventually: use mx.slice, once None is supported
	https://github.com/apache/incubator-mxnet/issues/8153 *)
(* partToNumpySlice[in_Integer] := {0, -1};
partToNumpySlice[Span[a_, b_]] := If[# < 0, #, # - 1]& /@ {a, b};
partToNumpySlice[spec_List] := Transpose[partToNumpySlice /@ spec]; *)
(* partToNumpySlice[rank_, spec_List] := If[# < 0, #, # - 1]& /@ {a, b} *)

Writer: Function @ Scope[
	spec = ToList @ StripVP @ #Specification;
	idims = GetInputDims["Input"];
	odims = GetOutputDims["Output"];
	input = GetInput["Input"];
	sliced = input;
	start = If[idims[[1, 0]] === LengthVar, 2, 1];
	Do[
		idim = idims[[i]];
		spec2 = spec[[i]] /. k_Integer ? Negative :> (k + 1 + idim);
		sliced = Match[spec2,
			Span[1, idim] :> Continue[],
			Span[a_, b_] :> SowTake[sliced, {a-1, b}, i],
			a_Integer :> SowTake[sliced, {a-1, a}, i]
		]
		,
		{i, start, Length[spec]}
	];
	If[!FreeQ[spec, _Integer],
		sliced = SowReshape[sliced, odims /. lv_LengthVar -> 0]
	];
	SetOutput["Output", sliced];
]

Tests: {
	{1, "Input" -> {3, 3, 3}} -> "3*3_YNBbVYLSHUk_Purc7467SgU=3.210989e+0",
	{3, "Input" -> {3, 3, 3}} -> "3*3_cQuubcUVLKw_amG3uZozvmA=4.306343e+0",
	{3, "Input" -> {5}} -> "_akQbzL2Wru4_SGnsYBYFcVs=1.119578e-1",
	{-5, "Input" -> {7}} -> "_akQbzL2Wru4_H57BCjL1SF4=1.119578e-1",
	{2 ;; 4, "Input" -> {5}} -> "3_cwgaM9SDvGc_Avbt9Hpc5tY=1.324527e+0",
	{1 ;; 2, "Input" -> {3, 3, 3}} -> "2*3*3_Qitdrci69/U_N6HuS/ImClk=7.379773e+0",
	{2 ;; 2, "Input" -> {3, 3, 3}} -> "1*3*3_USjn1b6jTHw_VnjfaAO82oo=4.168784e+0",
	{2 ;; 2, "Input" -> {3, 3, 3, Restricted["Integer", 10]}} -> "1*3*3_QiknLMs69Jg_DVDXS71AhM0=4.700000e+1",
	{-2 ;; All, "Input" -> {3, 3, 3}} -> "2*3*3_OlQBee2x8QY_ZHQb7ZWMjVM=8.475127e+0",
	{1 ;; All, "Input" -> {3, 3, 3}} -> "3*3*3_O/ez3TErae8_bl6p06Ct7OM=1.168612e+1",
	{4, "Input" -> {3, 3, 3}} -> "Type inconsistency in PartLayer: the specification 4 cannot reference positions greater than 3.",
	{3 ;; 2, "Input" -> {3, 3, 3}} -> "Value of Span[3, 2] given for the specification (first argument) was invalid: span is empty.",
	{{2, 1 ;; 3, -2 ;; -1}, "Input" -> {4, 5, 2}} -> "3*2_Y1c8ewOeiwI_AlFuQKy/wfs=2.893619e+0",
	{{2, 1, -2}, "Input" -> {4, 5, 2}} -> "_X60SkhiiT3Y_caz4ZlwLVLk=3.652418e-1",
	{{1 ;; All, 1 ;; All, -2}, "Input" -> {4, 5, 2}} -> "4*5_EkKQvfNxHHM_ILupXGmR+UY=7.416406e+0",
	{{1 ;; All, 1 ;; 6, 2}, "Input" -> {4, 5, 2}} -> "Type inconsistency in PartLayer: the specification 6 cannot reference positions greater than 5.",
	{{1, 2, 0 ;; 2}, "Input" -> {4, 5, 2}} -> "Value of {1, 2, Span[0, 2]} given for the specification (first argument) was invalid: part specification must be non-zero.",
	{{Sin ;; 2}, "Input" -> {3}} -> "Value of {Span[Sin, 2]} given for the specification (first argument) was invalid: Span[Sin, 2] is an invalid part specification.",
	{{1, 2, -1 ;; 1}, "Input" -> {4, 5, 2}} -> "Type inconsistency in PartLayer: negative range 2;;1 not supported.",
	{{2 ;; 3, 2 ;; 4}, "Input" -> {6, 5}} -> "2*3_Btbq0CTajng_AnT892+1nL4=2.629385e+0",
	{{3, 2 ;; 3, 2 ;; 4}, "Input" -> {3, 6, 5}} -> "2*3_E9CxNf73hUI_LiMCh0fgldU=3.083208e+0",
	{{2, 3}, "Input" -> {6, 5}} -> "_Q599EIylf18_LYRaTwPfyfc=8.371853e-1",
	{{2 ;; 4}, "Input" -> {5}} -> "3_cwgaM9SDvGc_Avbt9Hpc5tY=1.324527e+0",
	{{2, 1 ;; 3, 1 ;; All}, "Input" -> {4, 5, 2}} -> "3*2_Y1c8ewOeiwI_AlFuQKy/wfs=2.893619e+0",
	{{2, 1, 1}, "Input" -> {4, 5, 2}} -> "_X60SkhiiT3Y_caz4ZlwLVLk=3.652418e-1",
	{{1 ;; 4, 1 ;; 5, 1}, "Input" -> {4, 5, 2}} -> "4*5_EkKQvfNxHHM_ILupXGmR+UY=7.416406e+0",
	{{All, 2}, "Input" -> {3, 5, 5}} -> "3*5_I7i7t91L+5k_X8oXA7lYPN8=6.985818e+0",
	(* dynamic dimensions: *)
	{3, "Input" -> {"n", 3}} -> "Type inconsistency in PartLayer: cannot take part of a dynamic dimension.",
	{All, "Input" -> {"n", 3}} -> "3*3_YNBbVYLSHUk_d4RU0KGWjWs=3.210989e+0", (* <- allowed, no-op *)
	{{All, 1}, "Input" -> {"n", 3}} -> "3_XYBDxdGz1Cw_UYGKq0GvnN4=1.480030e+0"
}

upgradePart[params_] := Scope[
	spec = params["Specification"];
	spec = ValidatedParameter[spec[[1]]];
	Append[params, "Specification" -> spec]
];

Upgraders: {
	"11.3.3" -> ApplyParams[upgradePart],
	"11.3.1" -> RenameParam["PartSpecification" -> "Specification"],
	"11.3.2" -> DropAllHiddenParams
}
