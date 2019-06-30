Input: AnyTensorT

Output: RealTensorT

ShapeFunction: Identity

RankFunction: Identity

TypeFunction: Function[{RealT}]

Parameters:
	$Level: Defaulting[IntegerT, -1]

MaxArgCount: 1

toActualLevel[level_, rank_] := If[level < 0, rank + 1 + level, level];

PostInferenceFunction: Function @ Scope[
	idims = TDimensions[$Input];
	rank = TRank[$Input];
	If[idims === {}, 
		FailValidation["input should be a vector or higher-rank array, not a scalar."]];	
	If[$Level == 0, 
		FailValidation["level should be non-zero."]];
	If[IntegerQ[rank] && Abs[$Level] > rank, 
		FailValidation["absolute value of level should be less than ``.", rank]];
	If[IntegerQ[rank] && Part[idims, toActualLevel[$Level, rank]] === 1, 
		FailValidation["SoftmaxLayer[`1`] will produce a constant output when the input is ``.", $Level, MsgForm[$Input]]
	];
]

AllowDynamicDimensions: True

Writer: Function @ Scope[
	input = GetInputMetaNode["Input"];
	level = toActualLevel[#Level, GetInputRank["Input"]];
	output = SowMetaSoftmax[input, level];
	SetOutput["Output", output];
]

MXNet:
	Name: "softmax"
	Aliases: {"SoftmaxActivation"}
	Reader: Function @ Scope[
		level = ToExpression[Replace[#["axis"], _Missing -> -1]];
		If[level === 0, FailImport["MXNet", "softmax", "can't import when axis parameter is set to 0."]];
		"Level" -> level
	]

Upgraders: {
    "11.3.0" -> AddParam["Level" -> -1]
}

(* see the comment LayerTests.m about testing of layers that employ SowSeqMask *)

Tests: {
	(* Classical softmax, on the last dimension *)
	{"Input" -> {3}} -> "3_B7HUGnzh1Fk_Czr4LI/q+yE=1.000000e+0",
	{"Input" -> {3, 2}} -> "3*2_MKuq5+IjuZw_aDXMxKIKoOg=3.000000e+0",
	{"Input" -> {2, 3, 4}} -> "2*3*4_XoaIjPheTCw_UcIcNwjE4rs=6.000000e+0",
	{"Input" -> {"Varying", 4}} -> "3*4_eYnrf3zNzus_CknCSjCQIQM=3.000000e+0",
	{"Input" -> {"Varying", 4, "Integer"}} -> "3*4_bkw92widEWo_JUWyFihHUjg=3.000000e+0",
	{"Input" -> {"Varying", 4, Restricted["Integer", 10]}} -> "3*4_G2ut4iWBQQk_AjKD7KxFvoo=3.000000e+0",

	{"Input" -> {"Varying"}} -> "3_B7HUGnzh1Fk_Z3kqJcpWf5w=1.000000e+0",
	{1, "Input" -> {3}} -> "3_B7HUGnzh1Fk_Czr4LI/q+yE=1.000000e+0",
	{-1, "Input" -> {3}} -> "3_B7HUGnzh1Fk_Czr4LI/q+yE=1.000000e+0",
	{-1, "Input" -> {3, 2}} -> "3*2_MKuq5+IjuZw_aDXMxKIKoOg=3.000000e+0",
	{3, "Input" -> {2, 3, 4}} -> "2*3*4_XoaIjPheTCw_UcIcNwjE4rs=6.000000e+0",
	{2, "Input" -> {"Varying", 4}} -> "3*4_eYnrf3zNzus_CknCSjCQIQM=3.000000e+0",
	{1,"Input" -> {"Varying"}} -> "3_B7HUGnzh1Fk_Z3kqJcpWf5w=1.000000e+0",
	{-1,"Input" -> {"Varying"}} -> "3_B7HUGnzh1Fk_Z3kqJcpWf5w=1.000000e+0",

	(* Softmax on the first dimension *)
	{1, "Input" -> {10, 3, 1}} -> "10*3*1_TR33yK5iSjQ_J0JbXDxdttU=3.000000e+0",
	{1, "Input" -> {"Varying", 4}} -> "3*4_KJJcLFzZMXU_YXtKHwLzr4w=4.000000e+0",
	{1, "Input" -> {"Varying", 4, 2, 3}} -> "3*4*2*3_P7nSywJ55vA_Mik3h+uxl4M=2.400000e+1",
	{-4, "Input" -> {"Varying", 4, 2, 3}} -> "3*4*2*3_P7nSywJ55vA_Mik3h+uxl4M=2.400000e+1",

	(* Softmax on a dimension in the middle *)
	{2, "Input" -> {5, 4, 2, 3}} -> "5*4*2*3_Oj8+sojsXvQ_GQqYfj7mRPE=3.000000e+1",
	{2, "Input" -> {"Varying", 4, 2, 3}} -> "3*4*2*3_Q6FKVebKmnI_fOdVFwzOqqc=1.800000e+1",
	{-3, "Input" -> {"Varying", 4, 2, 3}} -> "3*4*2*3_Q6FKVebKmnI_fOdVFwzOqqc=1.800000e+1",

	(* Cases of errors *)
	{4, "Input" -> {2, 3, 4}} -> "Validation failed for SoftmaxLayer: absolute value of level should be less than 3.",
	{-4, "Input" -> {2, 3, 4}} -> "Validation failed for SoftmaxLayer: absolute value of level should be less than 3.",
	{3, "Input" -> {2, 3, 1}} -> "Validation failed for SoftmaxLayer: SoftmaxLayer[3] will produce a constant output when the input is a 2\[Times]3\[Times]1 array of real numbers.",
	{"Input" -> {}} -> "Validation failed for SoftmaxLayer: input should be a vector or higher-rank array, not a scalar."
}