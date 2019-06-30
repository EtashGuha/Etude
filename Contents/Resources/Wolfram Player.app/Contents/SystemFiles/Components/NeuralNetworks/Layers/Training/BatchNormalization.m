Input: InterleavingSwitchedT[$Interleaving, $Channels, $$SpatialDimensions, AtomT]

Output: InterleavingSwitchedT[$Interleaving, $Channels, $$SpatialDimensions]

Parameters:
	$Momentum: Defaulting[ScalarT, 0.9]
	$Epsilon: Defaulting[ScalarT, 0.001]
	$Interleaving: Defaulting[BooleanT, False]
	$Channels: SizeT
	$$SpatialDimensions: SizeListT[]

Arrays:
	$Scaling: VectorT[$Channels]
	$Biases: VectorT[$Channels]
	$MovingMean: VectorT[$Channels]
	$MovingVariance: VectorT[$Channels]

PosArgCount: 0

AuxArrays: {"MovingMean", "MovingVariance"}

AllowDynamicDimensions: True

ArgumentRewriter: RenameDeprecatedOption[{"Gamma" -> "Scaling", "Beta" -> "Biases"}]

FinalCheck: Function[
	CheckNoDynamicChannels[BatchNormalizationLayer, $Channels];
	(* ^ we leave the complaint about dynamic dims to just before eval, because dynamic dims can occur in the net
	temporarily and go away eventually during inference *)	
]

HasTrainingBehaviorQ: Function[True]

Writer: Function @ Scope[

	If[!$TMode || !DynamicDimsQ[#$SpatialDimensions], MXWriteDefaultAndReturn[]];

	(* for dynamic dimension we are forced into some contortions, specifically because of the 
	the aux arrays for the moving means and variances.
	specifically, we are forced to use the built-in BN layer because its the only one capable of
	writing to the aux mean and variance arrays. we have one BN layer that does *just* that, where
	we trick it into calculating the proper sequence-aware (i.e. masked) mean and variance, and then
	we calculate the true BN ourselves (though we reuse the variance calculation from the first BN
	layer) *)

	input = GetInput["Input"]; 

	lnode = GetDynamicLengthNode @ First @ #$SpatialDimensions;
	factor = SowSqrt @ SowDivide[SowNodeLength[input], SowNode["mean", lnode]];
	(* ^ calculate the correction factor we'll apply to increase the dispersion 
	of the inputs by an amount that compensates for number of zero'd entries *)

	input = SowSeqMaskBatchwise[input, lnode];
	sum = SowNode["sum", input, "axis" -> "-1", "exclude" -> "true"];
	elemsPerChannel = SowTimesScalar[SowSumAxis[lnode, 0], Apply[Times, Rest @ #$SpatialDimensions]];
	mean = SowBDivide[sum, elemsPerChannel];
	(* ^ calculate the mean value per-channel, ignoring masked elems *)

	centered = SowSeqMaskBatchwise[SowBMinus[input, mean], lnode];
	input2 = SowBPlus[SowBHad[centered, factor], mean];
	(* ^ replace the masked entries with the mean element value, and scale away from the mean
	by the correction factor so that dummyBN will calculate the correct moving variance *)

	dummyBN = SowNode["BatchNorm", 
		{input2, #Scaling, #Biases, #MovingMean, #MovingVariance},
		"eps" -> #Epsilon, "momentum" -> #Momentum,
		"fix_gamma" -> "false", "use_global_stats" -> "false", 
		"output_mean_var" -> "true", "axis" -> "-1",
		"cudnn_off" -> If[#Epsilon < 10^-5, "1", "0"] (* workaround for cuDNN limitation! *)
	];
	blocked = SowBlockGrad @ dummyBN;
	BagPush[$HiddenOutputNodes, blocked];
	(* ^ use the dummy batchnorm layer to compute the mean and variance
	and to update the running averages in the aux arrays *)

	invstddev = NthOutput[dummyBN, 2];
	output = bnCore[centered, stddev, #Scaling, #Biases, #Epsilon];
	(* ^ use the dummy BN's instantaneous mean and var to compute the
	BN output manually. its not actually a var, its an inverse stddev + epsilon, 
	but that suits us just fine *)

	SetOutput["Output", output];
]

bnCore[centeredInput_, stddev_, gamma_, beta_, epsilon_] := Scope[
	gammaBySD = SowHad[gamma, invstddev];
	SowBPlus[SowBHad[centeredInput, gammaBySD], beta]
];

MXNet:
	Name: "BatchNorm"
	Parameters: 
		$Epsilon: "eps"
		$Momentum: "momentum"
	Writer: Function[{
		"fix_gamma" -> "false", 
		"use_global_stats" -> "false",
		"axis" -> If[#Interleaving, "-1", "1"],
		"cudnn_off" -> If[#Epsilon < 10^-5, "1", "0"] (* workaround for cuDNN limitation! *)
	}]
	Arrays:
		$Scaling: "gamma"
		$Biases: "beta"
		$MovingMean: "moving_mean"
		$MovingVariance: "moving_var"

Tests: {
	{"Input" -> "Real"} -> "Specification Real is not compatible with port \"Input\", which must be an array of rank \[GreaterEqual] 1.",
	{"Input" -> 3} -> "3_QemtvEXs7rE_M2JMbKSj9a8=3.404268e+0",
	{"Input" -> {3, 6}} -> "3*6_Aedcp62lnek_IVmV8TiIhPU=1.990227e+1",
	{"Input" -> {3, 7, 3}} -> "3*7*3_TsTm3+hkNt8_NoFytlK/iUk=6.370230e+1",
	{"Input" -> {3, 2, 2}} -> "3*2*2_cOh7d9zooj4_LLss5sGz23w=1.210545e+1",
	{"Input" -> {3, 6, 5, 4}} -> "3*6*5*4_foPlhN3iLLE_PMHI9vslsaQ=3.653435e+2",
	{"Input" -> {3, 6, 5, 4, Restricted["Integer", 4]}} -> "3*6*5*4_T1LCQXZcLr8_PfcHEbZk6JM=8.859579e+2",
	{"Epsilon" -> 10^(-6), "Input" -> {3}} -> "3_QemtvEXs7rE_M2JMbKSj9a8=3.404258e+0",
	{"Epsilon" -> 10^(-5), "Input" -> {3}} -> "3_QemtvEXs7rE_M2JMbKSj9a8=3.404259e+0",
	{"Input" -> {"n", 3}, "Interleaving" -> False} -> "Validation failed for BatchNormalizationLayer: the number of channels cannot be dynamic. To use a dynamic spatial dimension, try using Interleaving -> True.",
	{"Input" -> "n", "Interleaving" -> True} -> "Validation failed for BatchNormalizationLayer: the number of channels cannot be dynamic. To use a dynamic spatial dimension, try using Interleaving -> True.",
	(* ^ variable number of channels is not supported because it makes Scaling variable-size *)

	{"Input" -> {"n", 3}, "Interleaving" -> True} -> "3*3_LCtk3zrLzKE_ESA2/FL5ZuI=9.745958e+0",
	{"Input" -> {"n", 3, 2}, "Interleaving" -> True} -> "3*3*2_XLVothVxVHo_VFERjTuzBRQ=2.738866e+1",
	{"Input" -> {"n", 4, 3, 2}, "Interleaving" -> True} -> "3*4*3*2_UIKcXBgQJso_Gw+JsK2PB98=1.029279e+2"
}

Upgraders: {
	"11.3.5" -> ApplyParams[toSpatialDims],
	"11.3.7" -> RenameArray["Gamma" -> "Scaling"] /* RenameArray["Beta" -> "Biases"]
}

toSpatialDims = Function[
	Append[
		KeyDrop[#, "$Shape"], {
		"Interleaving" -> False,
		"$SpatialDimensions" -> Replace[TDimensions[#$Shape], {
			list_List :> Rest[list],
			_ :> SizeListT[]
		}]
	}]
];