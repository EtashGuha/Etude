Input: InterleavingSwitchedT[$Interleaving, $$InputChannels, $$InputSize, AtomT]

Output: InterleavingSwitchedT[$Interleaving, $OutputChannels, $$OutputSize]

Arrays:
	$Weights: TensorT[{$$InputChannels, $$WeightsOutputChannels}, TensorT[$KernelSize]]
	$Biases: Nullable[VectorT[$OutputChannels]]

Parameters:
	$OutputChannels: 	SizeT
	$KernelSize:  		ArbSizeListT[$Dimensionality, PosIntegerT, None]
	$Stride: 			ArbSizeListT[$Dimensionality, PosIntegerT, 1]
	$PaddingSize:		PaddingSizeT[$Dimensionality, 0]
	$Dimensionality:	NaturalT
	$Interleaving:		Defaulting[BooleanT, False]
	$ChannelGroups: 		Defaulting[SizeT, 1]
	$$Dilation:			ArbSizeListT[$Dimensionality, PosIntegerT, 1] (* Dilation is broken on GPU *)
	$$InputChannels: 	SizeT
	$$InputSize: 		SizeListT[$Dimensionality]
	$$OutputSize: 		ComputedType[SizeListT[$Dimensionality], DeconvolutionShape[$$InputSize, $PaddingSize, $KernelSize, $Stride, $$Dilation]]
	$$WeightsOutputChannels: ComputedType[SizeT, $OutputChannels / $ChannelGroups, {$OutputChannels}]

ReshapeParams: {$$InputChannels, $$InputSize, $$OutputSize}

MinArgCount: 0
PosArgCount: 2

PostInferenceFunction: Function[
	CheckDeconvolutionFunction[$Dimensionality, $$InputSize, $KernelSize, $$OutputSize, $PaddingSize];
	CheckGroupNumberFunction[DeconvolutionLayer, $ChannelGroups, $$InputChannels, $OutputChannels]
]

Writer: Function @ Scope[
	If[#Interleaving === False, MXWriteDefaultAndReturn[]];
	input = GetInput["Input"];
	output = SowTransposedConvolutionOrPooling[#Dimensionality, input, #Weights, #Biases];
	SetOutput["Output", output];
]

MXNet:
	Name: "Deconvolution"
	Parameters: 
		$OutputChannels: "num_filter"
		$KernelSize: "kernel"
		$PaddingSize: "pad"
		$$Dilation: "dilate"
		$Stride: "stride"
		$ChannelGroups: "num_group"
	Arrays:
		$Weights: "weight"
		$Biases: "bias"
	Writer: Function[{
		"pad" -> writeIntList[#PaddingSize[[All, 1]]], (* Pad spec is assumed to be symmetric for now *)
		"no_bias" -> If[#2["Biases"] === None, "True", "False"]
	}]

Tests: {
	(* 1-d *)
	{2, {2}, "Input" -> {1, 16}} -> "2*17_KFh9X2MDjHY_CTmB/wmCU2k=2.794851e+1",
	{3, {3}, "Input" -> {3, 7}, "Stride" -> {2}, PaddingSize -> {2}} -> "3*11_POnpys5R/NE_LLMsC8mH0U0=1.848940e+1",
	{5, 3, "Stride" -> 2, "Input" -> {3, 1}, "Interleaving" -> True} -> "7*5_NjcaynTJ7Uw_eJ2XuZR2bgg=3.895047e+1",
	{3, {3}, "Input" -> {3, 7}, "Stride" -> {2}, "ChannelGroups" -> 3} -> "3*15_ccsMwTy2Svs_Wi0jcqp81c0=3.287278e+1",
	(* 2-d *)
	{3, {3, 2}, "Input" -> {3, 7, 5}, "Stride" -> {2, 1}, PaddingSize -> {2, 1}} -> "3*11*4_fcZfgAUTbus_Vvh6lEyrX8c=2.228037e+2",
	{2, 2, "Input" -> {2, 4, 4}} -> "2*5*5_Bk513pzhME4_CEVFTnDDS0Y=8.473589e+1",
	{1, 3, "Input" -> {2, 4, 4}} -> "1*6*6_XI4iSnWjXW0_XpcH+OwOSyc=8.442257e+1",
	{2, 2, "Stride" -> 2, "Input" -> {2, 8, 8}} -> "2*16*16_S+ExxOi8lbM_NEwjkXYMdjc=6.925425e+2",
	{2, 3, "ChannelGroups" -> 2, "Input" -> {4, 5, 5}} -> "2*7*7_NIBeiumbYJA_aSmbuRmPGLE=9.500083e+1",
	{2, 3, "ChannelGroups" -> 2, "Input" -> {4, 5, 5, Restricted["Integer", 2]}} -> "2*7*7_Wqd6t7TldOk_SSHseurjlWY=2.852916e+2"
}

Upgraders: {
	"11.3.1" -> RenameParam["InputChannels" -> "$InputChannels"],
	"11.3.4" -> AddParam["Interleaving" -> False],
	"11.3.7" -> RenameParam["$GroupNumber" -> "ChannelGroups"] /* AddParam["Dimensionality" -> 2] /* AddParam["$Dilation" -> {1, 1}] /* AddParam[Function[
		outputCh = #Parameters["OutputChannels"];
		groupN = #Parameters["ChannelGroups"];
		"$WeightsOutputChannels" -> If[IntegerQ[outputCh], outputCh / groupN, SizeT]
	]],
	"11.3.8" -> UpgradeAsymmetricPadding
}