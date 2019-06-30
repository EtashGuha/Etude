Input: InterleavingSwitchedT[$Interleaving, $$InputChannels, $$InputSize, AtomT]

Output: InterleavingSwitchedT[$Interleaving, $OutputChannels, $$OutputSize]

Arrays:
	$Weights: TensorT[{$OutputChannels, $$WeightsInputChannels}, TensorT[$KernelSize]]
	$Biases: Nullable[VectorT[$OutputChannels]]

Parameters:
	$OutputChannels:	SizeT
	$KernelSize:		ArbSizeListT[$Dimensionality, SizeT, None]
	$Stride:			ArbSizeListT[$Dimensionality, PosIntegerT, 1]
	$PaddingSize:		PaddingSizeT[$Dimensionality, 0]
	$Dilation:			ArbSizeListT[$Dimensionality, PosIntegerT, 1]
	$ChannelGroups:		Defaulting[SizeT, 1]	
	$Dimensionality:	NaturalT
	$Interleaving:		Defaulting[BooleanT, False]
	$$InputChannels:	SizeT
	$$InputSize:		SizeListT[$Dimensionality]
	$$OutputSize:		ComputedType[SizeListT[$Dimensionality], 
		MaybeDyn @ ConvolutionShape[$$InputSize, $PaddingSize, $KernelSize, $Stride, $Dilation]
	]
	$$WeightsInputChannels: ComputedType[SizeT, $$InputChannels / $ChannelGroups, {$$InputChannels}]

ReshapeParams: {$$InputChannels, $$InputSize, $$OutputSize, $$WeightsInputChannels}

MinArgCount: 0
PosArgCount: 2

PostInferenceFunction: Function[
	CheckConvolutionOrPoolingFunction[ConvolutionLayer, $Dimensionality, $$InputSize, $KernelSize, $$OutputSize, $PaddingSize, $Dilation];
	CheckGroupNumberFunction[ConvolutionLayer, $ChannelGroups, $$InputChannels, $OutputChannels]
]

FinalCheck: Function[
	CheckNoDynamicChannels[ConvolutionLayer, $$InputChannels]
	(* ^ we leave the complaint about dynamic dims to just before eval, because dynamic dims can occur in the net
	temporarily and go away eventually during inference *)
]

AllowDynamicDimensions: True

Constraints: SpatialConstraintGenerator[getFirstConvOut]

getFirstConvOut = Function[
	ConvolutionShape[#2, First @ #PaddingSize, First @ #KernelSize, First @ #Stride, First @ #Dilation]
];

(* currently, only cuDNN backend supports different layouts. 
Also: 1d + 3d conv don't seem supported, the latter crashes.
Also: cuDNN doesn't support dilations yet *)
mxConvLayout = Function[
	If[#Interleaving && (#Dimensionality == 2) && $GPUMode && ($DTypeMXName === "float32") && (Max[#Dilation] == 1), 
		If[#Dimensionality == 2, "NHWC", "NDHWC"]
		,
		"None"
	]
];

Writer: Function @ Scope[
	node = GetInput["Input"];
	fdim = First @ GetInputDims["Input"];
	lnode = GetDynamicLengthNode[fdim];

	SowDerivedSequenceLengthNode[lnode, First @ GetOutputDims["Output"], Function[x, getFirstConvOut[#, x]]];
	If[lnode =!= None, 
		node = SowSeqMaskBatchwise[node, lnode]
	];

	(* this is used instead of MXNET_CUDA_ALLOW_TENSOR_CORE as there is bug in mxnet:
		cuDNN Find doesn't respect this env variable. Remove this logic once its fixed *)
	bias = #Biases;
	weight = #Weights;
	If[$MixedPrecisionQ, 
		{node, weight} = SowCast[{node, weight}, $DType, "Real16"];
		If[bias =!= None, bias = SowCast[bias, $DType, "Real16"]];
		If[lnode =!= None, lnode = SowCast[lnode, $DType, "Real16"]];
	];

	(* ASYMMETRIC PADDING *)
	symmetricPad = Apply[SameQ, Transpose[#PaddingSize]];
	If[!symmetricPad, 
		{node, workaround} = ApplyAsymmetricConvPoolPadding[node, #Dimensionality, #Interleaving, #PaddingSize]
	];
	(* CONVOLUTION *)
	(* For clarity: #Interleaving === False || (#Interleaving === True && workaround === "Transpose") *)
	If[#Interleaving === False || workaround === "Transpose",
		output = SowCurrentNode[{node, weight, Replace[bias, None -> Nothing]}];
		,
		If[mxConvLayout[#] != "None",
			taxes = Join[{1}, Range[3, #Dimensionality + 2], {2}] - 1;
			tweights = SowTranspose[weight, taxes];
			output = SowCurrentNode[{node, tweights, Replace[bias, None -> Nothing]}];
		,
			output = 
				SowTransposedConvolutionOrPooling[#Dimensionality, node, weight, Replace[bias, None -> Nothing]];
		];
	];
	If[workaround === "Transpose", (* For clarity: (#Interleaving === True && workaround === "Transpose") *)
		output = SowTranspose[output, {0, 2, 3, 4, 1}]
	];
	If[$MixedPrecisionQ,  output = SowCast[output, "Real16", $DType]];
	SetOutput["Output", output];
]

MXNet:
	Name: "Convolution"
	Parameters: 
		$OutputChannels: "num_filter"
		$KernelSize: "kernel"
		$Dilation: "dilate"
		$Stride: "stride"
		$ChannelGroups: "num_group"
	Arrays:
		$Weights: "weight"
		$Biases: "bias"
	Reader: Function[
			(* readIntList is not PackageScoped :( *)
			mxPad = ToExpression /@ StringTrim /@ StringSplit[StringTrim[#pad, "(" | ")"], ","];
			"PaddingSize" -> Transpose @ {mxPad, mxPad}
	]
	Writer: Function[{
		(* cudnn convolutions don't support Real64 properly. Disable cuDNN *)
		"cudnn_off" -> If[$DTypeMXName === "float32", "0", "1"],
		"pad" -> writeIntList[#PaddingSize[[All, 1]]], (* Pad spec is assumed to be symmetric or zeros at this point *)
		"no_bias" -> If[#2["Biases"] === None, "True", "False"],
		"layout" -> mxConvLayout[#1]
	}] 

Tests: {
	(* 3-d *)
	{3, {3, 2, 3}, "Input" -> {3, 7, 5, 8}, "Stride" -> {2, 1, 2}, PaddingSize -> {2, 1, 0}} -> "3*5*6*3_QjwsIzYBkx0_Zy/ytfVHNDc=9.012281e+2",
	{2, 3, "Input" -> {3, 5, 5, 5}} -> "2*3*3*3_WXOq2JzKMbM_Re7tJu3Tr0Q=4.308729e+2",
	{2, {2, 3, 4}, "Input" -> {3, 3, 4, 6}} -> "2*2*2*3_YxTQzffuDAo_BOEVBthGWMw=1.384939e+2",
	{6, {2, 2, 2}, "ChannelGroups" -> 2, "Input" -> {2, 3, 4, 6}} -> "6*2*3*5_KLAjdckeE28_EPtgezweSK4=1.896086e+2",
	{3, 2, "Input" -> {2, 3, 4, 5}, "PaddingSize" -> {{1, 2}, {0, 1}, {2, 4}}} -> "3*5*4*10_b5JQ3BnZ8cQ_SgvHVLgtf0k=6.260842e+2",
 	(* 2-d *)
	{3, {3, 2}, "Input" -> {3, 7, 5}, "Stride" -> {2, 1}, PaddingSize -> {2, 1}, "Dilation" -> {2, 1}} -> "3*4*6_aOZ7TREp2CU_NaH4Gd7yaxI=1.352216e+2",
	{2, 2, "Input" -> {2, 4, 4}} -> "2*3*3_B430te9futY_S8Wlsk2ZI0Y=2.738794e+1",
	{1, 3, "Input" -> {2, 4, 4}} -> "1*2*2_XDSMO0pOdfY_IwDq3mBlVa0=1.335831e+1",
	{2, 2, "Stride" -> 2, "Input" -> {2, 8, 8}} -> "2*4*4_ADOzcKP1tl4_bFjVdOX3H9s=5.544535e+1",
	{2, 2, "Dilation" -> 2, "Input" -> {2, 8, 8}} -> "2*6*6_J6F6xTgybso_VpmU0KssAvw=1.280226e+2",
	{6, 2, "Dilation" -> 2, "ChannelGroups" -> 3, "Input" -> {3, 3, 3}} -> "6*1*1_QzGvVb7tnYQ_K8kpKtgM9vU=4.486287e+0",
	{3, 2, "Input" -> {2, 3, 4}, "PaddingSize" -> {{1, 2}, {2, 1}}} -> "3*5*6_QlNfET3j5RE_f4rDlPfvscg=8.439597e+1",
	(* 1-d *)
	{2, {2}, "Input" -> {1, 16}} -> "2*15_e2JXK6TJUSs_YQG1ClW5njs=2.433260e+1",
	{3, {3}, "Input" -> {3, 7}, "Stride" -> {2}, PaddingSize -> {2}, "Dilation" -> {2}} -> "3*4_LSn/XyDpp9o_XYoRX/f2nOY=2.168163e+1",
	{5, 3, "Stride" -> 2, "Dilation" -> 3, "Input" -> {3, 1}, "Interleaving" -> True} -> "Validation failed for ConvolutionLayer: output with non-positive dimensions was inferred.",
	{3, {3}, "Input" -> {3, 7}, "Stride" -> {2}, "ChannelGroups" -> 3, "Dilation" -> {2}} -> "3*2_VZIzAcSPx48_F42DxCBSM/E=3.068234e+0",
		{1, {2, 2}, Interleaving -> True, "PaddingSize" -> {1, 0}, "Input" -> {"x", 5, 1}} -> "11*4*1_JsziQ1oJ64A_J/V04aZUkbk=6.848254e+1",
	{1, {2, 2}, Interleaving -> True, "PaddingSize" -> {1, 0}, "Biases" -> None, "Input" -> {"x", 5, 1}} -> "11*4*1_IP4y9v22z4A_aa294LdBU9I=2.168957e+1",
	{3, 2, "Input" -> {2, 3}, "PaddingSize" -> {{1, 2}}} -> "3*5_b2RaT0753ok_YHQrmwxv03A=1.250973e+1",
	(* varying 1-d *)
	{3, 5, Interleaving -> True, "Input" -> {"x", 1}} -> "6*3_ImnAlquU2Bk_dDFd2zr+1vI=2.822585e+1",
	{1, 2, "Dilation" -> 2, Interleaving -> True, "Input" -> {"x", 1}} -> "8*1_CuLIsm054AY_aozRm6c13DQ=4.509578e+0",
	{1, 2, "Stride" -> 2, Interleaving -> True, "Input" -> {"x", 1}} -> "5*1_dK6N7U+OIvU_SopLEg1jqdk=2.302917e+0",
	{6, 2, "Stride" -> 2, "ChannelGroups" -> 3, Interleaving -> True, "Input" -> {"x", 9}} -> "5*6_G4YpB/Zpx2w_We8VXK1v0Y0=3.208564e+1",
	{1, {2}, Interleaving -> True, "PaddingSize" -> {1}, "Input" -> {"x", 1}} -> "11*1_aC/xH5j61xY_XARBg47/5TY=7.244053e+0",
	{3, 2, "Input" -> {"x", 3}, Interleaving -> True, "PaddingSize" -> {{1, 2}}} -> "12*3_HwamtX4kMM0_fI7iz+Sp9wo=4.792977e+1",
	{3, 5, "Input" -> {"x", 1}} -> "Validation failed for ConvolutionLayer: kernel size 5 cannot exceed input size 1 plus padding size 0.",
	(* varying 2-d *)
	{1, {2, 2}, Interleaving -> True, "Input" -> {"x", 5, 1}} -> "9*4*1_EffSaX0D/8Q_a9qoxWw354Y=5.491642e+1",
	{1, 2, "Stride" -> 2, Interleaving -> True, "Input" -> {"x", 5, 1}} -> "5*2*1_TppveqJVAsc_eoDXzVRBzko=1.575838e+1",
	{2, {2, 3}, "Stride" -> {2, 3}, "PaddingSize" -> {4, 3}, "Dilation" -> {2, 1}, Interleaving -> True, "Input" -> {"x", 7, 6}} -> "8*4*2_BlDVc2S+I/4_eYi41yvWGZI=1.131333e+2",
	{3, {2, 3}, "Stride" -> {2, 3}, "PaddingSize" -> {4, 3}, "Dilation" -> {2, 1}, Interleaving -> True, "ChannelGroups" -> 3, "Input" -> {"x", 7, 6}} -> "8*4*3_CJSUzvIqUJk_NVmekD7GxMw=7.994530e+1",
	{3, 2, "Input" -> {"x", 2, 3}, Interleaving -> True, "PaddingSize" -> {{0, 0}, {2, 1}}} -> "9*4*3_aFazV+F59Bc_BWkmvxtIrm8=1.073615e+2",
	{3, 2, "Input" -> {"x", 3, 4}, Interleaving -> True, "PaddingSize" -> {{1, 0}, {2, 1}}} -> "10*5*3_YuBgmWaqDxw_DINoz4CBijo=1.637360e+2",
	(* varying 3-d *)
	{1, {2, 2, 2}, Interleaving -> True, "Input" -> {"x", 5, 5, 1}} -> "9*4*4*1_Hu12a4Ahldg_F24VV033CZM=9.698269e+1",
	{1, 2, "Stride" -> 2, Interleaving -> True, "Input" -> {"x", 5, 5, 1}} -> "5*2*2*1_TwzOfeWxwyo_EH5n5VEdga8=1.357275e+1",
	{2, 3, "Stride" -> 2, Interleaving -> True, "ChannelGroups" -> 2, "Input" -> {"x", 5, 5, 2}} -> "4*2*2*2_J0KRWRF3cBM_Sc1uSBdKLMw=8.795095e+1",
	{3, 2, "Input" -> {"Varying", 3, 4, 2}, Interleaving -> True, "PaddingSize" -> {{1, 0}, {2, 1}, {1, 5}}} -> "3*5*9*3_Nsy6ZN/fHYI_Y6qE0Z7yKKs=4.007660e+2",
	(* 4-tensor inputs not supported due to limitations of pad layer *)
	(* integer *)
	{1, {2, 2, 2}, Interleaving -> True, "Input" -> {"x", 5, 5, 1, Restricted["Integer", 3]}} -> "9*4*4*1_EgEhzzT8MmQ_EwkhgemCxbc=4.787079e+2"
}

Upgraders: {
	"11.3.1" -> RenameParam["InputChannels" -> "$InputChannels"],
	"11.3.4" -> AddParam["Interleaving" -> False],
	"11.3.7" -> RenameParam["$GroupNumber" -> "ChannelGroups"] /* AddParam[Function[
		inputCh = #Parameters["$InputChannels"];
		groupN = #Parameters["ChannelGroups"];
		"$WeightsInputChannels" -> If[IntegerQ[inputCh], inputCh / groupN, SizeT]
	]],
	"11.3.8" -> UpgradeAsymmetricPadding
}