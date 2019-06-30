Input: InterleavingSwitchedT[$Interleaving, $$Channels, $$InputSize, AtomT]

Output: InterleavingSwitchedT[$Interleaving, $$Channels, $$OutputSize]

Parameters:
	$KernelSize:			ArbSizeListT[$Dimensionality, PosIntegerT, None]
	$Stride:				ArbSizeListT[$Dimensionality, PosIntegerT, 1]
	$PaddingSize:			PaddingSizeT[$Dimensionality, 0]
	$Function:				Defaulting[PoolingFunctionT, Max]
	$Dimensionality:		NaturalT
	$Interleaving:			Defaulting[BooleanT, False]
	$$Channels:				SizeT
	$$InputSize:			SizeListT[$Dimensionality]
	$$OutputSize:			ComputedType[SizeListT[$Dimensionality],
		If[$$MXGlobalPool,
			ConstantArray[1, Length @ $$InputSize],
			MaybeDyn @ PoolingShape[$$InputSize, $PaddingSize, $KernelSize, $Stride, $$MXPoolingConvention]
		]
	]
	$$MXPoolingConvention:	Defaulting[EnumT[{"valid", "full"}], "valid"]
	$$MXGlobalPool:			Defaulting[BooleanT, False]	

ReshapeParams: {$$Channels, $$InputSize, $$OutputSize}

MinArgCount: 1
PosArgCount: 2

PostConstructionFunction: Function[
	CheckPaddingSize[PoolingLayer, $PaddingSize, $KernelSize]
]

PostInferenceFunction: Function[
	CheckConvolutionOrPoolingFunction[PoolingLayer, $Dimensionality, $$InputSize, $KernelSize, $$OutputSize, $PaddingSize]
]

AllowDynamicDimensions: True

Constraints: SpatialConstraintGenerator[getFirstPoolOut]

getFirstPoolOut = Function[
	PoolingShape[#2, First @ #PaddingSize, First @ #KernelSize, First @ #Stride, #$MXPoolingConvention]
];

Writer: Function @ Scope[
	node = GetInput["Input"];

	fdim = First @ GetInputDims["Input"];
	lnode = GetDynamicLengthNode[fdim];
	SowDerivedSequenceLengthNode[lnode, First @ GetOutputDims["Output"], Function[x, getFirstPoolOut[#, x]]];
	If[lnode =!= None, node = SowSeqMaskBatchwise[node, lnode]];

	(* ASYMMETRIC PADDING *)
	symmetricPad = Apply[SameQ, Transpose[#PaddingSize]];
	If[!symmetricPad, 
		{node, workaround} = ApplyAsymmetricConvPoolPadding[node, #Dimensionality, #Interleaving, #PaddingSize]
	];
	(* POOLING *)
	output = If[#Interleaving === False || workaround === "Transpose", (* For clarity: (#Interleaving === True && workaround === "Transpose") *)
		SowCurrentNode[{node}],
		SowTransposedConvolutionOrPooling[#Dimensionality, node]
	];

	If[workaround === "Transpose", (* For clarity: (#Interleaving === True && workaround === "Transpose") *)
		output = SowTranspose[output, {0, 2, 3, 4, 1}]
	];
	SetOutput["Output", output];
]

MXNet:
	Name: "Pooling"
	Parameters:
		$KernelSize: "kernel"
		$Stride: "stride"
		$Function: "pool_type"
		$$MXGlobalPool: "global_pool"
		$$MXPoolingConvention: "pooling_convention"
	Reader: Function @ Scope[
		mxPad = readIntList[#pad];
		"PaddingSize" -> Transpose @ {mxPad, mxPad}
	]
	Writer: Function[{
		"pad" -> writeIntList[#PaddingSize[[All, 1]]] (* Pad spec is assumed to be symmetric or zeros at this point *)
	}] 

Tests: {
	(* 3-d case *)
	{{2, 3, 3}, "Input" -> {3, 7, 6, 5}, "Function" -> Max, "Stride" -> {1, 2, 2}, PaddingSize -> {1, 2, 0}} -> "3*8*4*2_EQ4P35Ccy0E_QhF3qggVUD0=1.703459e+2",
	{{2, 3, 3}, "Input" -> {3, 7, 6, 5}, "Function" -> Mean, "Stride" -> {1, 2, 2}, PaddingSize -> {1, 2, 0}} -> "3*8*4*2_aj0iVU0CxYg_Q8c3gYo6RTU=6.193873e+1",
	{{2, 3, 3}, "Input" -> {3, 7, 6, 5}, "Function" -> Total, "Stride" -> {1, 2, 2}, PaddingSize -> {1, 2, 0}} -> "3*8*4*2_MPgDdpgRivU_dLL114VT1PQ=1.114897e+3",
	{{2, 3, 3}, "Input" -> {3, 7, 6, 5}, "Function" -> Total, "Stride" -> {1, 2, 2}, PaddingSize -> {{1, 1}, {0, 1}, {0, 1}}} -> "3*8*3*2_Lgr/NxcvkKc_XuukL56k8ks=9.903695e+2",
	(* 2-d case *)
	{2, "Input" -> {2, 4, 4}} -> "2*3*3_ATYPBv07Hs8_Vdq2yW67/PY=1.333975e+1",
	{2, "PaddingSize" -> 2, "Input" -> {2, 4, 4}} -> "Validation failed for PoolingLayer: padding size {2, 2}\[Times]{2, 2} must be smaller than KernelSize 2\[Times]2.",
	{2, "Function" -> Mean, "Input" -> {2, 4, 4}} -> "2*3*3_JhX0kbpKO+8_G4GQnTKoDus=7.847875e+0",
	{3, "Input" -> {3, 7, 4}, "Function" -> Max, "Stride" -> {2, 1}, PaddingSize -> {2, 1}} -> "3*5*4_bUv3yMUEyeE_YyINkkaSszI=4.657709e+1",
	{3, "Input" -> {3, 7, 4}, "Function" -> Mean, "Stride" -> {2, 1}, PaddingSize -> {2, 1}} -> "3*5*4_CssmLlUjNzQ_UyPtpiLtKfw=1.722544e+1",
	{3, "Input" -> {3, 7, 4}, "Function" -> Total, "Stride" -> {2, 1}, PaddingSize -> {2, 1}} -> "3*5*4_K/J1/1bOqOc_I6KTqCNUt8A=1.550289e+2",
	{{2, 1}, "Input" -> {2, 4, 4}} -> "2*3*4_XfU+DxhvLsI_d+76Z3xB+IU=1.469866e+1",
	{2, "Stride" -> 2, "Input" -> {2, 4, 4}} -> "2*2*2_C96JdXhPPc0_LujfV2xlsXM=6.139787e+0",
	{4, "Input" -> {2, 4, 4}, "PaddingSize" -> {{1, 3}, {0, 2}}} -> "2*5*3_I6a/1rXeQLE_Onz/r4f/mHg=2.440774e+1",
	(* 1-d case *)
	{3, "Input" -> {1, 9}} -> "1*7_DwN+i+t8HKQ_U1eyO5fisj0=4.786678e+0",
	{3, "Input" -> {1, 9}, "Stride" -> 3} -> "1*3_EO8rW4iATf4_Pk50/gKW0Nk=2.049754e+0",
	{3, "Input" -> {3, 7}, "Function" -> Max, "Stride" -> 2, PaddingSize -> 2} -> "3*5_anX9wVnPIzU_B26+8ZmZN8s=9.388439e+0",
	{3, "Input" -> {3, 7}, "Function" -> Mean, "Stride" -> 1, PaddingSize -> 2} -> "3*9_Omv6NGNcZYw_TKs6wuvdNfA=9.341780e+0",
	{3, "Input" -> {3, 7}, "Function" -> Total, "Stride" -> 2, PaddingSize -> 2} -> "3*5_NjtjdHcDlF0_WejJ3EavX/k=1.513854e+1",
	{4, "Input" -> {2, 4}, "PaddingSize" -> {{1, 3}}} -> "2*5_PwsWaQlyfl0_BNN3zLeJqCU=7.230573e+0",
	(* varying 1-d *)
	{3, Interleaving -> True, "Input" -> {"x", 1}} -> "8*1_CoeqAORew0Q_bey3vLQNT1I=5.623863e+0",
	{3, "Function" -> Mean, Interleaving -> True, "Input" -> {"x", 1}} -> "8*1_Wzci/+KIF9Q_LtEgJoXK/LI=3.039057e+0",
	{2, "Stride" -> 2, Interleaving -> True, "Input" -> {"x", 1}} -> "5*1_B+trNZV6T8Y_BR1m0XYGIFo=2.951774e+0",
	{3, "Input" -> {"x", 3}} -> "10*1_DloG5eTkV/A_GeCMc1TncYg=7.257263e+0",
		{{2}, Interleaving -> True, "PaddingSize" -> {1}, "Input" -> {"x", 1}} -> "11*1_UPD5QPDVPwA_Xj/QYEx69M4=6.440885e+0",
	{3, "Input" -> {"x", 3}, PaddingSize -> {{2, 1}}} -> "10*4_Uqin+v38OuM_XhRPX2Nac2E=2.497419e+1",
	(* varying 2-d *)
	{{2, 2}, Interleaving -> True, "Input" -> {"x", 4, 1}} -> "9*3*1_ABPrCk7Z4KE_ZI23x3IKH+4=1.874703e+1",
	{2, "Stride" -> 2, Interleaving -> True, "Input" -> {"x", 5, 1}} -> "5*2*1_cctCelVR7dA_SK0uEW6cjNM=7.174675e+0",
		{{2, 2}, Interleaving -> True, "PaddingSize" -> {1, 1}, "Input" -> {"x", 4, 1}} -> "11*5*1_amMyDyz8FYk_VDo0O/1gyz8=3.463479e+1",
		{4, "Input" -> {"x", 4, 3}, Interleaving -> True, PaddingSize -> {{0, 0}, {3, 1}}} -> "7*5*3_fzokfP1VjKY_DCnxHVOWYfY=9.170128e+1",
	{5, Interleaving -> True, "Input" -> {"x", 5, 1}, PaddingSize -> {{4, 1}, {2, 3}}} -> "11*6*1_YMCYm1c5CkY_eHvNG0FUBJk=5.787729e+1",
	(* varying 3-d *)
	{{2, 2, 2}, Interleaving -> True, "Input" -> {"x", 5, 5, 1}} -> "9*4*4*1_QokQWgFIrRM_EVpJq1wCx/k=1.238599e+2",
	{4, "Input" -> {"x", 4, 7, 3}, Interleaving -> True, PaddingSize -> {{0, 0}, {1, 2}, {3, 1}}} -> "7*4*8*3_Lod5RWjacMM_IAgjYaSQYME=6.467638e+2",
	{5, Interleaving -> True, "Input" -> {"x", 5, 5, 1}, PaddingSize -> {{4, 1}, {2, 3}, {0, 1}}} -> "11*6*2*1_SlJWJX/FcHQ_FZDgl1I8fSQ=1.261822e+2",
	(* 4-tensor inputs not supported due to limitations of pad layer *)
	(* integers *)
	{5, Interleaving -> True, "Input" -> {"x", 5, 5, 1, Restricted["Integer", 3]}, PaddingSize -> {{4, 1}, {2, 3}, {0, 1}}} -> "11*6*2*1_aHUQVtuEHAU_VvgQ/Z6DqR0=3.960000e+2"
}

Upgraders: {
	"11.3.1" -> RenameParam["Channels" -> "$Channels"],
	"11.3.4" -> AddParam["Interleaving" -> False],
	"11.3.8" -> UpgradeAsymmetricPadding
}
