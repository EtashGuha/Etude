Input: TensorT[$$Dimensions, AtomT]

Output:	TensorT[$$Dimensions, AtomT]

Parameters:
	$Function: UnaryElementwiseFunctionT
	$$Dimensions: SizeListT[]

ShapeFunction: Identity (* only to trigger TypeFunction (see DefineLayer.m) *)
RankFunction: Identity
TypeFunction: Function[
	{outputType[
		ScalarFunctionApply @ First[$Function], (* evaluable elementwise function *)
		First @ First[# /. Var[_] :> $type[RealT]] (* input type *)
	]}
]

outputType[function_, inputType_] := Module[{},
	Switch[inputType,
		IndexIntegerT[n_Integer] /; n < 100000, outputTypeIntegerRange[function, 1, First[inputType]],
		IndexIntegerT[All], outputTypeIntegerRange[function, -100000, 100000, {-2^30, 2^30}],
		IndexIntegerT[Infinity], outputTypeIntegerRange[function, 1, 100000, {2^30}],
		_, RealT
	]
];
outputTypeIntegerRange[function_, min_, max_, additional_:{}] := Module[{outintQ, outmin, outmax},
	outs = function @ Join[Range[min, max], additional]; (* Listability is ensured by ScalarFunctionApply *)
	outintQ = AllTrue[outs, Round[#] == #&]; (* double == on purpose *)
	If[outintQ,
		{outmin, outmax} = MinMax[outs];
		If[TrueQ[outmin > 0],
			IndexIntegerT[If[additional === {} && IntegerQ[Round[outmax]], Round[outmax], Infinity]],
			IndexIntegerT[All] (* bounded integers only supported with lower bound 1 for now *)
		]
	,
		RealT
	]
];


AllowDynamicDimensions: True

MinArgCount: 1

Writer: Function[
	SetOutput["Output", SowScalarFunction[First @ #Function, GetInput["Input"]]];
]

SummaryFunction: Function @ Scope[
	func = First[#Function];
	If[!AtomQ[func] || !MemberQ[$PrimitiveUnaryElementwiseFunctions, func],
		func2 = ScalarFunctionToPureFunction[func];
		If[LeafCount[func2] > 10, Return[ElementwiseLayer]];
		func = func2[$DummyVar] /. r_Real /; (r == Round[r]) :> Round[r];
	];
	func
]

MXNet: 
	Name: "Activation"
	Parameters: 
		$Function: "act_type"
	Reader: Function[
		"Function" -> $FromActivation[#["act_type"]]
	]

Tests: {
	{Ramp, "Input" -> {}} -> "_UlS0Sd7vNAo_N6dBguLAOZ8=3.498157e-1",
	{Ramp, "Input" -> 4} -> "4_N6dBguLAOZ8_apeZsB5KfCw=1.674342e+0",
	{LogisticSigmoid, "Input" -> {4}} -> "4_XXj3lKcHKcs_Pbd8Eqi2aO4=2.406935e+0",
	{Tanh, "Input" -> 4} -> "4_G2cnksaJGog_G9G1MQtYalM=1.510522e+0",
	{ArcTan, "Input" -> {4}} -> "4_L8KbUzwPWR8_IFA7BSjJtEM=1.520939e+0",
	{ArcTanh, "Input" -> {4}} -> "4_BLa5o91bKUg_cCE/YBx4Zmk=1.972836e+0",
	{Sin, "Input" -> {4}} -> "4_VP4C9G8SjU8_Qye/ozWc77A=1.578934e+0",
	{Sinh, "Input" -> {4}} -> "4_PE57l3hgKbo_Frzn0nueCps=1.774620e+0",
	{ArcSin, "Input" -> {4}} -> "4_GLAhRQXQUV8_MkV3ig2o+0w=1.806655e+0",
	{Cos, "Input" -> 4} -> "4_YGr8Cph+fhU_TUxB107xNYA=3.554868e+0",
	{ArcCos, "Input" -> {4}} -> "4_LnduJwGj+Ao_VdcKy09xL8A=4.476530e+0",
	{Log, "Input" -> {4}} -> "4_IqnlIB+NouU_B5TPhDUJlNM=4.316050e+0",
	{Exp, "Input" -> {4}} -> "4_Qu2YQF45zdM_IYp8MvlHXqg=6.253453e+0",
	{Sqrt, "Input" -> {4}} -> "4_HIASKVnaFgY_bG9RQqaAlH4=2.468894e+0",
	{Gamma, "Input" -> {4}} -> "4_H/bLKcsliyE_MpZ5QW+TQsw=1.420238e+1",
	{Abs,  "Input" -> 4} -> "4_N6dBguLAOZ8_apeZsB5KfCw=1.674342e+0",
	{LogGamma, "Input" -> {4}} -> "4_f5J8p2T1kUM_fBfIM9vDhVM=3.945250e+0",
	{#1+1 & , "Input" -> 4} -> "4_DXWJKbj8WcI_U3DdNX4dvsI=5.674342e+0",
	{#1*2 & , "Input" -> 4} -> "4_RmDhgFbgFq8_WmilYV1De7Y=3.348685e+0",
	{#1/2 & , "Input" -> 4} -> "4_XwjPwhZxSS4_Zyd2AyViKWU=8.371712e-1",
	{#1-2 & , "Input" -> 4} -> "4_DqtzqPKHBPA_ABIucocKrUs=6.325658e+0",
	{#1^2 & , "Input" -> 4} -> "4_MdlGgeekRbg_fuS46yPUn5Q=9.233618e-1",
	{Exp[-#1^2] & , "Input" -> 4} -> "4_TKlcbB3sGUI_RmiN6AcUf7U=3.247266e+0",
	{Min[#1, 5] & , "Input" -> 4} -> "4_N6dBguLAOZ8_apeZsB5KfCw=1.674342e+0",
	{Ramp, "Input" -> {4, 4}} -> "4*4_apeZsB5KfCw_EthWqvd3jvs=6.833459e+0",
	{Clip[#1, {0.4, 0.5}] & , "Input" -> {4}} -> "4_ccz6AwGtEF8_SJIQcRA6E7U=1.743044e+0",
	{Clip[#1, {-2.3, 0.5}] & , "Input" -> {4}} -> "4_VZ+ShuX//NU_NabzX+KgSuE=1.404818e+0",
	{Clip[3.4*Sin[#1] + 3.4 + #1/3.2 + #1^2 - Max[#1] + Min[Cos[#1] + 2.3], {-2.3, 0.5}] & , "Input" -> {2}} -> "2_VYYdKBM7HxE_FbJGc4qAafQ=1.000000e+0",
	{"ReLU", "Input" -> 4} -> "4_N6dBguLAOZ8_apeZsB5KfCw=1.674342e+0",
	{"ELU", "Input" -> 4} -> "4_N6dBguLAOZ8_apeZsB5KfCw=1.674342e+0",
	{"SELU", "Input" -> 4} -> "4_KfFvsbScbGo_JcO7nh8wjbA=1.759233e+0",
	{"SELU", "Input" -> {}} -> "_X60SkhiiT3Y_KfFvsbScbGo=3.675517e-1",
	{"SoftSign", "Input" -> 4} -> "4_RKJiB/5+z0g_Tt9bOCh8h34=1.101740e+0",
	{"SoftPlus", "Input" -> 4} -> "4_XZO7YlYJ2NM_dRDkUvGIoCw=3.723146e+0",
	{"HardTanh", "Input" -> 4} -> "4_N6dBguLAOZ8_apeZsB5KfCw=1.674342e+0",
	{"HardSigmoid", "Input" -> 4} -> "4_AuBLnaB8RkQ_L0mF5Qar04I=2.837171e+0",
	{"Sigmoid", "Input" -> 4} -> "4_XXj3lKcHKcs_Pbd8Eqi2aO4=2.406935e+0",
	{Erf, "Input" -> {"Varying",3}} -> "3*3_agU2PZzu3Ok_K1TxO+B9dIk=3.230996e+0",
	{Cos, "Input" -> {4, "Integer"}} -> "4_MHd7pGCegCQ_JRRVmz+0Q1A=3.304497e+0",
	{Cos, "Input" -> {"Varying",4, "Integer"}} -> "3*4_EMUM+M7g6u0_Pw1AQRZYErk=7.461246e+0"
}