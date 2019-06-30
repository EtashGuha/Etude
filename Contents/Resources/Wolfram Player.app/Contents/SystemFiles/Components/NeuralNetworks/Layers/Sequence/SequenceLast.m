Input: ChannelT[SizeT, $Output]

Output: AnyTensorT

AllowDynamicDimensions: True

Writer: Function @ Scope[
	input = GetInputMetaNode["Input"];
	last = SowMetaLast[input];
	SetOutput["Output", last];
]

Tests: {
	{"Input" -> 4} -> "_IWqOWEDnRsY_JnzXy27PbJU=7.695246e-1",
	{"Input" -> {3, 5}} -> "5_N6sEq6rJ9TI_EDX3DcMW7R4=2.380216e+0",
	{"Input" -> {3, 2, 2}} -> "2*2_HpsfqpCDCtI_WMYO2gH/cV8=2.122520e+0",
	{"Input" -> "Varying"} -> "_akQbzL2Wru4_GrrrizPh1LI=1.119578e-1",
	{"Input" -> {"Varying", 1}} -> "1_ISPdreHzTcg_BncZwl5gfBY=1.119578e-1",
	{"Input" -> {"Varying", 2, 2}} -> "2*2_HpsfqpCDCtI_A3KOoAQHp3A=2.122520e+0",
	{"Input" -> {3, 2, 2, Restricted["Integer", 3]}} -> "2*2_W8D1Ql27IR0_aSOvYKNy7eE=7.000000e+0"
}