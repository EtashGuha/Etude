Input: AnyTensorT

Output: $Input

AllowDynamicDimensions: True

Writer: Function @ Scope[
	input = GetInputMetaNode["Input"];
	reverse = SowMetaReverse[input];
	SetOutput["Output", reverse];
]

Tests: {
	{"Input" -> {3, 2, 2}} -> "3*2*2_MWzfPOzBY/k_MzZ7ruTUvlo=5.231537e+0",
	{"Input" -> {"Varying", 1}} -> "3*1_c/ktmxJfUAw_RpRzvno9DU4=9.048177e-1",
	{"Input" -> {"Varying", 2, 2}} -> "3*2*2_MWzfPOzBY/k_O24ncqqz3QM=5.231537e+0",
	{"Input" -> {"Varying"}} -> "3_GvtHLiAnhdk_D20EjFN5x2g=9.048177e-1",
	{"Input" -> {"Varying", Restricted["Integer", 3]}} -> "3_Er0fn4A748U_VKdgVFTZYIo=7.000000e+0"
}
