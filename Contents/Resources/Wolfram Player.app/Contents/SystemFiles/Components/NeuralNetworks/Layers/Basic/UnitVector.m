Input: TensorT[$$Dimensions, IndexIntegerT[$ClassCount]] 

Output: TensorT[$$Dimensions, VectorT[$ClassCount]]

Parameters:
	$ClassCount: SizeT
	$$Dimensions: SizeListT[]

MinArgCount: 0

AllowDynamicDimensions: True

Writer: Function @ Scope[
	input = GetInput["Input", "Batchwise"];
	input = SowNode["cast", input, "dtype" -> "int32"];
	input = SowPlusScalar[input, "-1.0"];
	output = SowOneHot[input, #ClassCount];
	SetOutput["Output", output];
]

Tests: {
	{4, "Input" -> {}} -> "4_C87PL0UZQE0_WYtgqPQoZJI=1.000000e+0",
	{4, "Input" -> {2}} -> "2*4_c/yaq1QJ3Ds_L+cBfN/Rgu4=2.000000e+0",
	{4, "Input" -> "Varying"} -> "3*4_OcL/YBwC4FE_Qga5Vhsq7lI=3.000000e+0",
	{4, "Input" -> {"Varying", 2}} -> "3*2*4_Y1V9NjhlNNM_XXhHdwhkC0I=6.000000e+0"
}

