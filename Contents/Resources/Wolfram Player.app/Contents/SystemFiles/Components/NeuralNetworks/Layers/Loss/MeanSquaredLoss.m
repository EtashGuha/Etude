InheritsFrom: "MeanAbsoluteLoss"

Writer: Function[
	MeanLossImplementation["L2", #$Dimensions];
]

Tests: {
	{"Input" -> 3} -> "_akQbzL2Wru4_CXMm+k8IbXU=1.079522e-1",
	{"Input" -> {3, 3}} -> "_MlpwwRo1v4c_NqUr6V94NH8=1.933650e-1",
	{"Input" -> "Scalar", "Target" -> "Scalar"} -> "_XDP256GjoCY_fvTJ02EfWtY=8.691551e-3",
	{"Input" -> "Varying"} -> "_akQbzL2Wru4_RqvfcP3TWoc=1.079522e-1",
	{"Input" -> {"Varying", 2}} -> "_ZHCL9OGeDLY_DJ54zC4pVkk=1.359382e-1"
}