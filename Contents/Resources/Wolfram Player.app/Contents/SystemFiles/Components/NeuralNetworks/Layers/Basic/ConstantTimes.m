Input: TensorT[$Dimensions, AtomT]

Output: TensorT[$Dimensions]

Arrays:
	$Scaling: TensorT[$$ArrayDimensions]

Parameters:
	$Dimensions: SizeListT[]
	$$ArrayDimensions: ComputedType[SizeListT[],
		If[$Dimensions==={}, {1}, $Dimensions],
		{$Dimensions}
	]

PostInferenceFunction: Function[
	If[ListQ[$$ArrayDimensions] && !ListQ[$Dimensions],
		PostSet[$Dimensions, $$ArrayDimensions];
		RestartInference[];
	];
]

Upgraders: {
	"12.0.2" ->
		AddParam[Function["$ArrayDimensions" -> #Parameters["Dimensions"]]]
}

MinArgCount: 0

AllowDynamicDimensions: True

Writer: Function @ Scope[
	reshaped = SowNode["reshape", #Scaling, "shape" -> Prepend[#Dimensions, 1]];
	SetOutput["Output", SowNode["broadcast_mul", {GetInput["Input"], reshaped}]]
]

Tests: {
	{"Input" -> 3} -> "3_AIvAa0t3efE_dSrbrONjgxw=9.048177e-1",
	{"Input" -> {2, 3}} -> "2*3_XNJtjBjJ1rk_Luzvstmun4A=1.911142e+0",
	{"Input" -> "Scalar"} -> "1_dZt2SS1lszQ_EpNq1f2dLlk=3.498157e-1",
	{"Input" -> {3}, "Scaling" -> {1, 2, 3}} -> "3_EDrB4+vbRz0_M+TnFc63bxs=1.571778e+0",
	{"Input" -> "Real"} -> "_UlS0Sd7vNAo_N6dBguLAOZ8=3.498157e-1",
	{"Scaling" -> 7, "Input" -> "Real"} -> "_I/dT+CqR6ks_H6yPTchahP4=2.448710e+0",
	{"Input" -> {3, "Integer"}, "Scaling" -> {1, 2, 3}} -> "3_eroeA6k6h2I_Bu27LhDY9tQ=3.100000e+1"
}