Inputs: <||>

Output: TensorT[$Dimensions]

Arrays:
	Array: TensorT[$$ArrayDimensions]

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
MaxArgCount: 1

Writer: Function[
	SetOutput["Output", If[#Dimensions === {}, SowFlatten, Identity] @ SowBatchBroadcast[#Array]]
]

Tests: {
	{"Output" -> {3, 3}} -> "3*3_S+gOgNB2PmI_S+gOgNB2PmI=7.299426e+0",
	{"Array" -> ArrayReshape[Range[24],{2,3,4}]} -> "2*3*4_KXPQQAtTBlY_KXPQQAtTBlY=3.000000e+2",
	{"Output" -> "Real"} -> "_aNjECi4GCxE_aNjECi4GCxE=9.887545e-1",
	{"Array" -> -2, "Output" -> "Real"} -> "_bKaN5BKrftA_bKaN5BKrftA=2.000000e+0"
}
