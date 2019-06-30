Input: TensorT[$Dimensions, AtomT]

Output: TensorT[$Dimensions]

Arrays:
	$Biases: TensorT[$$ArrayDimensions]

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
	reshaped = SowNode["reshape", #Biases, "shape" -> Prepend[#Dimensions, 1]];
	SetOutput["Output", SowNode["broadcast_plus", {GetInput["Input"], reshaped}]]
]

Tests: {
	{"Input" -> 3} -> "3_QemtvEXs7rE_M2JMbKSj9a8=3.404258e+0",
	{"Input" -> {2, 3}} -> "2*3_VyAOG9jA63Y_GE3y5h8yo8A=6.663319e+0",
	{"Input" -> "Scalar"} -> "1_W6/4dnFZXcA_TfPVz8ZUa+w=6.389388e-1",
	{"Input" -> {3}, "Biases" -> {1, 2, 3}} -> "3_JOFasL+2yeg_ZvkzEF+Qyso=6.904818e+0",
	{"Input" -> "Real"} -> "_IDqkqUg/wo8_OV6j0ghHCKk=6.389388e-1",
	{"Biases" -> 7, "Input" -> "Real"} -> "_E6o+0fkMO/E_WHMWRjylU8g=7.349816e+0",
	{"Input" -> "Integer"} -> "_bKaN5BKrftA_K4IZh5gAaTg=1.988755e+0"
}