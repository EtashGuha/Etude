Input: $$Shape

Output: $$Shape

Suffix: "Operator"

Parameters:
	$Net: NetT[<|"Input" -> $$Shape|>, <|"Output" -> $$Shape|>]
	$Iterations: SizeT
	$$Shape: AnyTensorT

PostInferenceFunction: Function[
	PostSet[$Input, $Net["Inputs", "Input"]];
	PostSet[$Output, $Net["Outputs", "Output"]];
]

Writer: Function @ Scope[
	state = GetInput["Input"];
	Do[
		state = SowSubNet["Net", state]["Output"],
		#Iterations
	];
	SetOutput["Output", state];
]

Tests: {
	{Hold @ ElementwiseLayer[#*2&], 4, "Input" -> 2} -> "2_CuN/1W3FlN0_D5zk2DhgyDQ=1.268576e+1"
}