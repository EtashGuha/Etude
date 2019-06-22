Input: $$Shape
Output: $$Shape

Parameters:
	$ChannelWindowSize: Defaulting[PosIntegerT, 2]
	$Alpha: Defaulting[ScalarT, 1.0]
	$Beta: Defaulting[ScalarT, 0.5]
	$Bias: Defaulting[ScalarT, 1.0]
	$$Channel: SizeT
	$$Shape: TensorT[{$$Channel, SizeT, SizeT}]


Writer: Function @ Scope[
	data = GetInput["Input"];
	(* LRN in mxnet doesn't support Real64 yet. Change when it does *)
	data = SowCast[data, $DType, "Real32"];
	
	nsize = IntegerString[2 * #ChannelWindowSize + 1];
	output = SowNode["LRN", {data}, 
		"nsize" -> nsize, "alpha" -> #Alpha, 
		"beta" -> #Beta, "knorm" -> #Bias
	];

	output = SowCast[output, "Real32", $DType];
	SetOutput["Output", output]
]

MXNet:
	Name: "LRN"
	Parameters: 
		$Alpha: "alpha"
		$Beta: "beta"
		$Bias: "knorm"
	Reader: Function["ChannelWindowSize" -> (FromDigits[#nsize] - 1)/2]
	
PostInferenceFunction: Function[
	If[($ChannelWindowSize * 2 + 1) > $$Channel,
		FailValidation["2 * channel window size + 1 cannot be larger than the number of channels in the input."]
	];
]

Tests: {
	{"Input" -> {5, 4, 4}} -> "5*4*4_LvY7WFc5wAY_Foi5OFUK+vI=3.412520e+1"
}