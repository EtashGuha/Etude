Input: TensorT[$$Dimensions, IndexIntegerT[$ClassCount]] 

Output: TensorT[$$Dimensions, VectorT[$OutputDimension]]

Arrays:
	$Weights: MatrixT[$ClassCount, $OutputDimension]

Parameters:
	$OutputDimension: SizeT
	$ClassCount: SizeT
	$$Dimensions: SizeListT[]

MinArgCount: 0

AllowDynamicDimensions: True

Writer: Function @ Scope[
	input = GetInput["Input", "Batchwise"];
	offset = SowPlusScalar[input, "-1.0"];
	output = SowNode["Embedding", {offset, #Weights}, 
		"output_dim" -> #OutputDimension, 
		"input_dim" -> #ClassCount,
		"dtype" -> $DTypeMXName
	];
	output = SowCast[output, "Real32", $DType];
	SetOutput["Output", output];
]

MXNet:
	Name: "Embedding"
	Parameters:
		$OutputDimension: "output_dim"
		$ClassCount: "input_dim"
	Arrays:
		$Weights: "weight"

Tests: {
	{4, "Input" -> IndexIntegerT[2]} -> "4_WLkUtE42iHI_NnpebYT2Fxw=3.720326e+0",
	{1, "Input" -> IndexIntegerT[5]} -> "1_I68+/ob0ElA_Bzmme6kciWc=6.336144e-1",
	{4, "Input" -> TensorT[{LengthVar[0]}, IndexIntegerT[2]]} -> "3*4_c8O6PSm2MMo_cTSUaa0wXNU=1.048043e+1",
	{4, "Input" -> TensorT[{5}, IndexIntegerT[3]]} -> "5*4_UqRMFHPu98Y_VUXsj/QElv4=1.492344e+1"
}