Input: AnyTensorT

Output: AnyTensorT

AllowDynamicDimensions: True

ShapeFunction: Identity

TypeFunction: Function[{RealT}]

Parameters:
	$DropoutProbability: Defaulting[IntervalScalarT[0., 1.], 0.5]
	$Method: Defaulting[EnumT[{"Dropout", "AlphaDropout"}], "Dropout"]

PosArgCount: 1

PostInferenceFunction: Function @ If[$DropoutProbability == 1.,
	FailValidation["Value given for the dropout probability (first argument) should be a number between 0.0 and 1.0 excluded, but was 1.0 instead"]
] 

HasTrainingBehaviorQ: Function[True]

Writer: Function @ Scope[
	rate = #DropoutProbability;
	input = GetInput["Input"];
	If[!$TMode, SetOutput["Output", input]; Return[Null, Block]];
	Switch[#Method,
		"Dropout",
			out = SowDropout[input, rate],
		(* Should be consistent with: https://github.com/bioinf-jku/SNNs/blob/master/selu.py *)
		"AlphaDropout",
			alpha = 1.6732632423543772848170429916717;
			scale = 1.0507009873554804934193349852946;
			out = SowDropout[input, rate];
			mask = out;
			mask[[2]] = 1; (* Note: mask is scaled by 1 / (1 - DropoutProbability) *)
			(* rescale *)
			mask = SowPlusScalar[mask, -1];
			mask = SowTimesScalar[mask, alpha * scale];
			out = SowPlus[out, mask];
			const = Sqrt[(1 - rate) / (1 + alpha^2 * rate * scale^2)];
			out = SowTimesScalar[out, const];
	];
	SetOutput["Output", out];
]

MXNet:
	Name: "Dropout"
	Parameters:
		$DropoutProbability: "p"

Tests: {
	{"Input" -> {10}} -> "10_HLx5dVXlBsY_PLliAiWtGQo=3.939840e+0",
	{"Input" -> {7, 3, 2}} -> "7*3*2_NyFp29BWUDY_TBwD+IkTaE8=1.839211e+1",
	{"Input" -> {"Varying", 3}} -> "3*3_YNBbVYLSHUk_d4RU0KGWjWs=3.210989e+0",
	{"Method" -> "AlphaDropout", "Input" -> {10}} -> "10_HLx5dVXlBsY_PLliAiWtGQo=3.939840e+0",
	{"Method" -> "AlphaDropout", "Input" -> {7, 3, 2}} -> "7*3*2_NyFp29BWUDY_TBwD+IkTaE8=1.839211e+1",
	{"Method" -> "AlphaDropout", "Input" -> {"Varying", 3}} -> "3*3_YNBbVYLSHUk_d4RU0KGWjWs=3.210989e+0"
}
