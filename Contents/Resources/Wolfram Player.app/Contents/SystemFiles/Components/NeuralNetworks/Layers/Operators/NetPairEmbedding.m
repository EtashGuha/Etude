Input: SequenceT[2, $$InputShape]

Output: ScalarT

Suffix: "Operator"

Parameters:
	$Net: NetT[<|"Input" -> $$InputShape|>, <|"Output" -> $$OutputShape|>]
	$DistanceFunction: Defaulting @ EnumT[{EuclideanDistance, CosineDistance}]
	$$InputShape: AnyTensorT
	$$OutputShape: AnyTensorT

PosArgCount: 1

PostInferenceFunction: Function[
	PostSet[$Input, SequenceT[2, $Net["Inputs", "Input"]]]
]

Writer: Function @ Scope[
	idims = TDimensions[#$InputShape]; 
	odims = TDimensions[#$OutputShape];
	orank = Length[odims];
	input = GetInput["Input"];
	pre = SowFlatten @ input;
	out = SowSubNet["Net", pre]["Output"];
	out = SowUnflatten[out, input];
	{out1, out2} = SowUnpack[out, 2, 1];
	ofunc = Switch[#DistanceFunction,
		CosineDistance, SowCosineDistance,
		EuclideanDistance, SowEuclideanDistance
	];
	SetOutput["Output", ofunc[out1, out2, orank]];
]

Tests: {
	{Hold @ ElementwiseLayer[Ramp, "Input" -> 2]} -> "_Twwpo0AiU+o_KzteSorRuVk=1.631658e-1"
}