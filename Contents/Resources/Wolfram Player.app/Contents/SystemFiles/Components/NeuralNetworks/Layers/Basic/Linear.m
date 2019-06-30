Input: TensorT[$$InputDimensions, AtomT]

Output: TensorT[$OutputDimensions]

Arrays:
	$Weights: MatrixT[$$OutputSize, $$InputSize]
	$Biases: Nullable @ VectorT[$$OutputSize]

Parameters:
	$OutputDimensions: NormalizedT[SizeListT[], toSizeList]
	$$OutputSize: ComputedType[SizeT, Times @@ $OutputDimensions]
	$$InputSize: ComputedType[SizeT, Times @@ $$InputDimensions]
	$$InputDimensions: SizeListT[]

MinArgCount: 0

toSizeList[Automatic] := Automatic;

toSizeList[e_] := Scope[
	res = ToList[e] /. Automatic -> SizeT;
	If[MatchQ[res, {(Integer|SizeT)...}], BypassCoerce[res], res]
];

Writer: Function @ Scope[
	irank = Length[#$InputDimensions]; orank = Length[#OutputDimensions];
	If[orank == 1, MXWriteDefaultAndReturn[]];
	input = GetInput["Input"];
	output = SowFC[input, #Weights, #Biases, #$OutputSize];
	output = SowReshape[output, #OutputDimensions];
	SetOutput["Output", output];
]	

PostInferenceFunction: Function[
	If[VectorTypeQ[$Input] && IntegerQ[$$InputSize],   PostSet[$Input,  VectorT[$$InputSize, AtomT]]];
	If[VectorTypeQ[$Output] && IntegerQ[$$OutputSize], PostSet[$Output, VectorT[$$OutputSize]]];
	RestartInference[];
]

MXNet:
	Name: "FullyConnected"
	Parameters: 
		$$OutputSize: "num_hidden"
	Arrays:
		$Weights: "weight"
		$Biases: "bias"
	Writer: Function["no_bias" -> If[#2["Biases"] === None, "True", "False"]] 
	Reader: Function @ Scope[
		flatten = Lookup[#, "flatten", True];
		If[flatten == False, ImportParamException["flatten" -> False]];
		If[$ForceFullyConnectedVectorInputs,
			{"Output" -> {Automatic}, "Input" -> {Automatic}},
			{"Output" -> {Automatic}}
		]
	]

Tests: {
	{3, "Input" -> 4} -> "3_Q/Ie6ed/y2A_de2pQHdWQn4=2.952214e+0",
	{3, "Input" -> 4, "Biases" -> None} -> "3_AN0XNdKPGwE_f5VdEVlU9hY=2.777422e+0",
	{2, "Weights" -> {{0.1, 0.2}, {0.2, 0.4}}, "Biases" -> {0.1, 0.1}, "Input" -> 2} -> "2_KBSkqEK2twg_Pn2twHHtBkw=5.707712e-1",
	{"Input" -> {2, 2}, "Output" -> {3, 3}} -> "3*3_eg5mXSWE7ko_ZBMvCzkmWJA=9.928539e+0",
	{"Input" -> {2, "Integer"}, "Output" -> {2, 2}} -> "2*2_H1YWH076Xs0_INs6VtAObak=1.736631e+1",
	{"Input" -> {2, Restricted["Integer", Infinity]}, "Output" -> {2, 2}} -> "2*2_Kkqk2ptmGog_SDWdv2bTcMw=2.757776e+1",
	{"Input" -> {2, Restricted["Integer", 3]}, "Output" -> {2, 2}} -> "2*2_M/jm+s/DR04_OT1qXEKLi1k=6.583582e+0"
}