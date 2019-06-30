Input: TensorT[SizeListT[$InputDepth], RealT]

Parameters:
	$InputDepth: Defaulting[NaturalT, 0]

Upgraders: {
	"11.3.4" -> Append["InputDepth" -> 0]
}

ArrayDepth: Function[
	#InputDepth
]

AvailableProperties: {"Decision", "Probability", "Entropy", "RandomSample", {"RandomSample", "Temperature" -> _}}

DecoderToEncoder: Function[
	{"Boolean", "Dimensions" -> TDimensions[#2]}
]

ToDecoderFunction: Function[
	decision
]

ToPropertyDecoderFunction: Function[
	Replace[#2, {
		"Decision" -> decision,
		"Probability" :> Function[If[$ReturnNumericArray, Identity, Normal][#]],
		"Entropy" :> entropy @* CheckDecoderProbs[{Min, Max}],
		"RandomSample" :> randomSample,
		{"RandomSample", "Temperature" -> t_ ? NonNegative} :> If[t < 0.00001, 
			decision,
			resigmoidAtTemp[N[t]] /* randomSample
		] @* CheckDecoderProbs[{Min, Max}],
		_ :> $Failed
	}]
]

decision[probs_List] := decision /@ probs;
decision[proba_NumericArray] := Map[NonNegative, Normal[proba] - 0.5, {-1}];
decision[proba_?NumericQ] := (proba >= 0.5);

entropy[probs_List] := entropy /@ probs;
entropy[proba_ /; (NumericQ[proba] || NumericArrayQ[proba])] := 
	With[{p = Clip[Normal[proba], {4.440892098500626`*^-16, 1 - 4.440892098500626`*^-16}]},
		If[$ReturnNumericArray, toNA["Real32"], Identity] @
			(- p * Log[p] + (p - 1) * Log[1 - p])
	];

randomSample[probs_List] := randomSample /@ probs;
randomSample[probNA_ /; (NumericQ[probNA] || NumericArrayQ[probNA])] := Scope[
	proba = Normal[probNA];
	If[ListQ[proba],
		MapThread[LessEqual, {RandomReal[{0, 1}, Dimensions[proba]], proba}, Depth[proba] - 1],
		RandomReal[] <= proba
	]
];

resigmoidAtTemp[temp_][proba_List] := resigmoidAtTemp[temp] /@ proba;
resigmoidAtTemp[temp_][probNA_ /; (NumericQ[probNA] || NumericArrayQ[probNA])] := Module[{scaledP, scaledQ, proba = Normal[probNA]},
	scaledP = Quiet @ Power[proba, 1./temp];
	scaledQ = Quiet @ Power[1 - proba, 1./temp];
	scaledP / (scaledP + scaledQ)
];