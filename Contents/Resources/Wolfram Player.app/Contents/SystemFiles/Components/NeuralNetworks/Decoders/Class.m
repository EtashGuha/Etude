Input: TensorT[SizeListT[$$Rank], VectorT[$Dimensions]]

Parameters:
	$Labels: Defaulting[EitherT[{MatchT[Automatic], ListT[$Dimensions, ExpressionT]}], Automatic]
	$InputDepth: Defaulting[SizeT, 1]
	$Dimensions: SizeT
	$$Rank: ComputedType[NaturalT, $InputDepth - 1]

Upgraders: {
	"11.3.4" -> Append[{"InputDepth" -> 1, "$Rank" -> 0}],
	"12.0.1" -> fixIDepth
}

(* this fixes the earlier, incorrect upgrade rule in 12.0.0 *)
fixIDepth[params_] := If[params["InputDepth"] === 0, Append[params, {"InputDepth" -> 1, "$Rank" -> 0}], params];

ArrayDepth: Function[
	#InputDepth
]

MaxArgCount: 1

(* unfortunately if labels is Automatic the dims never flow back from the input! *)

DecoderToEncoder: Function[
	{"Class", Replace[#Labels, Automatic :> Range[#Dimensions]], "Dimensions" -> Most @ TDimensions[#2]}
]

ToDecoderFunction: Function[
	decision[#Labels]
]

ToPropertyDecoderFunction: Function @ With[
	{labels = #Labels},
	{dims = If[labels === Automatic, Infinity, Length @ labels]},
	Replace[#2, {
		"Decision" :> 
			decision[labels],
		"Probabilities" :> 
			Map @ threadProbabilities[labels, #InputDepth],
		{"Probabilities"|"Probability", class_} :> If[
			(MemberQ[labels, Verbatim @ class] || (labels === Automatic && IntegerQ[class])), 
				Map @ probOf[class, labels],
				ThrowFailure[NetDecoder::notclassmember	, class]],
		"TopProbabilities" :> 
			Map[topProbs[labels]] @* CheckDecoderProbs[{Min, Total}],
			(* TODO: Consider "TopProbablities" -> n *)
		{"TopProbabilities", n_Integer /; n > 0} :> 
			Map[topProbs[labels, Min[n, dims]]],
		{"TopDecisions", n_Integer /; n > 0} :> 
			Map @ topDec[labels, Min[n, dims]],
		"RandomSample" :> 
			Map @ randomSampler[labels],
		{"RandomSample", "Temperature" -> t_ ? NonNegative} :> 
			Map[temperatureSampler[N[t], labels]] @* CheckDecoderProbs[{Min, Max}],
		"Entropy" :> 
			entropy,
		_ :> $Failed
	}]
]

 NetDecoder::notclassmember = "Requested probability for ``, which is not one of the possible classes.";

AvailableProperties: {"Decision", "TopProbabilities", 
	{"TopDecisions", _}, {"TopProbabilities", _Integer}, "Probabilities", 
	{"Probabilities", _}, {"Probability", _}, 
	"RandomSample", {"RandomSample", "Temperature" -> _}, 
	"Entropy"
}

(*********************)

(* helper used to fill in Automatic class at runtime based on final
dimension of probability input array *)
makeAutoLabels[arr_] := Range @ Last @ arrayDimensions @ arr;

(*********************)

decision[labels_][input_List] := 
	UnsafeQuietCheck @ Map[Part[labels, #] &, ArrayMaxIndex[Normal /@ input], {-1}];

decision[Automatic][input_List] := Scope[
	res = UnsafeQuietCheck @ ArrayMaxIndex[Normal /@ input];
	If[!IntegerQ[res] && $ReturnNumericArray, 
		toNumericArray[res, "UnsignedInteger32"], 
		res
	]
];

(*********************)

threadProbabilities[labels_List, 1][input_NumericArray] :=
	AssociationThread[labels, Normal @ input];

threadProbabilities[labels_List, depth_][input_NumericArray] := 
	AssociationThread[
		labels, 
		If[$ReturnNumericArray, 
			toNAList[NumericArrayType @ input],
			Identity
		] @ Transpose[Normal @ input, RotateLeft @ Range @ depth]
	];

threadProbabilities[Automatic, depth_][input_NumericArray] := 
	threadProbabilities[makeAutoLabels @ input, depth] @ input;

(*********************)

probOf[class_, labels_][input_NumericArray] := Scope[
	index = IndexOf[labels, class];
	res = Map[#[[index]]&, Normal[input], {-2}];
	If[$ReturnNumericArray,
		toNA[NumericArrayType[input]],
		ToPackedArray
	] @ res
];

probOf[class_, Automatic][input_NumericArray] := Scope[
	res = Map[Extract[class], Normal[input], {-2}];
	If[$ReturnNumericArray,
		toNA[NumericArrayType[input]],
		ToPackedArray
	] @ res
];

(*********************)

topProbs[labels_][inputNA_NumericArray] := Scope[ 
	input = Normal[inputNA];
	indicesValues = compiledTopProbs[input]; 
	partSpec = Sequence@@Table[All, Depth[input] - 2]; 
	(* TODO: the following two lines become EXTREMELY slow for 
	   very big tensors (e.g. ~17s for 500x500x150). Investigate
	   improvements *)
	labelsValues = MapAt[Part[labels, Round[#]]&, indicesValues, {partSpec, 1}]; 
	Map[Thread@*Apply[Rule], labelsValues, {Depth[input] - 2}] 
]; 

topProbs[labels_, n_][naInput_NumericArray] := Scope[
	input = Normal[naInput];
	indices = flattenedTopMaxIndices[input, n];
	If[Depth[input] === 2,
		values = Part[input, indices],
		values = MapThread[
			Part, 
			{arrayFlatten[input, Ramp[Depth[input] - 3]], indices}
		]
	];
	ArrayReshape[
		Thread /@ Thread[Map[Part[labels, #] &, indices] -> values],
		Append[Most@arrayDimensions[input], n]
	]
];

topProbs[Automatic][input_NumericArray] := 
	topProbs[makeAutoLabels @ input] @ input;

compiledTopProbs := compiledTopProbs = Compile[ 
	{{input, _Real, 1}}, 
		Module[ 
			{ordering, max, indices}, 
			max = Max[input]; 
			ordering = Reverse@Ordering[input]; 
			indices = Select[ordering, Part[input, #] > 0.1*max&]; 
			{indices, input[[indices]]} 
		], 
		RuntimeAttributes -> {Listable}, 
		Parallelization -> True 
]; 

(*********************)

topDec[labels_, n_][input_NumericArray] := 
	Map[Part[labels, #] &, topMaxIndices[Normal[input], n], {-2}];

topDec[Automatic, n_][input_NumericArray] := 
	topMaxIndices[Normal[input], n];

topMaxIndices[array_, n_] := 
	ArrayReshape[
		flattenedTopMaxIndices[array, n], 
		Append[Most @ arrayDimensions @ array, n]
	];

flattenedTopMaxIndices[array_, n_] := 
	NumericArrayUtilities`PartialOrdering[arrayFlatten[array, Ramp[Depth[array] - 3]], -n];
 
(*********************)

randomSampler[labels_][input_] := 
	Map[RandomChoice[# -> labels]&, Normal @ input, {-2}]; 

randomSampler[Automatic][input_] :=
	randomSampler[makeAutoLabels @ input] @ input;

(*********************)
	
temperatureSampler[t_, labels_][inputNA_NumericArray] := Scope[
	input = Normal[inputNA];
	If[FailureQ[resoftmax = resoftmaxAtTemp[t] @ input],
		decision[labels] @ input,
		randomSampler[labels] @ resoftmax	
	]
];

temperatureSampler[t_, Automatic][input_NumericArray] :=
	temperatureSampler[t, makeAutoLabels @ input] @ input;

resoftmaxAtTemp[temp_][input_] := Scope[
	expRescaledLog = Quiet @ Exp[Log[input] / temp];
	(* For low temperatures, some entries in expRescaledLog may have prob = 0
	   for all classes. In that case we return $Failed and just take the argmax
	   in temperatureSampler *)
	total = Total[expRescaledLog, {-1}];
	If[TrueQ[Min[total] > 0],
		expRescaledLog / total,
		$Failed
	]
];

(*********************)

entropy[in_List] := Scope[
	res = compiledEntropy[Normal /@ in];
	type = NumericArrayType @ First[in];
	If[$ReturnNumericArray, Map[toNA[type]], Identity] @ res
];

compiledEntropy := compiledEntropy = Compile[{{probs, _Real, 1}}, Module[{e = 0.},
	Do[
		If[4.440892098500626`*^-16 <= p <= 1., e -= p * Log[p]],
		{p, probs}
	];
	e
], RuntimeAttributes -> {Listable}, Parallelization -> True];
