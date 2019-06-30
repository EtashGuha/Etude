Input: VectorT[$$InputLength, AtomT]

Output: TensorT[ListT[NaturalT, SizeT], $$MaxIndex]

Parameters:
	$Elements: ValidatedParameterT[parseOrderingSpec, All]
	$$InputLength: LengthVar[]
	$$MaxIndex: ComputedType[IndexIntegerT[SizeT], maxIndex[$$InputLength], {$Input}]

parseOrderingSpec[spec_] := If[!MatchQ[spec, (n_ /; IntegerQ[n] && n =!= 0) | All],
	FailValidation[OrderingLayer, "specification `` should be either a non-zero integer or All.", spec],
	spec
];

maxIndex[inputLength_] := IndexIntegerT@Switch[inputLength,
	_Integer,   inputLength,
	_LengthVar, Infinity,
	SizeT, SizeT,
	_, $Unreachable
];

ShapeFunction: Map1[OrderingShape[#, StripVP @ $Elements]&]

RankFunction: Function[{1}]

AllowDynamicDimensions: True

Constraints: Function[
	SowConstraint[StripVP[$Elements] === All || $$InputLength >= Abs[StripVP[$Elements]]];
]

Writer: Function @ Scope[
	data = GetInput["Input"];
	lenNode = GetDynamicLengthNode[#$InputLength];
	elements = StripVP[#Elements];

	If[lenNode =!= None, 
		useNegativeMask = TrueQ[elements < 0];
		data = SowSeqMaskBatchwise[data, lenNode, If[useNegativeMask, "-1e37", "1e37"]];
	];

	(* argmin and argamx are slightly faster than topk, so let's use those when possible *)
	ordering = Which[
		elements === 1,
			SowNode["argmin", data, "axis" -> 1, "keepdims" -> True],
		elements === -1,
			SowNode["argmax", data, "axis" -> 1, "keepdims" -> True], 
		elements === All,
			SowNode["argsort", data, "axis" -> -1, "dtype" -> $DTypeMXName],
		elements > 0,
			SowNode["topk", {data}, "axis" -> 1, "k" -> elements, "is_ascend" -> True, "dtype" -> $DTypeMXName],
		elements < 0,
			temp = SowNode["topk", {data}, "axis" -> 1, "k" -> -elements, 
				"is_ascend" -> False, "dtype" -> $DTypeMXName
			];
			SowNode["reverse", {temp}, "axis" -> 1],
		True,
			Panic[] 			
	];
	ordering = SowPlusScalar[ordering, 1]; (* Convert to 1-index *)

	SetOutput["Output", ordering]
]

Tests: {
	{All, "Input" -> {10}} -> "10_TwH6NDevUeU_UfICYULVFPo=5.500000e+1",
	{4, "Input" -> {10}} -> "4_N0af0Nzp8eQ_RpwQ2wXse4Y=2.300000e+1",
	{-4, "Input" -> {10}} -> "4_HHowjqSNUcs_Jjnjs2/Ue/Q=2.400000e+1",
	{1, "Input" -> {10}} -> "1_HW8HQWmqp2c_PB1hYlKrVeU=5.000000e+0",
	{-1, "Input" -> {10}} -> "1_XUzvtya1Pe0_fh57T4oo5n8=8.000000e+0",
	{All, "Input" -> {"Varying"}} -> "3_KgEKcfgmR3c_Nlzzh/FlgFI=6.000000e+0",
	{4, "Input" -> {"x"}} -> "4_N0af0Nzp8eQ_fKcFtOuR7XU=2.300000e+1",
	{-4, "Input" -> {"x"}} -> "4_HHowjqSNUcs_ZI8wvu+Fae0=2.400000e+1",
	{1, "Input" -> {"Varying"}} -> "1_a6h4DOKBKtM_LERp/Xwlg3s=3.000000e+0",
	{-1, "Input" -> {"Varying"}} -> "1_XUZXk1W2EHU_cinkL9JwkkE=2.000000e+0",
	{-1, "Input" -> {"Varying", Restricted["Integer", 100]}} -> "1_XUZXk1W2EHU_PRugy8NGGJc=2.000000e+0",
	{0, "Input" -> 0} -> "Value of 0 given for the elements (first argument) was invalid: specification 0 should be either a non-zero integer or All."
}
