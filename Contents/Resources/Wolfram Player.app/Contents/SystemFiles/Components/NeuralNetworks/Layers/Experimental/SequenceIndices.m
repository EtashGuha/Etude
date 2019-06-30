Inputs: 
	$Input: SequenceT[$$Length, AnyTensorT]

Outputs: 
	$Output: SequenceT[$$Length,
		ComputedType[
			EitherT[{IndexIntegerT[PosIntegerT], IndexIntegerT[Infinity]}],
			outputType[$$Length, $Maximum],
			{$$Length, $Maximum}
		]
	]

Parameters:
	$Maximum: Defaulting[EitherT[{PosIntegerT, MatchT[Infinity]}], Infinity]
	$$Length: LengthVar[]

outputType[length_, max_] := IndexIntegerT @ If[IntegerQ[length],
	Min[max, length],
	max
];

(*ReshapeParams: {$$Length} ?*)

MaxArgCount: 1

AllowDynamicDimensions: True

Writer: Function @ Scope[

	input = GetInput["Input"];

	rangeIndices = SowNode["_arange", {}, "start" -> "1", "infer_range" -> True, "dtype" -> $DTypeMXName];

	(* Trick to resphape the range of indices *)
	zeros = SowNode["_zeros", {}, "shape" -> "(1,)", "dtype" -> $DTypeMXName];
	zeros = SowNode["broadcast_like", {zeros, input}, "lhs_axes" -> "(0,)", "rhs_axes" -> "(1,)"];
	rangeIndices = SowPlus[rangeIndices, zeros];

	(* Clip indices *)
	If[NumericQ[#Maximum],
		rangeIndices = SowMinScalar[rangeIndices, #Maximum];
	];

	(* Reshape *)
	rangeIndices = SowNode["expand_dims", rangeIndices, "axis" -> 0];
	rangeIndices = SowNode["broadcast_like", {rangeIndices, input}, "lhs_axes" -> "(0,)", "rhs_axes" -> "(0,)"];

	SetOutput["Output", rangeIndices];
]

Tests: {
	{"Input" -> {"Varying"}}				-> "3_bXKx7ICsvVQ_I6pZ9uW/YPQ=6.000000e+0",
	{"Input" -> {"Varying", 2}}				-> "3_bXKx7ICsvVQ_I6pZ9uW/YPQ=6.000000e+0",
	{"Input" -> {"Varying", 2, 3, 4}}		-> "3_bXKx7ICsvVQ_I6pZ9uW/YPQ=6.000000e+0",
	{2, "Input" -> {"Varying"}}				-> "3_OBrP6oYyq7A_MIjU9MKFsHw=5.000000e+0",
	{2, "Input" -> {"Varying", 2}}			-> "3_OBrP6oYyq7A_MIjU9MKFsHw=5.000000e+0",
	{3, "Input" -> {"Varying", 2, 1, 1}}	-> "3_bXKx7ICsvVQ_Wh+ieGZp0oc=6.000000e+0",
	{1*^6, "Input" -> {"Varying", 2, 3, 4}}		-> "3_bXKx7ICsvVQ_I6pZ9uW/YPQ=6.000000e+0",
	{"Input" -> {6}}						-> "6_ft11epNyzIw_YFexeI4+EwY=2.100000e+1",
	{"Input" -> {6, 2, 2}}					-> "6_ft11epNyzIw_YFexeI4+EwY=2.100000e+1",
	{3, "Input" -> {6}}						-> "6_JtUJ+qBWHDQ_Z5utPVTNHQY=1.500000e+1",
	{3, "Input" -> {6, 2, 2}}				-> "6_JtUJ+qBWHDQ_Z5utPVTNHQY=1.500000e+1",
	{1*^6, "Input" -> {6}}					-> "6_ft11epNyzIw_YFexeI4+EwY=2.100000e+1",
	{1*^6, "Input" -> {6, 2, 2}}			-> "6_ft11epNyzIw_YFexeI4+EwY=2.100000e+1",
	{0, "Input" -> {"Varying", 2}}			-> "Value given for the maximum (first argument) should be either a positive integer or Infinity, but was 0 instead."
}
