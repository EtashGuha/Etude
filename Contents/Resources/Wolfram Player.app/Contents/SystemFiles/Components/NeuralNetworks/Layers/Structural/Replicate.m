Input: RealTensorT

Output: RealTensorT

Parameters:
	$Specification: ValidatedParameterT[toRepSpec]
	$Level: Defaulting[IntegerT, 1]

ShapeFunction: Map1[ReplicateShape[#, StripVP @ $Specification, $Level]&]

RankFunction: Map1[# + Length[ToList @ StripVP @ $Specification]&]

toRepSpec[{}] := FailValidation[ReplicateLayer, "output array cannot be identical to the input array"];
toRepSpec[list:{Repeated[_Integer ? Positive | Automatic]}] := list;
toRepSpec[other:(_Integer ? Positive | Automatic)] := other;
toRepSpec[spec_] := FailValidation[ReplicateLayer, "specification `` should be a positive integer, Automatic, or a list of these.", spec];

Writer: Function @ Scope[
	oldDims = GetInputDims["Input"];
	outDims = GetOutputDims["Output"];
	extraRank = Length[outDims] - Length[oldDims];
	level = #Level;
	If[level < 0, level = Length[oldDims] + 2 + level];
	newDims = Take[outDims, {level, level + extraRank - 1}];
	ones = CTable[1, extraRank];
	reshapeDims = Flatten @ List[-1, Insert[oldDims, ones, level]];
	reshaped = SowNode["reshape", GetInput["Input"], "shape" -> reshapeDims];
	newAxes = Range[extraRank] + level - 1;
	broadcasted = SowNode["broadcast_axis", reshaped, "axis" -> newAxes, "size" -> newDims];
	SetOutput["Output", broadcasted];	
]

Tests: {
	{1, "Input" -> {2, 3}} -> "1*2*3_SINkZe0REmI_A/KyG112bzs=1.911142e+0",
	{2, "Input" -> {2, 3}} -> "2*2*3_aUAZjJ9aN7M_M0aZQwwSKn8=3.822285e+0",
	{{4, 2}, "Input" -> 2} -> "4*2*2_fwTSrLiIOqg_L4UO8r6Zk6A=6.342879e+0",

	{3, -1, "Input" -> 4} -> "4*3_J77TsSvB1GE_G81/r7UMcNA=5.023027e+0",
	{3, -1, "Input" -> {2, 3}} -> "2*3*3_BB5FnLqEW3o_dMdnyKPcYHo=5.733427e+0",
	{3, 2, "Input" -> {4, 5}} -> "4*3*5_HgvANNtIn/Y_PIZinXe+EN8=2.572131e+1",

	{{2, 2}, -1, "Input" -> 3} -> "3*2*2_DeRXXXdr5GU_ObnZEE/vQLQ=3.619271e+0",
	{{2, 2}, 2, "Input" -> {3, 3}} -> "3*2*2*3_B3uzbRr3lFk_dUqtFC4KmGo=1.284395e+1",

	{Automatic, "Output" -> {3, 2}} -> "3*2_MAhq8kwFiII_MpLCxaABuDY=2.378579e+0",
	{Automatic, "Output" -> {3}} -> "3_FKsM10YEll4_J77TsSvB1GE=1.049447e+0",
	{{Automatic, 2}, "Output" -> {2, 2, 2}} -> "2*2*2_f/8pXwM4MhE_DtXFdZq3psQ=3.171439e+0",
	{{Automatic, Automatic}, "Output" -> {2, 2, 2}} -> "2*2*2_f/8pXwM4MhE_DtXFdZq3psQ=3.171439e+0",
	{{Automatic}, -1, "Output" -> {2, 3}} -> "2*3_bay7sk5FezM_EuF0uxGvHco=2.378579e+0",
	{{Automatic}, -2, "Output" -> {2, 3, 4}} -> "2*3*4_Ybj5SQQu4X0_QQFfykAgyAo=9.327051e+0",

	{{Automatic, Automatic}, "Input" -> {2}, "Output" -> {3, 2}} -> "Inferred inconsistent ranks for output (a matrix versus a rank-3 array).",
	{{Automatic, Automatic, Automatic}, "Output" -> {3, 2}} -> "Type inconsistency in ReplicateLayer: rank of output conflicts with rank implied by input.",
	{{Automatic}, "Input" -> {2}, "Output" -> {1, 2, 3, 4}} -> "Inferred inconsistent ranks for output (a rank-4 array versus a matrix)."
}

Upgraders: {
	"11.3.1" -> RenameParam["$InsertedDimCount" -> "$InsertedDimensionCount"],
	"11.3.2" -> DropParam["OutputSize"] /* DropAllHiddenParams
}
