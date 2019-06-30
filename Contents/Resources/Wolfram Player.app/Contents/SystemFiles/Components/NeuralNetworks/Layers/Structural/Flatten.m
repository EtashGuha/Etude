Input: AnyTensorT

Output: AnyTensorT

Parameters:
	$Level: NormalizedT[EitherT[{IntegerT, MatchT[Infinity]}], checkNotZero, Infinity]

AllowDynamicDimensions: True

ShapeFunction: Map1[FlattenShape[#, $Level]&]

RankFunction: Map1[FlattenRank[#, $Level]&]

TypeFunction: Identity

FinalCheck: Function @ Scope[
	dims = FlattenShape[TDimensions[$Input], $Level];
	If[ContainsQ[dims, $Failed], 
		FailValidation[FlattenLayer, "dynamic dimensions cannot be flattened with other levels."]];
]

checkNotZero[0] := FailValidation[FlattenLayer, "level specification should be a non-zero integer."];
checkNotZero[e_] := e;

Writer: Function @ Scope[
	odims = GetOutputDims["Output"];
	If[#Level === Infinity && VectorQ[odims, IntegerQ], MXWriteDefaultAndReturn[]];
	id = SowNode["reshape", GetInput["Input"], "shape" -> Prepend[odims /. _LengthVar -> 0, 0]];
	SetOutput["Output", id];	
]

MXNet:
	Name: "flatten"
	Aliases: {"Flatten"}

inf = Infinity;
Tests: {
	{"Input" -> {3, 3}}           -> "9_QX4+1gAwAAs_FV86edJNxyw=3.210989e+0",
	{inf, "Input" -> {3, 3}}      -> "9_QX4+1gAwAAs_FV86edJNxyw=3.210989e+0",
	{1, "Input" -> {3, 3}}        -> "9_QX4+1gAwAAs_FV86edJNxyw=3.210989e+0",
	{1, "Input" -> {3, 3, 3}}     -> "9*3_GK0x+JYTZFo_HPby7MRz6QM=1.168612e+1",
	{inf, "Input" -> {3, 3, 3}}   -> "27_L+sl+aBXySM_TxW4wMfFyC0=1.168612e+1",
	{2, "Input" -> {3, 3, 3}}     -> "27_L+sl+aBXySM_TxW4wMfFyC0=1.168612e+1",
	{-2, "Input" -> {3, 3, 3}}    -> "27_L+sl+aBXySM_TxW4wMfFyC0=1.168612e+1",
	{-2, "Input" -> {3, 3, 3, Restricted["Integer", 4]}}    -> "27_BYzPMu2Qq5A_OCC9K3vrBsw=6.900000e+1",
	{-1, "Input" -> {3, 3, 3}}    -> "3*9_VL+OaXWNx0M_ebN0bgjfa2M=1.168612e+1",
	{4, "Input" -> {3, 3, 3}} -> "Type inconsistency in FlattenLayer: level specification of 4 is incompatible with input tensor, which has rank 3."
}

Upgraders: {
	"11.3.1" -> RenameParam["OutputSize" -> "$OutputSize"],
	"11.3.2" -> DropAllHiddenParams /* setVectorOut
}

setVectorOut[p_] := Scope[
	If[p["Parameters", "Level"] === Infinity && p["Outputs", "Output"] === RealTensorT, 
		p["Outputs", "Output"] = VectorT[];
	];
	p
];
