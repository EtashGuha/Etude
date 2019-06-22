Inputs: 
	$Multiport: RealTensorT (* TODO: handle integers -- see TypeFunction below *)

Output: RealTensorT

Parameters:
	$Level: Defaulting[PosIntegerT, 1]

ShapeFunction: List[CatenateShape[#, $Level]]&

RankFunction: List[CatenateRank[#, $Level]]&

(*
TypeFunction: Function[{Head[First @ #] @ looserType @ Map[Last, #]}]

SetAttributes[looserType, Orderless];
looserType[RealT, _] := RealT;
looserType[AtomT, other_] := other;
looserType[IndexIntegerT[a_], IndexIntegerT[b_]] := IndexIntegerT[Max[a,b]];
looserType[IntegerT, IndexIntegerT[_]] := IntegerT;
looserType[types_List] := Fold[looserType, First[types], Rest[types]];
*)

Writer: Function @ Scope[
	level = #Level;
	odim = GetOutputDims["Output"]; orank = Length[odim];
	idims = GetInputDims[All]; iranks = Length /@ idims;
	If[level === 1 && SameQ @@ iranks, 
		If[DynamicDimsQ @ idims[[All, 1]],
			(* todo: move this into FinalCheck *)
			FailValidation[CatenateLayer, "cannot catenate on level 1 as one or more of the input arrays is \"Varying\" on that level."];
		];
	];
	maxRank = Max[iranks];
	inputs = GetInput[All];
	reference = Part[inputs, MaxIndex[iranks]];
	nodes = MapThread[SowBroadcastAgainst[#1, Range[1, maxRank - #2], reference]&, {inputs, iranks}];
	catLevel = toCatLevel[iranks, orank, level];
	out = SowJoin[Sequence @@ nodes, catLevel];
	SetOutput["Output", out];
]

toCatLevel[iranks:{__Integer}, orank_Integer, level_] := level + orank - Min[iranks];
toCatLevel[___] := $Failed;

AllowDynamicDimensions: True

Tests: {
	{"Inputs" -> {3, 3}} -> "6_bmV1lCuQViA_Pz28xF7/94U=1.911142e+0",
	{"Inputs" -> {3, 3, 3}} -> "9_QX4+1gAwAAs_FV86edJNxyw=3.210989e+0",
	{"Inputs" -> {{2, 3}, 4}} -> "2*7_bB1w5Tw+z9U_bA5+L2TA5qQ=5.968538e+0",
	{1, "Inputs" -> {{2, 3}, {2, 3}}} -> "4*3_dSrbrONjgxw_VHhk2ne/8wg=5.231537e+0",
	{2, "Inputs" -> {{2, 3}, {2, 3}}} -> "2*6_eKEyj6rNQcE_RA39mF2jL98=5.231537e+0",
	{3, "Inputs" -> {{2, 3}, {2, 3}}} -> "Type inconsistency in CatenateLayer: specified level (3) cannot exceed rank of lowest-rank input (2).",
	{2, "Inputs" -> {{"a", 3}, {"a", 3}}} -> "3*6_a3ztEUY5/mE_T/CPSWcxTkc=7.379773e+0",
	{1, "Inputs" -> {{"a", 3}, {2}}} -> "3*5_PLVPrYjTx0I_VZ+cVskOdiA=6.493268e+0",
	(* TODO {1, "Inputs" -> {{2, 3, Restricted["Integer", 3]}, {2, 3, Restricted["Integer", 3]}}} -> "4*3_aEKaX0XU8wY_eKoIVOhrnKg=2.500000e+1", *)
	{1, "Inputs" -> {{"a", 3}, {"a", 3}}} -> "Validation failed for CatenateLayer: cannot catenate on level 1 as one or more of the input arrays is \"Varying\" on that level."
}

MXNet:
	Name: "concat"
	Writer: Function[{
		"num_args" -> IntegerString[Length[#$InputShapes]]
	}]
	Aliases: {"Concat"}

Upgraders: {
	"11.3.2" -> DropAllHiddenParams /* UpgradeToMultiport
}
