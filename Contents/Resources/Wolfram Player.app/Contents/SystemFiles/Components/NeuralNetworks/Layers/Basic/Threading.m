Inputs: 
	$Multiport: RealTensorT (* TODO: handle integers *)

Output: RealTensorT

Parameters:
	$Function: NAryElementwiseFunctionT

ShapeFunction: EqualityShape

RankFunction: EqualityRank

PostConstructionFunction: Function @ Scope[
	count = Match[
		First @ $Function, 
		s_ScalarFunctionObject :> s["ArgumentCount"],
		Times | Plus | Min | Max :> Indefinite,
		2
	];
	If[IntegerQ[count], PCFExpandMultiport[count]];
]

AllowDynamicDimensions: True

Writer: Function @ Scope[
	inputs = GetInput[All];
	SetOutput["Output", SowScalarFunction[First @ #Function, Sequence @@ inputs]];
]

SummaryFunction: Function @ Scope[
	func = First[#Function];
	If[SymbolQ[func], func, ThreadingLayer]
]

Tests: {
	{Plus, "Inputs" -> {"Real", "Real"}} -> "_AqJ2ueHiI3M_CtS5uSw8pPc=7.928598e-1",
	(* TODO {Plus, "Inputs" -> {"Integer", Restricted["Integer", 3]}} -> "_AxM7g+UIh2A_AhrlzyaNCwI=1.308400e+4", *)
	{Plus, "Inputs" -> {3, 3}} -> "3_RfcM8A5RZGc_fx9hIWnBBJk=1.911142e+0",
	{Times, "Inputs" -> {3, 3}} -> "3_UFuI/OphZaw_D9jZep9XXgg=3.167708e-1",
	{#1 + #2^#1 & , "Input" -> {3, 3}} -> "3_E3+NLGSR10E_UYXmVo33y6k=2.934101e+0",
	{Max, "Inputs" -> {{2}, {2}}} -> "2_YTB/HRE2rQ0_AvDK/KbJbRI=1.119340e+0", (* <- Max isn't Listable *)
	{Plus, "Inputs"->{{4, 3}, {4, 3}}} -> "4*3_bgu8m2mVlVg_AFoK/6uUkKo=1.067583e+1",
	{Plus, "Inputs" -> {{"a", 3}, {"a", 3}}} -> "3*3_AW13OEwY/QU_O3x3Us5Snsw=7.379773e+0",
	{Plus, "Inputs" -> {{"a", 3}, {"a", 3}}} -> "3*3_AW13OEwY/QU_O3x3Us5Snsw=7.379773e+0",
	{Zeta, "Inputs" -> {3, 3}} -> "Zeta could not be symbolically evaluated as a binary scalar function.",

	(* more than 2 inputs: *)
	{#1 + #2*#3 & , "Output" -> 3} -> "3_Lg87FpSPitA_UlJkzEpOzTY=1.253307e+0",
	{Times, "Inputs" -> {3, 3, 3, 3, 3}} -> "3_K3so5k3uvF4_UwjrmoYsct4=1.804183e-2",
	{#1 + Ramp[#2]*#3 & , "Inputs" -> {3, 3, 3}} -> "3_Lg87FpSPitA_UlJkzEpOzTY=1.253307e+0",
	{#1 + "SELU"[#2]*#3 & , "Inputs" -> {3, 3, 3}} -> "3_VDaQ5pYX/rc_A2JSlDZWWik=1.270975e+0"
}

Upgraders: {
	"11.3.2" -> DropAllHiddenParams /* UpgradeToMultiport
}
