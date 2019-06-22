Inputs: 
	$Multiport: RealTensorT (* TODO: handle integers *)

Output: RealTensorT

ShapeFunction: EqualityShape

RankFunction: EqualityRank

AllowDynamicDimensions: True

MXNet:
	Name: "ElementWiseSum"
	Writer: Function[
		"num_args" -> IntegerString @ GetInputCount[]
	]

Tests: {
	{"Inputs" -> {"Real", "Real"}} -> "_AqJ2ueHiI3M_CtS5uSw8pPc=7.928598e-1",
	{"Inputs" -> {{3, 3}, {3, 3}}} -> "3*3_AW13OEwY/QU_X9SMT3zHl9k=7.379773e+0",
	{"Inputs" -> {2, 2, 2}} -> "2_X+AT7Z4kShU_OlocKghMgJo=1.911142e+0",
	{"Inputs" -> {{"a", 2}, {"a", 2}}} -> "3*2_TNFS5EJ1brk_FLM02S5wGis=5.231537e+0"
	(* TODO {"Inputs" -> {{"a", 2, Restricted["Integer", 3]}, {"a", 2, Restricted["Integer", 3]}}} -> "3*2_S9uVQzZxa7Y_MqO7FvLCjm8=2.500000e+1" *)
}

Upgraders: {
	"11.3.2" -> DropAllHiddenParams /* UpgradeToMultiport
}
