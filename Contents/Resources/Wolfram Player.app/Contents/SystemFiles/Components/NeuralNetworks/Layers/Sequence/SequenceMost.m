Input: SequenceT[$$Length, $$Shape]

Output: SequenceT[$$OutputLength, $$Shape]

Parameters:
	$$Length: LengthVar[]
	$$Shape: AnyTensorT
	$$OutputLength: ComputedType[LengthVar[], 
		If[$$Length > 1, $$Length - 1, FailValidation["input array must be at least length 1."]],
		$$Length, 
		IntegerQ[$$Length]
	]

Constraints: Function[
	SowConstraint[$$Length > 1];
	SowConstraint[$$OutputLength == $$Length - 1]
]

Writer: Function @ Scope[
	input = GetInputMetaNode["Input"];
	most = SowMetaDrop[input, #$OutputLength, False];
	SetOutput["Output", most];
]

Tests: {
	{"Input" -> 4} -> "3_AIvAa0t3efE_fSBwZO7kXLY=9.048177e-1",
	{"Input" -> {3, 5}} -> "2*5_M6g8ur0bTdU_SSxhQeurDpc=3.939840e+0",
	{"Input" -> "x"} -> "9_QX4+1gAwAAs_cX/lQ3pAV40=3.210989e+0",
	{"Input" -> {"x", 2}} -> "9*2_W/FQGCwZyf0_BBi5rIX6KIQ=7.379773e+0",
	{"Input" -> {"x", 2, 2}} -> "9*2*2_GVcOjQ2r29Q_e0y1eIrznKg=1.622493e+1",
	{"Input" -> {"x", Restricted["Integer", 3]}} -> "9_CrfCegpXArU_AgB5V/K5OWY=1.900000e+1"
}

Upgraders: {
	"11.3.1" -> RenameParam["$LengthOut" -> "$OutputLength"]
}
