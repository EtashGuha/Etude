Inputs: 
	$Input: SequenceT[$$Length, $$Shape]
	$Element: $$Shape

Output: SequenceT[$$OutputLength, $$Shape]

Parameters:
	$$Length: LengthVar[]
	$$Shape: AnyTensorT
	$$OutputLength: ComputedType[LengthVar[], $$Length + 1]

Constraints: Function[
	SowConstraint[$$OutputLength == $$Length + 1]
]

Writer: Function @ Scope[
	lenNode = GetDynamicLengthNode[#$Length];
	appendee = GetInput["Element"];

	If[lenNode === None,
		out = SowAppend[GetInput["Input"], appendee]
		,
		timewise = SowSeqReverse[GetInput["Input", "Timewise"], lenNode];
		timewise = SowJoin[SowInsertDim[appendee, 0], timewise, 0];
		lnode2 = SowDerivedSequenceLengthNode[lenNode, #$OutputLength, # + 1&];
		timewise = SowSeqReverse[timewise, lnode2];
		out = ToMetaNode[timewise, #$OutputLength, True];
	];
	SetOutput["Output", out];
]

Tests: {
	{"Input" -> {"Varying", 2}} -> "4*2_Liad5oMW3eI_Oh0C2qE1ZYE=3.109017e+0",
	{"Input" -> {"Varying"}} -> "4_N6dBguLAOZ8_Flt9MeAY6Vg=1.674342e+0",
	{"Input" -> {"Varying", 2, 3}} -> "4*2*3_Luzvstmun4A_XWnBnUnxxQs=1.067583e+1",
	{"Input" -> {3, 2, 3}} -> "4*2*3_Luzvstmun4A_HBA8kTLmEqE=1.067583e+1",
	{"Input" -> {"Varying", Restricted["Integer", 10]}} -> "4_Qd4r01t9q/s_NjsHhnjtsbI=3.400000e+1"
}