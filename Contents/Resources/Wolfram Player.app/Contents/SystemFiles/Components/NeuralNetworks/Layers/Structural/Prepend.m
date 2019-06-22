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
	prependee = GetInput["Element"];
	input = GetInput["Input"];
	out = SowPrepend[input, prependee];
	SetOutput["Output", out];

	lenNode = GetDynamicLengthNode[#$Length];
	If[lenNode =!= None,			
		SowDerivedSequenceLengthNode[lenNode, #$OutputLength, # + 1&];
	];	
]

Tests: {
	{"Input" -> {"Varying", 2}} -> "4*2_UsdLzVjz53w_EMYSykJB4KE=3.109017e+0",
	{"Input" -> {"Varying"}} -> "4_d+9r7oLKKiE_QQ82OX9/SA8=1.674342e+0",
	{"Input" -> {"Varying", 2, 3}} -> "4*2*3_IgCAMtOhuJk_fyP1QP7QviM=1.067583e+1",
	{"Input" -> {3, 2, 3}} -> "4*2*3_IgCAMtOhuJk_bqJxoC57Nto=1.067583e+1",
	{"Element" -> 4} -> "4*4_OZtzpSmjO8M_CeDAlOR/Sr4=6.833459e+0"
}