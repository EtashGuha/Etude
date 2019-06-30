Inputs: 
	$Input: SequenceT[$$SequenceLength, $$Shape]

Outputs: 
	$Output: SequenceT[$$SequenceLength, $$Shape]

States:
	$State: $$Shape

Parameters:
	$$Shape: RealTensorT
	$$SequenceLength: LengthVar[]

Writer: Function @ Scope[
	input = GetInput["Input", "Timewise"];
	state = GetState["State"];	
	{output, state} = SowRNNLoop[
		List @ SowRamp @ SowPlus[#1, #2]&, 
		{input}, {state}, #$SequenceLength
	];
	SetOutput["Output", output];
	SetState["State", state];
]
