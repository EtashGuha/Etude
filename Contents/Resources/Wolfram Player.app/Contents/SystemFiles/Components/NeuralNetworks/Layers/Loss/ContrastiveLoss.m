Inputs: 
	$Input: ScalarT
	$Target: NetEncoder["Boolean"]

Outputs:
	$Loss: ScalarT 

Parameters:
	$Margin: Defaulting[ScalarT, 0.5]

Writer: Function @ Scope[
	input = GetInput["Input"];
	target = GetInput["Target"];
	loss = SowMix[input, SowMarginLoss[input, #Margin], target];
	SetOutput["Loss", loss];
]

IsLoss: True

Tests: {
	{} -> "_UlS0Sd7vNAo_NbWLHOJvuoU=3.498157e-1",
	{0.9} -> "_UlS0Sd7vNAo_drz5wk92MIU=3.498157e-1"
}

