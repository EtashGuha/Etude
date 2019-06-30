Inputs: 
	$Input: SequenceT[$$SequenceLength, VectorT[$$InputSize, AtomT]]

Outputs: 
	$Output: SequenceT[$$SequenceLength, VectorT[$OutputSize]]

States:
	$State: VectorT[$OutputSize]

Parameters:
	$OutputSize: SizeT
	$Dropout: RecurrentDropoutMethodT
	$$InputSize: SizeT
	$$SequenceLength: LengthVar[]

MaxArgCount: 1

Arrays:
	$InputGateInputWeights: TensorT[{$OutputSize, $$InputSize}]
	$InputGateStateWeights: TensorT[{$OutputSize, $OutputSize}]
	$InputGateBiases: VectorT[$OutputSize]
	$ResetGateInputWeights: TensorT[{$OutputSize, $$InputSize}]
	$ResetGateStateWeights: TensorT[{$OutputSize, $OutputSize}]
	$ResetGateBiases: VectorT[$OutputSize]
	$MemoryGateInputWeights: TensorT[{$OutputSize, $$InputSize}]
	$MemoryGateStateWeights: TensorT[{$OutputSize, $OutputSize}]
	$MemoryGateBiases: VectorT[$OutputSize]

FusedRNNArray: Function[
	{
		"ResetGateInputWeights", "InputGateInputWeights", "MemoryGateInputWeights", 
		"ResetGateStateWeights", "InputGateStateWeights", "MemoryGateStateWeights", 
		"ResetGateBiases", "InputGateBiases", "MemoryGateBiases", 
		3 * #OutputSize
	}
]

HasTrainingBehaviorQ: Function[
	StripVP[#Dropout] =!= None
]

Writer: Function @ Scope[
	input = GetInput["Input", "Timewise"];
	state = GetState["State"];
	If[#FusedRNNArray =!= None, (* <- if we're on GPU and dropout was None *)
		{output, state} = SowRNNNode[
			"gru", GetOutputDims["Output"],
			input, #FusedRNNArray, state
		];
	,
		{{xidrop, xrdrop, xmdrop}, {sidrop, srdrop, smdrop}, sudrop, wdrop} = MakeRNNDropoutData[First @ #Dropout, 3];
		numHidden = #OutputSize;
		xig = SowMappedFC[xidrop @ input, wdrop @ #InputGateInputWeights,  #InputGateBiases, numHidden];
		xrg = SowMappedFC[xrdrop @ input, wdrop @ #ResetGateInputWeights,  #ResetGateBiases, numHidden];
		xmg = SowMappedFC[xmdrop @ input, wdrop @ #MemoryGateInputWeights, #MemoryGateBiases, numHidden];
		igsw = wdrop @ #InputGateStateWeights;
		rgsw = wdrop @ #ResetGateStateWeights;
		mgsw = wdrop @ #MemoryGateStateWeights;
		smdrop @ srdrop @ sidrop[state]; (* prime the masks in a deterministic order *)
		{output, state} = SowRNNLoop[
			Function[
				z = SowSigmoid @ SowPlus[#1, SowFC[sidrop @ #4, igsw, None, numHidden]];
				r = SowSigmoid @ SowPlus[#2, SowFC[srdrop @ #4, rgsw, None, numHidden]];
				h =    SowTanh @ SowPlus[#3, SowHad[r, SowFC[smdrop @ #4, mgsw, None, numHidden]]];
				s = SowMix[sudrop @ h, #4, z];  (* s = (1-z)*s + z*h *)
				{s}
			],
			{xig, xrg, xmg}, {state}, #$SequenceLength
		]
	];
	SetOutput["Output", output];
	SetState["State", state];
]

SummaryFunction: Function[
	GatedRecurrentLayer
]

Tests: {
	{4, "Input" -> {3,2}} -> "3*4_GZ+VnS6rVlk_IvY3AX/1N9U=2.403149e+0",
	{4, "Input" -> {"Varying", 2}} -> "3*4_GZ+VnS6rVlk_OiuvVkqvBWg=2.403149e+0",
	{4, "Input" -> {"Varying", 2, Restricted["Integer", 3]}} -> "3*4_f/DSOL0+Hrk_aykgwF/W7YE=7.930332e+0"
}
