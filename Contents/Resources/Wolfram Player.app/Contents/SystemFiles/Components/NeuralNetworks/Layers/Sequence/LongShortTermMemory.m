Inputs: 
	$Input: SequenceT[$$SequenceLength, VectorT[$$InputSize, AtomT]]

Outputs: 
	$Output: SequenceT[$$SequenceLength, VectorT[$OutputSize]]

States:
	$State: VectorT[$OutputSize]
	$CellState: VectorT[$OutputSize]

Parameters:
	$OutputSize: SizeT
	$Dropout: RecurrentDropoutMethodT
	$$InputSize: SizeT
	$$SequenceLength: LengthVar[]
	$$CellStateConnectedQ: Defaulting[BooleanT, False] (* only for https://bugs.wolfram.com/show?number=348910 *)

Upgraders: {
	"12.0.4" ->
		AddParam[Function["$CellStateConnectedQ" -> False]]
}

ReshapeParams: {$$InputSize, $$SequenceLength}

MaxArgCount: 1

Arrays:
	$InputGateInputWeights: TensorT[{$OutputSize, $$InputSize}]
	$InputGateStateWeights: TensorT[{$OutputSize, $OutputSize}]
	$InputGateBiases: VectorT[$OutputSize]
	$OutputGateInputWeights: TensorT[{$OutputSize, $$InputSize}]
	$OutputGateStateWeights: TensorT[{$OutputSize, $OutputSize}]
	$OutputGateBiases: VectorT[$OutputSize]
	$ForgetGateInputWeights: TensorT[{$OutputSize, $$InputSize}]
	$ForgetGateStateWeights: TensorT[{$OutputSize, $OutputSize}]
	$ForgetGateBiases: VectorT[$OutputSize]
	$MemoryGateInputWeights: TensorT[{$OutputSize, $$InputSize}]
	$MemoryGateStateWeights: TensorT[{$OutputSize, $OutputSize}]
	$MemoryGateBiases: VectorT[$OutputSize]


FusedRNNArray: Function[
	{
		"InputGateInputWeights", "ForgetGateInputWeights", "MemoryGateInputWeights", "OutputGateInputWeights",
		"InputGateStateWeights", "ForgetGateStateWeights", "MemoryGateStateWeights", "OutputGateStateWeights",
		"InputGateBiases", "ForgetGateBiases", "MemoryGateBiases", "OutputGateBiases",
		4 * #OutputSize
	}
]

HasTrainingBehaviorQ: Function[
	StripVP[#Dropout] =!= None
]

Writer: Function @ Scope[
	input = GetInput["Input", "Timewise"];
	state = GetState["State"];
	cellState = GetState["CellState"];
	If[#FusedRNNArray =!= None, (* <- if dropout was None *)
		{output, state, cellState} = SowRNNNode[
			"lstm", GetOutputDims["Output"],
			input, #FusedRNNArray, {state, cellState}
		];
		(* When packed var-length RNN stuff is hooked up, we can derive non-cell state using NthOutput[out, 1], and
		save a SowMetaLast! also do this for the other layers... *)
	,
		{{xidrop, xodrop, xfdrop, xmdrop}, {sidrop, sodrop, sfdrop, smdrop}, sudrop, wdrop} = MakeRNNDropoutData[First @ #Dropout, 4];
		numHidden = #OutputSize;
		(* jeromel: TODO This can be packed by 4 (optimization) *)
		xig = SowMappedFC[xidrop @ input, wdrop @ #InputGateInputWeights, #InputGateBiases, numHidden];
		xog = SowMappedFC[xodrop @ input, wdrop @ #OutputGateInputWeights, #OutputGateBiases, numHidden];
		xfg = SowMappedFC[xfdrop @ input, wdrop @ #ForgetGateInputWeights, #ForgetGateBiases, numHidden];
		xmg = SowMappedFC[xmdrop @ input, wdrop @ #MemoryGateInputWeights, #MemoryGateBiases, numHidden];
		igsw = wdrop @ #InputGateStateWeights;
		ogsw = wdrop @ #OutputGateStateWeights;
		fgsw = wdrop @ #ForgetGateStateWeights;
		mgsw = wdrop @ #MemoryGateStateWeights;
		smdrop @ sfdrop @ sodrop @ sidrop[state]; (* prime the masks in a deterministic order *)
		{output, state, cellState} = SowRNNLoop[
			Function[
				i = SowSigmoid @ SowPlus[#1, SowFC[sidrop @ #5, igsw, None, numHidden]];
				o = SowSigmoid @ SowPlus[#2, SowFC[sodrop @ #5, ogsw, None, numHidden]];
				f = SowSigmoid @ SowPlus[#3, SowFC[sfdrop @ #5, fgsw, None, numHidden]];
				m = SowTanh @    SowPlus[#4, SowFC[smdrop @ #5, mgsw, None, numHidden]];
				c = SowPlus[SowHad[f, #6], SowHad[i, sudrop @ m]];
				s = SowHad[o, SowTanh[c]];
				{s, c}
			],
			{xig, xog, xfg, xmg}, {state, cellState}, #$SequenceLength
		];
	];
	SetOutput["Output", output];
	SetState["State", state];		
	SetState["CellState", cellState]; 
]

Tests: {
	{4, "Input" -> {3, 2}} -> "3*4_Vj7LfXURExY_LUShOwEGM/U=2.185031e+0",
	{4, "Input" -> {"Varying", 2}} -> "3*4_Vj7LfXURExY_VQJXAFORTEg=2.185031e+0",
	{4, "Input" -> {"Varying", 2, Restricted["Integer", 3]}} -> "3*4_J1F3rkHTMpM_Gn890EJHI9I=8.450744e-1"
}
