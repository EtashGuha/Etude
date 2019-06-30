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
	$InputWeights: TensorT[{$OutputSize, $$InputSize}]
	$StateWeights: TensorT[{$OutputSize, $OutputSize}]
	$Biases: VectorT[$OutputSize]

PostConstructionFunction: Function[
	If[MatchQ[StripVP @ $Dropout, KeyValuePattern["UpdateVector" -> n_ /; n > 0]],
		FailConstruction["\"UpdateVector\" dropout method is not applicable to BasicRecurrentLayer."]
	]
]

FusedRNNArray: Function[
	{"InputWeights", "StateWeights", "Biases", #OutputSize}
]

HasTrainingBehaviorQ: Function[
	StripVP[#Dropout] =!= None
]

Writer: Function @ Scope[
	input = GetInput["Input", "Timewise"];
	state = GetState["State"];	
	If[#FusedRNNArray =!= None, 
		{output, state} = SowRNNNode[
			"rnn_tanh", GetOutputDims["Output"],
			input, #FusedRNNArray, state
		];
	,
		numHidden = #OutputSize;
		{{xdrop}, {sdrop}, sudrop, wdrop} = MakeRNNDropoutData[StripVP @ #Dropout, 1];
		xg = SowMappedFC[xdrop @ input, wdrop @ #InputWeights, #Biases, numHidden];
		sw = wdrop @ #StateWeights;
		sdrop[state]; (* prime the mask *)
		{output, state} = SowRNNLoop[
			List @ SowTanh @ SowPlus[#1, SowFC[sudrop @ sdrop @ #2, sw, None, numHidden]]&,
			{xg}, {state}, #$SequenceLength
		]
	];
	SetOutput["Output", output];
	SetState["State", state];
]

Tests: {
	{4, "Input" -> {3,2}} -> "3*4_W3d/l8OrNmk_cO9y4bq8eFc=1.006620e+1",
	{4, "Input" -> {"Varying",2}} -> "3*4_W3d/l8OrNmk_WwyfiKjZRMk=1.006620e+1",
	{4, "Input" -> {"Varying",2,Restricted["Integer", 3]}} -> "3*4_Ul5XLqwE1Pc_KQaDzMvUQCY=7.871475e+0"
}
