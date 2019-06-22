Package["NeuralNetworks`"]



PackageExport["NetTrainBenchmarkScore"]

$MBProScores = Association[
	"LSTM" -> 5142.46, "LSTMLarge" -> 97.9946, 
	"LSTMLargeStack" -> 17.5732, "LSTMStack" -> 1320.44, 
	"GRU" -> 7650.61, "GRULarge" -> 139.972, "GRUStack" -> 1872.22, 
	"RNN" -> 19446.6, "RNNLarge" -> 405.387, "RNNStack" -> 6105.95, 
	"LeNet" -> 92.3582, "Conv3x3" -> 24.5079, "Linear" -> 66158.7, 
	"LinearDeep" -> 14369.4, "LinearWide" -> 196.44, 
	"Sorting" -> 221.604
];

Options[NetTrainBenchmarkScore] = {
	TargetDevice -> "CPU",
	PerformanceGoal -> Automatic,
	WorkingPrecision -> "Real32",
	"Rounds" :> Quantity[5, "Seconds"]
}

NetTrainBenchmarkScore[OptionsPattern[]] := Scope[
	UnpackOptions[targetDevice, rounds, workingPrecision, performanceGoal];
	times = NetTrainBenchmarkTiming[All, 
		TargetDevice -> targetDevice, 
		"Rounds" -> rounds, 
		"ShowProgress" -> False,
		WorkingPrecision -> workingPrecision,
		PerformanceGoal -> performanceGoal
	];
	Round[100 * times / $MBProScores]
];

NetTrainBenchmarkScore[name_String, opts:OptionsPattern[]] := Scope[
	UnpackOptions[targetDevice, rounds];
	time = NetTrainBenchmarkTiming[name, 
		TargetDevice -> targetDevice, "Rounds" -> rounds, "ShowProgress" -> False];
	Round[100 * time / Lookup[$MBProScores, name, Return[$Failed]]]
];


PackageExport["NetTrainBenchmarkTiming"]
PackageExport["NetTrainBenchmarkDuration"]

Options[NetTrainBenchmarkTiming] = {
	"Rounds" :> Quantity[10, "Seconds"],
	"Batches" -> 4,
	"SequenceLength" -> 32,
	TargetDevice -> "CPU",
	BatchSize -> 64,
	"ShowProgress" -> True,
	PerformanceGoal -> Automatic,
	WorkingPrecision -> "Real32"
};

NetTrainBenchmarkTiming::notype = "Don't know how to synthesize data for type ``."
NetTrainBenchmarkTiming::notnet = "Input was not a net."

stack[layer_, n_, isize_] := NetChain[Table[layer, n], "Input" -> {Automatic, isize}];
lstm[sz_, in_:Automatic] := LongShortTermMemoryLayer[sz, "Input" -> {Automatic, in}];
gru[sz_, in_:Automatic] := GatedRecurrentLayer[sz, "Input" -> {Automatic, in}];
rnn[sz_, in_:Automatic] := BasicRecurrentLayer[sz, "Input" -> {Automatic, in}];

$NamedBenchmarkNets = Association[
	"LSTM" :> lstm[32, 16],
	"LSTMLarge" :> lstm[1024, 16],
	"LSTMLargeStack" :> stack[lstm[1024], 4, 16],
	"LSTMStack" :> stack[lstm[32], 4, 16],
	"GRU" :> gru[32, 16],
	"GRULarge" :> gru[1024, 16],
	"GRUStack" :> stack[gru[32], 4, 16],
	"RNN" :> rnn[32, 16],
	"RNNLarge" :> rnn[1024, 16],
	"RNNStack" :> stack[rnn[32], 4, 16],
	"LeNet" :> NetChain[{
		ConvolutionLayer[20, 5], Ramp, PoolingLayer[2],
		ConvolutionLayer[50, 5], Ramp, PoolingLayer[2],
		FlattenLayer[], 500, Ramp, 10, SoftmaxLayer[]},
		"Input" -> {1, 28, 28}
	],
	"Conv3x3" :> NetChain[
		{ConvolutionLayer[4,3], ConvolutionLayer[32,3], ConvolutionLayer[4,3], FlattenLayer[], 1},
		"Input" -> {1,64,64}
	],
	"Linear" :> NetChain[{5, 100, Ramp, 200, Tanh, 50, Ramp, 5, SoftmaxLayer[]}, "Input" -> 5],
	"LinearDeep" :> NetChain[5 * Mod[Range[100], 3, 1], "Input" -> 5],
	"LinearWide" :> NetChain[{10,1000,5000,5000,1000,10}, "Input" -> 5],
	"Sorting" :> Block[{n = 4}, NetGraph[<|
		"enc" -> {EmbeddingLayer[2*n], LongShortTermMemoryLayer[50]},
		"dec" -> {LongShortTermMemoryLayer[50]},
		"attend" -> AttentionLayer[], 
		"cat" -> CatenateLayer[2], 
		"classify" -> {NetMapOperator[LinearLayer[n]], SoftmaxLayer[]}|>, 
		{"enc" -> "dec" -> NetPort["attend", "Query"], 
		 "enc" -> {NetPort["attend", "Key"], NetPort["attend", "Value"]}, 
		 "dec" -> "cat", "attend" -> "cat", "cat" -> "classify"}, 
		"Input" -> TensorT[{LengthVar[0]}, IndexIntegerT[n]], 
		"Output" -> TensorT[{LengthVar[0]}, NetDecoder[{"Class", Range[n]}]]
	]]
];

PackageExport["NamedBenchmarkNet"]

NamedBenchmarkNet[net_String] := Lookup[$NamedBenchmarkNets, net];

NetTrainBenchmarkTiming[All, opts:OptionsPattern[]] :=
	NetTrainBenchmarkTiming[Keys[$NamedBenchmarkNets], opts];

NetTrainBenchmarkTiming[list_List ? StringVectorQ, opts:OptionsPattern[]] := Scope[
	time = None;
	AssociationThread[list, dynamicMap[Function[
		If[time =!= None, Pause[time]];
		{time, res} = AbsoluteTiming[NetTrainBenchmarkTiming[#, opts]]; res
		], list
	]]
];

dynamicMap[f_, list_] := Map[f, list];

dynamicMap[f_, list_] /; $Notebooks := Scope[
	i = 1; 
	SetupTeardown[
		cell = PrintTemporary @ Dynamic[
			Row[{ProgressIndicator[i, {1, Length[list]+1}], "   ", list[[i]]}],
			TrackedSymbols :> {i}, DefaultBaseStyle -> {FontFamily -> CurrentValue["PanelFontFamily"]}]
		,
		Table[i = j; f[list[[j]]], {j, 1, Length[list]}]
		,
		NotebookDelete[cell]
	]
];

NetTrainBenchmarkTiming[str_String, opts:OptionsPattern[]] := Scope[
	If[!$Notebooks, Print["Benchmarking ", str]];
	res = NetTrainBenchmarkTiming[Lookup[$NamedBenchmarkNets, str, ReturnFailed["notnet"]], opts];
	If[!$Notebooks, Print["\tInputs/s = ", res]];
	res
];

NetTrainBenchmarkTiming[net_, OptionsPattern[]] := CatchFailureAsMessage @ Scope[
	If[!ValidNetQ[net], ThrowFailure["notnet"]];
	UnpackOptions[rounds, batches, sequenceLength, targetDevice, batchSize, showProgress, performanceGoal, workingPrecision];
	dataSize = batches * batchSize;
	tnet = NetAttachLoss[net, Automatic];
	SeedRandom[1];
	tdata = synthesizeData /@ Inputs[tnet];
	NetTrain[tnet, tdata, "MeanExamplesPerSecond",
		LossFunction -> All,
		BatchSize -> batchSize, MaxTrainingRounds -> rounds,
		TargetDevice -> targetDevice, 
		TrainingProgressReporting -> If[TrueQ @ showProgress, {"ProgressIndicator", "Interval" -> 1.0}, None],
		WorkingPrecision -> workingPrecision,
		PerformanceGoal -> performanceGoal
	]
]

synthesizeData[TensorT[dims_, RealT]] := 
	RandomReal[1, Prepend[dims, dataSize] /. _LengthVar -> sequenceLength];

synthesizeData[TensorT[{n_}, EncoderP["Class", assoc_]]] :=
	RandomChoice[assoc["Labels"], {dataSize, n} /. _LengthVar -> sequenceLength];

synthesizeData[IndexIntegerT[n_]] := RandomInteger[{1, n}, dataSize];

synthesizeData[TensorT[dims_, IndexIntegerT[n_]]] := 
	RandomInteger[{1,n}, Prepend[dims, dataSize] /. _LengthVar -> sequenceLength];

synthesizeData[t_] := ThrowFailure["notype", t];

(*----------------------------------------------------------------------------*)
PackageExport["NetTrainBenchmarkScoreReport"]

Options[NetTrainBenchmarkScoreReport] = {
	TargetDevice -> "CPU",
	PerformanceGoal -> Automatic,
	WorkingPrecision -> "Real32"
}

SetUsage @ "
NetTrainBenchmarkScoreReport[] returns an Association with the system information \
and the result of NetTrainBenchmarkScoreReport[].

The following options are available:
|TargetDevice | 'CPU' | Device to run on. |
|PerformanceGoal | Automatic | NetTrain PerformanceGoal. |
|WorkingPrecision | 'Real32' | NetTrain WorkingPrecision option. |
"

NetTrainBenchmarkScoreReport[opts:OptionsPattern[]] := Scope[
	UnpackOptions[targetDevice, performanceGoal, workingPrecision];
	Print["Warmup Round..."];
	(* in my experiments, there is quite a difference the second time NetTrainBenchmarkScore is run *)
	NetTrainBenchmarkScore[opts, "Rounds" -> Quantity[0.01, "Seconds"]];
	PrintTemporary["Warmup Finished."];
	scores = NetTrainBenchmarkScore[opts];
	(* change All to explicit list for mongo compatibility. Could also change to string *)
	targ = Replace[targetDevice, All -> Range[$GPUCount]];
	<|
		"Scores" -> scores,
		"WorkingPrecision" -> ToString[workingPrecision],
		"PerformanceGoal" -> ToString[performanceGoal],
		"TargetDevice" -> targ,
		"MachineInformation" -> NNMachineInformation[TargetDevice -> targetDevice],
		"PacletInformation" -> nnBenchmarkPacletInfo[]
	|>
]

