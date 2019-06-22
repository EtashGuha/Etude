Package["NeuralNetworks`"]

$cachedDataDir := $cachedDataDir = EnsureDirectory[{$NNCacheDir, "TrainingData"}];

Clear[resolveNamedNet];

resolveNamedNet[name_] := Scope[
	If[KeyExistsQ[$BuiltinNets, name],
		BlockRandom[
			SeedRandom[1];
			res = resolveNamedNet[name] = $BuiltinNets[name];
		];
		Return[res]
	];
	lobj = LocalObject[name];
	If[FileExistsQ[lobj], Return @ Import @ lobj];
	names = NetModelNames[];
	If[!StringVectorQ[names], ThrowFailure["nonmind"]];
	If[!MemberQ[names, name],
		match = SelectFirst[names, StringStartsQ[#, name, IgnoreCase -> True]&];
		(* find 'Foo Trained on Bar' if given just 'Foo' *)
		If[StringQ[match], name = match]
	];
	NetModelInternal[NetTrain, name, "UninitializedEvaluationNet"]
];


PackageScope["$BuiltinNets"]

$BuiltinNets = <||>;

$BuiltinNets["SentimentAnalysisTutorialNet"] := NetChain[
	{EmbeddingLayer[40], DropoutLayer[0.3], LongShortTermMemoryLayer[20], SequenceLastLayer[], LinearLayer[2], SoftmaxLayer[]},
	"Input" -> NetEncoder[{"Tokens"}], "Output" -> NetDecoder[{"Class", {"positive", "negative"}}]
];

$BuiltinNets["IntegerAdditionTutorialNet"] := Scope[
	targetEnc = NetEncoder[{"Characters", {DigitCharacter, {StartOfString, EndOfString} -> Automatic}}];
	inputEnc = NetEncoder[{"Characters", {DigitCharacter, "+"}}];
	encoderNet = NetChain[{UnitVectorLayer[], GatedRecurrentLayer[150], SequenceLastLayer[]}];
	decoderNet = NetGraph[{
		UnitVectorLayer[11], SequenceMostLayer[],
		GatedRecurrentLayer[150], NetMapOperator[LinearLayer[]], SoftmaxLayer[]},
		{NetPort["Input"] -> 1 -> 2 -> 3 -> 4 -> 5, NetPort["State"] -> NetPort[3, "State"]}
	];
	trainingNet = NetGraph[
		<|"encoder" -> encoderNet, "decoder" -> decoderNet, "loss" -> CrossEntropyLossLayer["Index"], "rest" -> SequenceRestLayer[]|>,
		{NetPort["Input"] -> "encoder" -> NetPort["decoder","State"], NetPort["Target"] -> NetPort["decoder","Input"],
		"decoder" -> NetPort["loss","Input"], NetPort["Target"] -> "rest" -> NetPort["loss","Target"]}, 
		"Input" -> inputEnc,"Target"->targetEnc
	];
	trainingNet
]

$BuiltinNets["IntegerSortingTutorialNet"] := Scope[
	digits = Range[6];
	net = NetGraph[<|
			"enc" -> {EmbeddingLayer[12], LongShortTermMemoryLayer[50]},
			"dec" -> LongShortTermMemoryLayer[50],
			"attend" -> AttentionLayer[],
			"cat" -> CatenateLayer[2],
			"classify" -> {NetMapOperator[LinearLayer[]], SoftmaxLayer[]}
		|>, 
		{"enc" -> "dec" -> NetPort["attend", "Query"],
		 "enc" -> NetPort["attend","Key"], "enc" -> NetPort["attend","Value"],
		 "dec" -> "cat","attend" -> "cat","cat" -> "classify"},
		"Input" -> {"Varying", NetEncoder[{"Class", digits}]},
		"Output" -> {"Varying", NetDecoder[{"Class", digits}]}
	]
];

$BuiltinNets["QuestionAnsweringTutorialNet"] := Scope[
	dict = {"back", "bathroom", "bedroom", "Daniel", "garden", "hallway", "is", 
		"John", "journeyed", "kitchen", "Mary", "moved", "office", "Sandra", 
		"the", "to", "travelled", "went", "Where"};
	classes = {"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"};
	enc = NetEncoder[{"Tokens", Join[dict, {"."}]}];
	net = NetGraph[<|
		"context" -> {EmbeddingLayer[50], DropoutLayer[0.3]}, 
		"question" -> {EmbeddingLayer[50], DropoutLayer[0.3], 
		LongShortTermMemoryLayer[50], SequenceLastLayer[]}, 
		"cat" -> CatenateLayer[], 
		"classifier" -> {LongShortTermMemoryLayer[50], SequenceLastLayer[],DropoutLayer[0.3], LinearLayer[Length[classes]], SoftmaxLayer[]}
	|>, {
		NetPort["Context"] -> "context", 
		NetPort["Question"] -> "question",
		{"question", "context"} -> "cat", "cat" -> "classifier" -> NetPort["Answer"]},
	"Context" -> enc, "Question" -> enc, "Answer" -> NetDecoder[{"Class", classes}]];
	net
];

$BuiltinNets["LanguageModelTutorialNet"] := Scope[
	characters = Characters["!()_-.,;\"?':
 0125678aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ"];
	n = Length[characters] + 1;
	predict = NetChain[
		{UnitVectorLayer[n], GatedRecurrentLayer[128], GatedRecurrentLayer[128], NetMapOperator[LinearLayer[n]], SoftmaxLayer[]}
	];
	teacherForcingNet = NetGraph[
		<|"predict" -> predict, "rest" -> SequenceRestLayer[], "most" -> SequenceMostLayer[], "loss" -> CrossEntropyLossLayer["Index"]|>,
  		{NetPort["Input"] -> "most" -> "predict" -> NetPort["loss", "Input"], NetPort["Input"] -> "rest" -> NetPort["loss", "Target"]},
		"Input" -> NetEncoder[{"Characters", {characters, EndOfString}, "TargetLength" -> 26}]
	];
	teacherForcingNet
];

CachedResourceData::nodatares = NetTrain::nodatares = "Could not obtain a resource with the name \"``\".";

$namedDataCache = <||>;


PackageScope["CachedResourceData"]

CachedResourceData[name_, trainq_:True] := 
	CatchFailure @ resolveNamedData[name, trainq];

resolveNamedData[name_, trainq_] := Scope[
	key = Hash[{name, trainq}];
	res = Lookup[$namedDataCache, key, $Failed];
	If[FailureQ[res],
		res = getNamedData[name, trainq];
		AppendTo[$namedDataCache, key -> res];
		If[Length[$namedDataCache] > 6, $namedDataCache ^= Rest[$namedDataCache]];
	];
	res
];

$customDataFunctions = <|
	"SentimentAnalysisTutorialData" -> makeSentimentAnalysisData,
	"IntegerAdditionTutorialData" -> makeIntegerAdditionData,
	"IntegerSortingTutorialData" -> makeIntegerSortingData,
	"QuestionAnsweringTutorialData" -> makeQuestionAnsweringData,
	"LanguageModelTutorialData" -> makeLanguageModelData
|>;

getNamedData[name:(Alternatives @@ Keys[$customDataFunctions]), trainq_] := Scope[
	elem = If[trainq, "Train", "Test"];
	testPath = FileNameJoin[{$cachedDataDir, name <> "_test.mx"}];
	trainPath = FileNameJoin[{$cachedDataDir, name <> "_train.mx"}];
	path = If[trainq, trainPath, testPath];
	If[FileExistsQ[path], Return @ Import @ path];	
	{testData, trainData} = BlockRandom[SeedRandom[1]; $customDataFunctions[name][]];
	testData = TryPack[testData];
	trainData = TryPack[trainData];
	Export[testPath, testData, "MX"];
	Export[trainPath, trainData, "MX"];
	If[trainq, trainData, testData]
];

makeSentimentAnalysisData[] := Scope[
	trainData = getNamedData["MovieReview", True];
	testData = getNamedData["MovieReview", False];
	{testData, trainData}
];

makeIARule[a_, b_] := IntegerString[a] <> "+" <> IntegerString[b] -> IntegerString[a + b];
makeIntegerAdditionData[] := Scope[
	data = RandomSample[Flatten @ Table[makeIARule[i, j], {i, 0, 999}, {j, 0, 999}], 60000];
	{testData, trainData} = TakeDrop[data, 6000];
	{testData, trainData}
];

makeIntegerSortingData[] := Scope[
	digits = Range[6];
	seqs = RandomSample[Flatten[Table[Tuples[digits, n], {n, 3, 6}], 1]];
	data = Map[# -> Sort[#] &, seqs];
	{testData, trainData} = TakeDrop[data, Ceiling[Length[data]/10]];
	{testData, trainData}
];

makeQuestionAnsweringData[] := Scope[
	trainData = ResourceData["The 20-Task bAbI Question-Answering Dataset v1.2", "TrainingData"]["Task1"];
	testData = ResourceData["The 20-Task bAbI Question-Answering Dataset v1.2", "TestData"]["Task1"];
	{testData, trainData}
];


sampleCorpus[corpus_, len_, num_] := Module[
	{positions = RandomInteger[{len + 1, StringLength[corpus] - 1}, num]},
	<|"Input" -> StringTake[corpus, {# - len, #} & /@ positions], "Target" -> StringPart[corpus, positions + 1]|>
];
makeLanguageModelData[] := Scope[
	corpus = ResourceData["The American"] <> " " <> ResourceData["Don Quixote - English"];
	trainData = sampleCorpus[corpus, 25, 300000];
	testData = sampleCorpus[corpus, 25, 10000];
	{testData, trainData}
];


$MLEDNames = {"BostonHomes", "FisherIris", "MovieReview", "Mushroom", "Satellite", "Titanic", "UCILetter", "WineQuality"};

getNamedData[name_, trainq_] := Scope[
	path = FileNameJoin[{$cachedDataDir, name <> If[trainq, "_train.mx", "_test.mx"]}];
	If[FileExistsQ[path], Return @ Import @ path];
	If[StringEndsQ[name, "Sample"], 
		name = StringTrim[name, "Sample"];
		isSample = True;
	];
	elem = If[trainq, "TrainingData", "TestData"];
	res = Quiet @ ResourceData[name, elem];
	If[Head[res] === ResourceData || res === Null, res = $Failed];
	If[FailureQ[res] && MemberQ[$MLEDNames, name], 
		res = ExampleData[{"MachineLearning", name}, elem]];
	If[FailureQ[res],
		ThrowFailure["nodatares", name]];
	res = TryPack[res];
	If[isSample === True, res = Map[subSample, res]];
	Export[path, res, "MX"];
	res
];

subSample[e_List] := BlockRandom[SeedRandom[1]; RandomSample[e, 1000]];
subSample[e_] := Panic["CannotSubsample"];
