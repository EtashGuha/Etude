Package["NeuralNetworks`"]


GuessDataLength[data_] := Switch[data,
	{} | <||>, 		ThrowFailure["nodata"],
	"RandomData",	$NetTrainRandomDataSize,
	{"RandomData", "RoundLength" -> _Integer ? Positive}, data[[2, 2]],
	{_, "RoundLength" -> _}, Infinity,
	_List, 			Length[data],
	_Association, 	Length[First[data]],
	_List -> _List, Length[First[data]],
	HoldPattern[_Dataset], GuessDataLength[Normal[data]],
	_, 				Infinity
];


PackageScope["ParseTrainingData"]

(* NOTE: ParseTrainingData expects the input to have been preprocessed with TryPack, which turns
e.g. lists of rules into a rule of lists. *)

ParseTrainingData[data_, inputs_, batchSize_, factory_] := Timed @ Scope[
	Which[
		data === {} || data === <||>,
			NetTrain::nodata = "No training data provided.";
			ThrowFailure["nodata"]
		,
		AssociationQ[data],
			If[!ListOfListsQ[Values[data]],
				NetTrain::invtdata = "Training data should be an association of lists, or a rule from input to output examples.";
				ThrowFailure["invtdata"]
			]
		,
		MatchQ[data, _List | _NumericArray -> _List | _NumericArray],
			{ikey, okey} = getRuleKeys[inputs];
			data = <|ikey -> First[data], okey -> Last[data]|>;
		,
		MatchQ[Head[data], HoldPattern @ Dataset],
			If[!MatchQ[Dataset`GetType[data], TypeSystem`Vector[_TypeSystem`Struct, _]],
				NetTrain::invdataset = "Datasets provided to NetTrain must consist of a list of associations with fixed keys.";
				ThrowFailure["invdataset"]
			];
			data = AssociationTranspose @ Normal[data]
		,
		AssociationVectorQ[data],
			data = AssociationTranspose @ data
		,
		data === "RandomData" && (data = {"RandomData", "RoundLength" -> $NetTrainRandomDataSize}; False),
			Null,
		ListQ[data] && Length[data] === 2 && MatchQ[data, {"RandomData", "RoundLength" -> _Integer ? Positive}],
			Return @ ParseTrainingData[
				CreateRandomData[inputs, data[[2, 2]]], 
				inputs, batchSize, factory
			]
		,
		StringQ[data],
			data = resolveNamedData[data, factory === ConstructTrainingGenerator || factory === inferNetTypes];
			Return @ ParseTrainingData[data, inputs, batchSize, factory];
		,
		factory === inferNetTypes,
			NetTrain::jitgen = "Generator function cannot be used when the net to be trained has not been fully specified.";
			Message[NetTrain::jitgen];
			ReturnFailed[];
		,
		MatchQ[data, File[path_String /; FileType[path] === File]],
			data = OpenHDF5TrainingData[First @ data];
			AppendTo[$CleanupQueue, Hold[CloseHDF5TrainingData][data]];
			length = data[[2]];
		,
		MatchQ[data, {_ ? System`Private`MightEvaluateWhenAppliedQ, "RoundLength" -> (n_Integer ? Positive)}] &&
			(genRoundLength= data[[2,2]]; data = First[data]; False), 
			$Unreachable (* bit of a hack to reuse code below *)
		,
		MatchQ[data, _List], (* because MightEvaluateWhenAppliedQ also looks inside the list *)
			ThrowFailure["invtdata"],
		Compose[System`Private`MightEvaluateWhenAppliedQ, data], (* custom generator *)
			(* the gen input has to compensate for the fact that we prefetch one round ahead *)
			setupCurrentNetData[];
			If[factory === ConstructTrainingGenerator, $trainingGeneratorIsCustom ^= True];
			$generatorInput ^= ReplacePart[$livePropertyData, {
				"Round" :> Boole[$batch === $batchesPerRound] + $round, 
				"AbsoluteBatch" :> $absoluteBatch + 1,
				"Batch" :> If[IntegerQ[$batchesPerRound], 1 + Mod[$absoluteBatch, $batchesPerRound], 1]
			}];
			wrapper = checkAndWrapGeneratorOutput[
				data,
				data[$generatorInput], 
				inputs, batchSize
			];
			length = If[FastValueQ[genRoundLength], Ceiling[genRoundLength, batchSize], batchSize];
			If[ContainsVarSequenceQ[inputs],
				varKeys = Keys @ Select[inputs, VarSequenceQ];
				varIDs = First[UniqueLengthVars[#]]& /@ Lookup[inputs, varKeys];
				bucketf = MakeBucketWrapperFunction[varKeys, varIDs];
			,
				bucketf = Identity;
			];
			data = CustomGenerator[data /* wrapper, $generatorInput, bucketf];
		,
		True,
			ThrowFailure["invtdata"]
	];

	(* this kicks in for HDF5 files and user-provided generators *)
	If[!AssociationQ[data], Goto[SkipChecks]];

	lengths = Map[Length, data];
	If[!Apply[SameQ, lengths], 
		NetTrain::invinlen = "Inconsistent numbers of examples provided to ports: lengths were ``.";
		ThrowFailure["invinlen", lengths]
	];

	length = First[lengths];

	NetTrain::missinslot = "Training data specification should include values for ``, but only contains values for ``. You may need to explicitly specify the loss port(s) of the net using LossFunction.";
	inKeys = Keys[inputs];
	dataKeys = Keys[data];
	If[!SubsetQ[dataKeys, inKeys], ThrowFailure["missinslot", portAndList @ inKeys, portAndList @ dataKeys]];
	If[Length[dataKeys] > Length[inKeys], data = KeyTake[data, inKeys]];

	Map[ (* <- 322488 *)
		If[!PackedArrayQ[#] && System`Private`CouldContainQ[#, Missing] && MemberQ[#, _Missing],
			NetTrain::contmiss = "NetTrain does not currently support data that contains missing values.";
			ThrowFailure["contmiss"]
		]&,
		data
	];

	Label[SkipChecks];

	If[batchSize > length && factory =!= ConstructValidationGenerator, batchSize = length];
	data = factory[data, inputs, batchSize]; 

	{data, length}
];

portAndList[{}] := "(no ports)";
portAndList[{a_}] := StringForm["the port ``", a];
portAndList[l_List] := StringForm["the ports ``", QuotedStringRow[l, " and "]];

(* we test the very first batch for basic spec, to make sure it produces
a list of rules or an association with the right ports, and that the
length is correct. This is mainly done here for specificity of the
error message, and also to prevent you using a generator with e.g.
ValidationSet and have that fail when you've already been training a while.

Subsequently, ConstructXXXGenerator will create a compiled predicate
that checks the association's actual values. This will give a generic
error message when things aren't correct.
*)

NetTrain::invgenout = "Output of training data generator function was incorrect: ``."

genfpanic[args__] := genfpanic[StringFormToString @ StringForm[args]];
genfpanic[arg_] := (Message[NetTrain::invgenout, arg]; ThrowRawFailure[$Failed]);

$invgensize = "generator did not return data with length equal to the requested BatchSize of ``";

checkAndWrapGeneratorOutput[genf_, data_, inputs_, batchSize_] := Scope[

	If[AssociationVectorQ[data],
		data = TransposeAssocVector[data];
		If[FailureQ[data], genfpanic["generator returned associations with inconsistent keys"]];
		Return @ Function[
			If[!AssociationVectorQ[#], genfpanic["generator did not return a list of associations"]];
			Replace[TransposeAssocVector[#], $Failed :> genfpanic["generator returned associations with inconsistent keys"]]
		];
	];

	keys = Keys[inputs];
	If[AssociationQ[data], 
		TestSubsetQ[
			Keys[data], keys,
			genfpanic["generator result did not specify a value for the port \"``\"", #]&
		];
		lens = Length /@ Lookup[data, keys];
		If[!AllSameAs[lens, batchSize],
			genfpanic[$invgensize, batchSize]
		];
		Return @ Identity
	];

	If[RuleVectorQ[data],
		{ikey, okey} = getRuleKeys[inputs];
		If[Length[data] =!= batchSize,
			genfpanic[$invgensize, batchSize]
		];
		Return @ With[{keys = {ikey, okey}},
			Function[input,
				If[!MatchQ[input, {__Rule}],
					genfpanic["generator did not return a list of rules"],
					AssociationThread[keys, KeysValues[input]]
				]
			]
		]
	];

	(* TODO: do detailed checks here on the actual data, so we can
	we can fail upfront if the payload is wrong, with a good error
	message *)
	genfpanic["generator did not return an association or list of rules"];
];

getRuleKeys[inputs_] := Scope[
	If[Length[inputs] != 2, 
		NetTrain::invsimplein = "Given training data specification can only be used when net has one input and one output, or two inputs and an explicit loss function. You may need to explicitly specify the loss port(s) of the net using LossFunction.";
		ThrowFailure["invsimplein"]
	];
	{ikey, okey} = Keys[inputs];
	Which[
		ikey === "Input", Null,
		okey === "Input", {ikey, okey} = {okey, ikey},
		True, ThrowFailure["invsimplein"]
	];
	{ikey, okey}
];
