Package["NeuralNetworks`"]

PackageScope["SetupDeprecationLogic"]

General::depropt = "Option `` is deprecated, use `` instead.";
General::deprlayer = "Layer `1` is deprecated, please use `2` instead. An equivalent `2` will be constructed for this evaluation.";

SetupDeprecationLogic[] := (

	KeyAppendTo[$LayerUpgradeData, "11.3.6", "SequenceAttention" -> upgradeSequenceAttention];
	SequenceAttentionLayer[args___] := warnAndBuild[
		SequenceAttentionLayer, AttentionLayer, "$InputPorts" -> "InputQuery", args];

	KeyAppendTo[$LayerUpgradeData, "11.3.6", "InstanceNormalization" -> upgradeInstanceNormalization];
	InstanceNormalizationLayer[args___] := warnAndBuild[
		InstanceNormalizationLayer, NormalizationLayer, args];

	DotPlusLayer[args___] := warnAndBuild[
		DotPlusLayer, LinearLayer, args, "Input" -> {Automatic}, "Output" -> {Automatic}];

);

warnAndBuild[old_, new_, args___] := (
	Message[old::deprlayer, old, new];
	Quiet[new[args], {General::depropt}]
);

upgradeInstanceNormalization[assoc_] := Scope[
	assoc["Type"] = "Normalization";
	assoc["Parameters"] = 
	KeyDrop[
		Join[
			assoc["Parameters"], 
			With[{
				channels = assoc["Parameters", "$Channels"],
				inputDimensions = assoc["Parameters", "$InputDimensions"]
			}, Association[
				"AggregationLevels" -> ValidatedParameter[2;;All],
				"ScalingLevels" -> ValidatedParameter["Complement"],
				"$Dimensions" -> Prepend[inputDimensions, channels],
				"$StatsDimensions" -> {channels}
				]
			]
		],
		{"$Channels", "$InputDimensions"}
	];
	assoc["Arrays"] = Association[
		"Scaling" -> assoc["Arrays", "Gamma"],
		"Biases" -> assoc["Arrays", "Beta"]
	];
	assoc
];

upgradeSequenceAttention[assoc_] := Scope[
	assoc["Type"] = "Attention";
	assoc["Parameters"] = 
	KeyDrop[
		Join[
			assoc["Parameters"], 
			With[{
				inputLength = assoc["Parameters", "$InputLength"],
				queryLength = assoc["Parameters", "$QueryLength"],
				inputSize = assoc["Parameters", "$InputSize"],
				querySize = assoc["Parameters", "$QuerySize"]
			}, Association[
				"$InputPorts" -> "InputQuery",
				"ScoreRescaling" -> None,
				"Mask" -> None,
				"$InputShape" -> {inputLength},
				"$QueryShape" -> {queryLength},
				"$KeyChannels" -> {inputSize},
				"$ValueChannels" -> {inputSize},
				"$QueryChannels" -> {querySize},
				"$ScoreChannels" -> {}
				]
			]
		],
		{"$InputLength", "$QueryLength", "$InputSize", "$QuerySize"}
	];
	assoc
];
