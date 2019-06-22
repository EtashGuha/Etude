Package["NeuralNetworks`"]


$LayerUpgradeData = <||>;

(* This file contains the legacy upgrade rules, which are the way we did things
prior to 11.2. Now, rules are populated directly from layer definitions. *)

(******************************************************************************)

$LayerUpgradeData["11.1.0 to 11.1.1"] = {
	"Linear" -> upgradePrereleaseLinear
}

upgradePrereleaseLinear[assoc_] :=
	upgradeLinear[assoc, TDimensions @ assoc["Inputs", "Input"], assoc["Parameters", "Size"]];

(******************************************************************************)

$LayerUpgradeData["11.1.1 to 11.2.0"] = {
	"Resize" -> ApplyParams[upgradeResize],
	"Aggregation" -> ApplyParams[upgradeAggregation],
	"Dropout" -> ApplyParams[upgradeDropout],
	"Replicate" -> ApplyParams[upgradeReplicate],
	"BatchNormalization" -> reorderBatchNormArraysAndRemoveEither
};

upgradeResize[params_] := Append[params, "Resampling" -> "Linear"];

upgradeAggregation[params_] := Join[
	KeyDrop[params, "$Channels"], 
	<|
		"Levels" -> ValidatedParameter[2;;],
		"$InputDimensions" -> Join[{params["$Channels"]}, params["$InputDimensions"]],
		"$OutputDimensions" -> {params["$Channels"]}
	|>
];

upgradeDropout[params_] := Append[params, "Method" -> "Dropout"];

upgradeReplicate[params_] := <|
	"Specification" -> ValidatedParameter@params["Specification"],
	"Level" -> 1,
	"OutputSize" -> params["OutputSize"],
	"$InsertedDimCount" -> Length @ ToList @ params["Specification"],
	"$InsertedDimensions" -> ToList @ params["Specification"],
	"$InputSize" -> params["$InputSize"]
|>;

reorderBatchNormArraysAndRemoveEither[assoc_] :=
	MapAt[KeySortBy[Position[{"Gamma", "Beta", "MovingMean", "MovingVariance"}, #]&], assoc, "Arrays"] /. _EitherT :> 
		RuleCondition @ ChannelT[SizeT, RealTensorT];

(******************************************************************************)

$LayerUpgradeData["11.0.0 to 11.1.1"] = {
	(* simple upgraders that just modify params *)
	"Pooling" -> ApplyParams[upgradePooling],
	"Softmax" -> ApplyParams[upgradeSoftmax],
	"Convolution" -> ApplyParams[upgradeConvolution],
	"CrossEntropyLoss" -> ApplyParams[upgradeCrossEntropyLoss],
	"Elementwise" -> ApplyParams[upgradeElementwise],

	"Split" -> splitGone,
	"BroadcastPlus"-> broadcastGone,
	"Upsample" -> renameTo["Resize"] /* upgradeUpsample,
	"ScalarTimes" -> renameTo["Elementwise"] /* ApplyParams[scalarFunctionParams[Times]],
	"ScalarPlus" -> renameTo["Elementwise"] /* ApplyParams[scalarFunctionParams[Plus]],
	"Catenate" -> upgradeCatenate,

	(* more complicated upgraders *)
	"Graph"|"Chain" -> renameContainerKeys,
	"MeanAbsoluteLoss"|"Reshape"|"Softmax"|"Transpose" -> removeRanks,
	"Flatten" -> upgradeFlatten,
	"DotPlus" -> upgradeDotPlus
};

scalarFunctionParams[f_][assoc_] := Association[
	"Function" -> ValidatedParameter[
		CompileScalarFunction[1, f[#, assoc["Scalar"]]&]
	],
	"$Dimensions" -> assoc["$Dimensions"]
];

upgradeConvolution[params_] := 
	Append[params, "Dimensionality" -> 2];

upgradeElementwise[params_] := 
	MapAt[ValidatedParameter, KeyDrop[params, "$Rank"], "Function"];

upgradePooling[params_] := Scope[
	Function[
		newSize = PoolingShape[#$InputSize, #PaddingSize, #KernelSize, #Stride, "valid"];
		If[VectorQ[newSize, IntegerQ] && newSize =!= #$OutputSize,
			FailValidation[PoolingLayer, 
				"PoolingLayer with given stride and input size cannot be imported into Mathematica " <> 
					ToString[$VersionNumber] <> "."
			]
		]
	] @ params;
	Append[params, {"Dimensionality" -> 2, "$MXPoolingConvention" -> "valid", "$MXGlobalPool" -> False}]
];

(* TODO: Convert other upgrades to use ApplyParams *)

upgradeSoftmax[params_] := Association[];

upgradeCrossEntropyLoss[params_] := Association[
	"TargetForm" -> params["TargetForm"],
	"$InputDimensions" -> {},
	"$Classes" -> params["$Dimensions"]
]


upgradeFlatten[assoc_] := 
	ReplacePart[assoc, 
		"Parameters" -> Association[
			"Level" -> Infinity,
			"$InputSize" -> OldTDimensions[assoc["Inputs", "Input"]],
			"OutputSize" -> OldTDimensions[assoc["Outputs", "Output"]]
		]
	];

upgradeCatenate[assoc_] := 
	ReplacePart[assoc,
		"Parameters" -> Association[
			"Level" -> 1,
			"$InputShapes" -> assoc["Inputs", "Input"],
			"$InputCount" -> assoc["Parameters", "$InputCount"],
			"$OutputShape" -> assoc["Outputs", "Output"]
		]
	];

upgradeDotPlus[assoc_] := Scope[
	idims = OldTDimensions @ assoc["Inputs", "Input"];
	osize = assoc["Parameters", "Size"];
	upgradeLinear[assoc, idims, osize]
];

upgradeLinear[assoc_, idims_, osize_] := Scope[
	assoc = assoc;
	assoc["Type"] = "Linear";
	assoc["Parameters"] = Association[
		"OutputDimensions" -> {osize},
		"$OutputSize" -> osize,
		"$InputSize" -> First[idims, SizeT],
		"$InputDimensions" -> idims
	];
	assoc
];

OldTDimensions[EncodedType[_, t_] | DecodedType[_, t_]] := OldTDimensions[t];
OldTDimensions[ChannelT[n_, inner_]] := Replace[OldTDimensions[inner], l_List :> Prepend[l, n]];
OldTDimensions[TensorT[n_Integer, dims_List]] := dims;
OldTDimensions[TensorT[size_, _]] := If[IntegerQ[size], Table[SizeT, size], SizeListT[]];
OldTDimensions[_] := SizeListT[];

renameContainerKeys[assoc_] := 
	MapAt[ReplaceAll[LayerUpgradeRules["11.0.0 to 11.1.1"]], "Nodes"] @ 
	MapAt[ReplaceAll["Vertices"|"Layers" -> "Nodes"], "Edges"] @ 
	KeyMap[Replace[{"Vertices"|"Layers" -> "Nodes", "Connections" -> "Edges"}]] @ 
	assoc;

removeRanks[layer_] := MapAt[
	KeyDrop[{"$Rank", "Rank", "$InputRank", "$OutputRank"}], 
	layer, "Parameters"
];

renameTo[name_] := ReplacePart["Type" -> name];

broadcastGone[_] := goneFail["BroadcastPlusLayer", "ReplicateLayer followed by ElementwiseLayer[Plus]"];
splitGone[_] := goneFail["SplitLayer", "several PartLayers"]

goneFail[name_, replaceHint_] := upgradeFail @ StringJoin[
	"the experimental layer ", name, " no longer exists, and has no direct analogue in this version. ",
	"The same functionality can be achieved with ", replaceHint
];

upgradeUpsample[assoc_] := NData @ ResizeLayer[
	{1,1} * Scaled[assoc["Parameters", "Scale"]], 
	"Input" -> ReplaceRepeated[assoc["Inputs", "Input"], $TensorUpgradeRule]
];

General::netupgfail = "Net could not be upgraded: ``.";
upgradeFail[msg_] := ThrowFailure["netupgfail", msg];

