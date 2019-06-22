(* ::Section:: *)
(*AudioIdentify*)


PackageExport["AudioIdentify"]


Options[AudioIdentify] =
SortBy[
	{
		Masking -> Automatic,
		PerformanceGoal :> $PerformanceGoal,
		SpecificityGoal -> Automatic,
		AcceptanceThreshold -> Automatic,
		TargetDevice -> "CPU"
	},
	ToString@*First
];


$AudioIdentifyHiddenOptions = {"Caching" -> True};


DefineFunction[
	AudioIdentify,
	iAudioIdentify,
	{1, 4},
	"ExtraOptions" -> $AudioIdentifyHiddenOptions
]


iAudioIdentify[args_, opts_]:=
Module[
	{
		a, specificity, confidence, threshold, masking, categories, n, properties, speedQ, targetDevice,
		res
	},

	(*-- check for the ontology paclet to be loaded --*)
	If[!AssociationQ[AudioIdentify`Private`Ontology],
		DBPrint[ToString[$FunctionName] <> ": could not load the AudioIdentify ontology data."];
		$AllowFailure = True;
		ThrowFailure[AudioIdentify::"interr", "the ontology data could not be loaded"]
	];

	(*-- argument parsing --*)

	(* support for batched eval of list of audios *)
	If[ListQ[args[[1]]],
		a = Catch @ Audio`Utilities`checkAudio[#, AudioIdentify]& /@ args[[1]];
		,
		a = Catch @ Audio`Utilities`checkAudio[args[[1]], AudioIdentify];
	];
	If[MemberQ[a, $Failed, {0, 1}],
		ThrowFailure[];
	];
	(* Support for input AudioStream *)


	If[Length[args] > 1,
		Switch[args[[2]],
			All,
				categories = {};,

			_String | Entity["Sound", _] | {(_String | Entity["Sound", _]) ..},
				categories = ToList[args[[2]]];,

			Verbatim[Alternatives][(_String | Entity["Sound", _])..],
				categories = List@@args[[2]];,

			_,
				ThrowFailure["catinv", args[[2]]];
		]
		,
		categories = {};
	];

	categories = Replace[categories, Entity["Sound", x_] :> Lookup[$entityMapping, x, x], {1}];
	lowerCaseCategories = Replace[categories, x_String :> ToLowerCase[x], {1}];

	If[Length[categories] > 0,
		If[AnyTrue[Lookup[$totalNumberOfChildrenLowerCase, lowerCaseCategories], MissingQ[#] || # < 2 &],
			SelectFirst[
				categories,
				!ValidPropertyQ[
					AudioIdentify,
					#,
					Keys[AudioIdentify`$AudioIdentifyCategories],
					"Type" -> "category",
					MaxItems -> (1 + Length[AudioIdentify`$AudioIdentifyCategories])
				]&
			];
			ThrowFailure[];
			,
			categories = First@Nearest[Keys[$totalNumberOfChildren], #] & /@ categories;
			categories = Union @@ Lookup[AudioIdentify`Private`Ontology["Data"], categories][[All, "FlattenedChildren"]];
		]
	];

	If[Length[args] > 2,
		If[MatchQ[args[[3]], _?PositiveIntegerQ | All | Automatic],
			n = args[[3]];
			,
			ThrowFailure["numinv", args[[3]]];
		]
		,
		n = Automatic;
	];

	properties = Automatic;
	If[Length[args] > 3,
		properties = args[[4]];
		If[
			And[
				properties =!= Automatic,
				AnyTrue[Developer`ToList[properties], !ValidPropertyQ[AudioIdentify, #, $validProperties]&]
			],
			ThrowFailure[];
		];
	];


	(*-- option parsing --*)

	(*heuristic for node selection*)
	specificity = Replace[GetOption[SpecificityGoal], Automatic -> .5];
	Switch[specificity,
		"Low",
			specificity = .8
		,
		"High",
			specificity = .2
		,
		_?(RealValuedNumericQ[#] && Quiet@TrueQ[0 <= # <= 1] &),
			(* This is to accomodate the computing of the weighted score *)
			specificity = Rescale[1. - specificity, {-.2, 1.1}];
		,
		_,
			ThrowFailure["bdtrg", specificity]
	];

	(*manual threshold specification*)
	threshold = GetOption[AcceptanceThreshold];
	If[threshold =!= Automatic,
		If[!(RealValuedNumericQ[threshold] && Quiet@TrueQ[0 <= threshold <=1]),
			ThrowFailure["thrs", threshold]
			,
			threshold = threshold + 10^-5
		]
		,
		If[(RealValuedNumericQ[#] && Quiet@TrueQ[0 <= # <= 1]) & @defaultParameters["AudioIdentify","AcceptanceThreshold"],
			threshold = defaultParameters["AudioIdentify","AcceptanceThreshold"]
			,
			threshold = 10^-3
		]
	];

	$Caching = TrueQ[GetOption["Caching"]];

	masking = Replace[GetOption[Masking], Automatic -> All];
	If[masking =!= All,
		a = Replace[a, x_Audio :> Quiet@AudioJoin[ToList@AudioTrim[x, masking]], {0, 1}];
		If[!(AudioQ[a] || VectorQ[a, AudioQ]),
			ThrowFailure["msk", masking];
		];
	];

	speedQ = TrueQ[GetOption[PerformanceGoal] === "Speed"];

	targetDevice = GetOption["TargetDevice"];
	If[!NeuralNetworks`TestTargetDevice[targetDevice, AudioIdentify], ThrowFailure[];];

	(* Cached *)
	res = getNetworkResult[speedQ, a, targetDevice];

	(* Cached *)
	res = oAudioIdentify[res, specificity, confidence, threshold, categories, n, properties];

	res
]


oAudioIdentify[a_List, specificity_, confidence_, threshold_, categories_, n_, properties_] :=
Map[oAudioIdentify[#, specificity, confidence, threshold, categories, n, properties]&, a]


oAudioIdentify[a_Association, specificity_, confidence_, threshold_, categories_, n_, properties_] :=
Cached@Module[
	{
		res = a
	},

	(* Cached *)
	res = scoreResult[res, specificity, confidence, categories];

	(* Cached *)
	res = selectResult[res, threshold, n];

	(* Cached *)
	res = formatResult[res, properties, n];

	res
]


(* ::Subsubsection::Closed:: *)
(*$validProperties*)


$entityProperties = {"AlternateNames", "Description", "Examples", "Name",
	"NarrowerSounds", "Restrictions", "SourceEntity", "BroaderSounds"
};


$validProperties = Join[{"Sound", "Probability"}, $entityProperties];


(* ::Subsubsection::Closed:: *)
(*getNetworkResult*)


getNetworkResult[speedQ_, input_, targetDevice_] :=
Cached@With[
	{
		net = If[speedQ,
			GetNetModel["HiddenAudioIdentifyMobileNetDepth1-0"]
			,
			GetNetModel["HiddenAudioIdentifyMobileNetDepth1-3"]
		]
	},
	SafeNetEvaluate[net[input, "Probabilities", TargetDevice -> targetDevice]]
];


(* ::Subsubsection::Closed:: *)
(*scoreResult*)


scoreResult[input_, specificity_, confidence_, categories_] :=
Cached@Module[
	{res = input},
	If[categories =!= {},
		res = KeySelect[res, MatchQ[#, Alternatives@@categories]&];
	];
	res = Association[
		KeyValueMap[
			#1 -> With[
				{max = Max[Lookup[res, Flatten[{#1, #2[["FlattenedChildren"]]}], 0.]]},
				<|
					"DepthIndex" -> #2[["DepthIndex"]],
					"OriginalNetScore" -> Lookup[res, #1, max],
					"MaxScoreInBranch" -> max
				|>
			] &,
			If[categories =!= {},
				KeySelect[AudioIdentify`Private`Ontology["Data"], MatchQ[#, Alternatives@@categories]&]
				,
				AudioIdentify`Private`Ontology["Data"]
			]
		]
	];
	Map[
		Append[#,
			"WeightedScore" ->
				(*
					This has been decided experimentally and probably should be changed if the internal net is changed.
					The idea is that the net result, augmented with the max result in that branch, is weighted through the
					specificity with the dpeth of a category in its branch.
				*)
				specificity*((.5 +.2specificity)#["OriginalNetScore"] + (.5 -.2specificity)#["MaxScoreInBranch"]) + (1 - specificity)*#["DepthIndex"]
		]&,
		res
	]
]


(* ::Subsubsection::Closed:: *)
(*selectResult*)


selectResult[in_, threshold_, n_] :=
Cached@Module[
	{res},
	res = SortBy[Select[in, #WeightedScore > threshold&], -#WeightedScore &];
	If[Length[res] === 0,
		{Missing["Unidentified"]}
		,
		Switch[n,
			All,
				res,

			Automatic,
				res[[{1}]],

			_?PositiveIntegerQ,
				Take[res, UpTo[n]]
		]
	]
]


(* ::Subsubsection::Closed:: *)
(*formatResult*)


computeProp[p_, key_, input_] /; MemberQ[$entityProperties, p] := key[p]
computeProp["Sound", key_, input_] := key
computeProp["Probability", key_, input_] := input["WeightedScore"]
computeProp[p_List, key_, input_] := Map[computeProp[#, key, input] &, p]


formatResult[in_, properties_, n_] :=
Cached@Module[
	{res},
	If[in === {Missing["Unidentified"]},
		Return[First[in]]
	];
	res = KeyMap[
		Entity["Sound", AudioIdentify`Private`Ontology["Data", #, "EntityCanonicalName"]]&,
		in
	];
	If[MatchQ[properties, Automatic|"Sound"],
		If[n === Automatic,
			First[Keys[res]]
			,
			Keys[res]
		]
		,
		Association @ KeyValueMap[#1 -> computeProp[properties, #1, #2] &, res]
	]
]


(* ::Section::Closed:: *)
(*AudioSet AudioIdentify`Private`Ontology data*)


PackageScope["AudioIdentify`Private`Ontology"]
PackageScope["AudioIdentify`$AudioIdentifyCategories"]

AudioIdentify`Private`Ontology :=
AudioIdentify`Private`Ontology = If[
		FileExistsQ[FileNameJoin[{PacletManager`PacletResource["NeuralFunctions", "AudioClassification"], "AudioSetOntologyData.m"}]],
		Import[FileNameJoin[{PacletManager`PacletResource["NeuralFunctions", "AudioClassification"], "AudioSetOntologyData.m"}]]
	];

$entityMapping :=
$entityMapping = Association[Reverse /@ Normal[AudioIdentify`Private`Ontology["Data"][[All, "EntityCanonicalName"]]]];

$totalNumberOfChildren :=
$totalNumberOfChildren = Length /@ AudioIdentify`Private`Ontology[["Data", All, "FlattenedChildren"]];

$totalNumberOfChildrenLowerCase:=
$totalNumberOfChildrenLowerCase = KeyMap[ToLowerCase, $totalNumberOfChildren];

$reverseEntityMapping :=
$reverseEntityMapping = Association[Reverse /@ Normal[$entityMapping]];

AudioIdentify`$AudioIdentifyCategories :=
AudioIdentify`$AudioIdentifyCategories =
	AssociationMap[
		Entity["Sound", $reverseEntityMapping[#]] &,
		Keys@Select[$totalNumberOfChildren, # > 2 &]
	];

defaultParameters["AudioIdentify", "SpecificityGoal"] = 1.;
defaultParameters["AudioIdentify", "AcceptanceThreshold"]= .5;
