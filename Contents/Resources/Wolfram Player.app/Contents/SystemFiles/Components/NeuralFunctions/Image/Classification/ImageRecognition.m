
PackageExport["ImageIdentify"]
PackageExport["ImageInstanceQ"]


(****************************************************************************
    ImageIdentify
****************************************************************************)

Options[ImageIdentify] = {
	SpecificityGoal -> Automatic,
	AcceptanceThreshold -> Automatic,
	TargetDevice -> "CPU"
} // SortBy[ToString @* First];

ImageIdentifyHiddenOptions = {
	"Caching" -> True,
	PerformanceGoal -> Automatic,
	RecognitionThreshold -> Automatic
} // SortBy[ToString @* First];

DefineFunction[
	ImageIdentify,
	iImageIdentify,
	{1,4},
	"ExtraOptions" -> ImageIdentifyHiddenOptions
]

(*Implementation*)

iImageIdentify[args_, opts_] :=
Scope[

	(*-- check for the CNN_GraphData paclet to be loaded --*)
	ontologyData = LoadPaclet["CNN_GraphData4", "Construct.m"];
	If[FailureQ[ontologyData],
		DBPrint[ToString[$FunctionName] <> ": could not load the ontology data."];
		$AllowFailure ^= True;
		ThrowFailure["interr", $FunctionName]
	];

	(*-- argument parsing --*)
	
	Which[
		Image`Utilities`SetImage2D[img, args[[1]]],
		img = RemoveAlphaChannel@args[[1]];
		If[ImageColorSpace[img] =!= "RGB",
			img = ColorConvert[img, "RGB"]
		]
		,
		Image`Utilities`SetImageList2D[img, args[[1]]],
		img = If[ImageColorSpace[#] =!= "RGB",
			ColorConvert[RemoveAlphaChannel@#, "RGB"], RemoveAlphaChannel@#]&/@ img
		,
		True,
		ThrowFailure["imginv", args[[1]]]
	];
	
	If[Length[args] > 1,
		Switch[args[[2]],
			All,
			categories = {},
			
			_String|_Entity|{__String},
			categories = {args[[2]]},
			
			Verbatim[Alternatives][(_String|_Entity|{__String})..] | {(_String|_Entity|{__String})..},
			categories = List@@args[[2]],
			
			_,
			ThrowFailure["catinv", args[[2]]]
		],
		categories = {}
	];
	If[Length[categories] > 0,
		If[AnyTrue[Lookup[objectInterpretation, categories], MissingQ],
			categories = Replace[categories, Entity["Word", w_] :> w, 1];
			If[AnyTrue[Lookup[objectInterpretation, categories], MissingQ],
				ThrowFailure["catinv", args[[2]]],
				categories = Intersection[Flatten[Lookup[objectInterpretation, categories]], VertexList[classifierGraph]]
			],
			categories = Intersection[Flatten[Lookup[objectInterpretation, categories]], VertexList[classifierGraph]]
		]
	];
	
	If[Length[args] > 2,
		If[MatchQ[args[[3]], _?PositiveIntegerQ | All | Automatic],
			n = args[[3]],
			ThrowFailure["numinv", args[[3]]]
		],
		n = Automatic
	];
	
	If[Length[args] > 3,
		If[MatchQ[args[[4]], Alternatives@@validProperties | {(Alternatives@@validProperties)..} | Automatic],
			properties = args[[4]],
			ThrowFailure["propinv", args[[4]]]
		],
		properties = Automatic;
	];
	
	
	(*-- option parsing --*)
	
	(*heuristic for node selection*)
	specificity = GetOption[SpecificityGoal];
	Switch[specificity,
		Automatic,
		If[(NumericQ[#] && TrueQ[0 <= # <= 1]) & @defaultParameters["ImageIdentify","SpecificityGoal"],
			confidence = 1 - defaultParameters["ImageIdentify","SpecificityGoal"] + 10^-5,
			confidence = .1
		],
		"Low",
		confidence = .8,
		"High",
		confidence = .1,
		_?(NumericQ[#] && TrueQ[0 <= # <= 1] &),
		confidence = 1 - specificity + 10^-5,
		_,
		ThrowFailure["bdtrg", specificity]
	];
	
	(*manual threshold specification*)
	If[
		FilterRules[opts, RecognitionThreshold] =!= {},

		Message[ImageIdentify::deprec, RecognitionThreshold, AcceptanceThreshold];
		If[
			FilterRules[opts, AcceptanceThreshold]  === {},
			threshold = GetOption[RecognitionThreshold];
			thresholdMessage = "thrs2",

			threshold = GetOption[AcceptanceThreshold];
			thresholdMessage = "thrs"
		],

		threshold = GetOption[AcceptanceThreshold];
		thresholdMessage = "thrs"
	];

	If[threshold =!= Automatic,
		If[!(NumericQ[threshold] && 0 <= threshold <=1),
			ThrowFailure[Evaluate@thresholdMessage, threshold],
			threshold = Max[threshold, 10^-5]
		],
		If[(NumericQ[#] && TrueQ[0 <= # <= 1]) & @defaultParameters["ImageIdentify","AcceptanceThreshold"],
			threshold = defaultParameters["ImageIdentify","AcceptanceThreshold"],
			threshold = 10^-3
		]
	];

	targetDevice = GetOption[TargetDevice];
	If[!TrueQ[NeuralNetworks`TestTargetDevice[targetDevice, $FunctionName]],
		ThrowFailure[]
	];

	(*classifier performance target*)
	performancegoal = GetOption[PerformanceGoal];
	If[!MatchQ[performancegoal, "Quality"|"Speed"|Automatic],
		ThrowFailure["mlbdpg", performancegoal]
	];
	
	(*classifier performance target*)
	cacheQ = TrueQ@GetOption["Caching"];
	
	(*-- recognition --*)
	If[VectorQ[img],
		Map[
			If[cacheQ,
				recognition = Internal`CheckImageCache[{"ImageRecognition", key = Hash[#], categories, n, properties, threshold, confidence, performancegoal, opts}];
				If[recognition === $Failed,
					recognition = oImageIdentify[#, key, categories, n, properties, threshold, confidence, performancegoal, targetDevice, opts];
					Internal`SetImageCache[{"ImageRecognition", key, categories, n, properties, threshold, confidence, performancegoal, opts}, recognition]
				];
				recognition,
				oImageIdentify[#, None, categories, n, properties, threshold, confidence, performancegoal, opts]
			]&, 
			img
		],
		If[cacheQ,
			recognition = Internal`CheckImageCache[{"ImageRecognition", key = Hash[img], categories, n, properties, threshold, confidence, performancegoal, opts}];
			If[recognition === $Failed,
				recognition = oImageIdentify[img, key, categories, n, properties, threshold, confidence, performancegoal, targetDevice, opts];
				Internal`SetImageCache[{"ImageRecognition", key, categories, n, properties, threshold, confidence, performancegoal, opts}, recognition]
			];
			recognition,
			oImageIdentify[img, None, categories, n, properties, threshold, confidence, performancegoal, opts]
		]
	]
	
]

oImageIdentify[img_, key_, categories_, n_, properties_, threshold_, confidence_, performancegoal_, targetDevice_, opts_] := 
Module[
	{probabilities, selection, rank, propertyfunction, res},
	(*-- recognition --*)
	
	(* recognition *)
		
	If[!cacheQ || (probabilities = Internal`CheckImageCache[{"ImageRecognition", key, performancegoal}]) === $Failed,

		(* imageClassification returns an association of leaf synsets with probabilities *)
		(* probabilities = CatchMachineLearning[NeuralNetworkImageClassifier[img, PerformanceGoal -> performancegoal]]; *)
		probabilities = imageClassification[img, performancegoal, targetDevice];
		
		(* Update the leaf probabilities with inbuild class priors *)
		probabilities = priorUpdate[leafClassPriors, probabilities];
	
		(* Obtain non-leaf synset probabilities *)
		probabilities = parentProbabilities[probabilities, descendantLeaves];
		
		If[cacheQ,
			Internal`SetImageCache[{"ImageRecognition", key, performancegoal}, probabilities]
		];
	];

	(* filtering *)
	If[Length[categories] > 0,
		With[{span = VertexOutComponent[classifierGraph, categories]},
			probabilities = KeyTake[probabilities, span];
	
			(* the probabilities are renormalised within the selected categories *)
			probabilities/= Total[KeyTake[probabilities, MinimalBy[nodeDepth[#, "BottomDepth"]&][span]]];
			
		]
	];
			
	If[!cacheQ || (rank = Internal`CheckImageCache[{"ImageRecognition", key, performancegoal, categories, confidence}]) === $Failed,
		rank = nodeRank[probabilities, confidence];
		If[cacheQ,
			Internal`SetImageCache[{"ImageRecognition", key, performancegoal, categories, confidence}, rank]
		]
	];
	
	probabilities = KeyDrop[probabilities, Keys@Select[rank, # === 0 &]];
	
	Switch[n,
		_Integer?Positive | All,
		selection = Function[MaximalBy[#, rank, Min[n /. All->Infinity, Length[#]]]]@Keys@Select[# > threshold &][probabilities];
		,
		Automatic,
		selection = MaximalBy[rank]@Keys@Select[# > threshold &][probabilities];
		If[Length[selection] > 1,
			selection = {First[selection]}
		]
	];
	
	
	(*properties*)
	If[properties =!= Automatic,
		res = AssociationThread[
			Lookup[synset2Concept, selection],
			computeProperties[{selection, probabilities, rank}, properties]
		],
		res = Lookup[synset2Concept, selection]
	];
	
	(*result*)
	Which[
		Length[res] == 1 && n === Automatic && Head[res] =!= Association,
		First[res],
		Length[res] == 0,
		Missing["Unidentified"],
		True,
		res
	]
	
]

validGeneralProperties = {"Probability", "Entity", "Concept", "InternalRank", "InternalID"};

PackageScope["validConceptProperties"]
validConceptProperties := validConceptProperties = 
With[{list = Quiet[EntityValue[Entity["Concept"], "Properties"]]},
	If[MatchQ[list, {__EntityProperty}],
		Join[list, list[[All, 2]]],
		{}
	]
];
validProperties := validProperties = Join[validGeneralProperties, validConceptProperties];


computeProperties[data: {synsets_, probabilities_, rank_}, properties_List] :=
	Transpose[computeProperty[data, #]& /@ properties]

computeProperties[data: {synsets_, probabilities_, rank_}, property_String] :=
	computeProperty[data, property]

computeProperty[{synsets_, probabilities_, rank_}, "InternalID"] :=
	synsets

computeProperty[{synsets_, probabilities_, rank_}, "Concept"] :=
	Lookup[synset2Concept, synsets]

computeProperty[{synsets_, probabilities_, rank_}, "Entity" | "EquivalentEntity"] :=
	Lookup[synset2Entity, synsets]

computeProperty[{synsets_, probabilities_, rank_}, "Probability"] :=
	Lookup[probabilities, synsets]

computeProperty[{synsets_, probabilities_, rank_}, "InternalRank"] :=
	Lookup[rank, synsets]

computeProperty[{synsets_, probabilities_, rank_}, p: Alternatives @@ validConceptProperties] :=
	EntityValue[Lookup[synset2Concept, synsets], p]

(****************************************************************************
    ImageInstanceQ
****************************************************************************)

Options[ImageInstanceQ] = {
	RecognitionPrior -> 0.5,
	PerformanceGoal -> Automatic,
	AcceptanceThreshold -> 0.5,
	TargetDevice -> "CPU"
} // SortBy[ToString @* First];

ImageInstanceQHiddenOptions = {
	RecognitionThreshold -> Automatic,
	SpecificityGoal -> Automatic
} // SortBy[ToString @* First];

DefineFunction[
	ImageInstanceQ,
	iImageInstanceQ,
	{2, 3},
	"ExtraOptions" -> ImageInstanceQHiddenOptions
];

(*Implementation*)

iImageInstanceQ[args_, opts_] :=
Scope[
(* 	{
		img, objects, ontologyData,
		categories, threshold,
		performancegoal, recognitionprior, probabilities
	}, *)
	
	(*-- check for the CNN_GraphData paclet to be loaded --*)
	ontologyData = LoadPaclet["CNN_GraphData4", "Construct.m"];
	If[FailureQ[ontologyData],
		DBPrint[ToString[$FunctionName] <> ": could not load the ontology data."];
		$AllowFailure ^= True;
		ThrowFailure["interr", $FunctionName]
	];

	(*-- argument parsing --*)
	
	If[Image`Utilities`SetImage2D[img, args[[1]]],
		img = RemoveAlphaChannel[args[[1]]];
		If[ImageColorSpace[img] =!= "RGB",
			img = ColorConvert[img, "RGB"]
		],
		ThrowFailure["imginv", args[[1]]]
	];
	
	Switch[args[[2]],
		_String|_Entity|{s__String},
		objects = {args[[2]]},
		
		Verbatim[Alternatives][(_String|_Entity|{__String})..],
		objects = List@@args[[2]],
		
		_,
		ThrowFailure["objinv", args[[2]]]
	];
	
	If[AnyTrue[Lookup[objectInterpretation, objects] , Head[#] === Missing &],
		objects = Replace[objects, Entity["Word", w_] :> w, 1];
		If[AnyTrue[Lookup[objectInterpretation, objects] , Head[#] === Missing &],
			ThrowFailure["objinv", args[[2]]],
			objects = Intersection[Flatten[Lookup[objectInterpretation, objects]], VertexList[classifierGraph]]
		],
		objects = Intersection[Flatten[Lookup[objectInterpretation, objects]], VertexList[classifierGraph]]
	];
	
	If[Length[args] > 2,
		Switch[args[[3]],
			All,
			categories = {},
			
			_String|_Entity|{__String},
			categories = {args[[3]]},
			
			Verbatim[Alternatives][(_String|_Entity|{__String})..] | {(_String|_Entity|{__String})..},
			categories = List@@args[[3]],
			
			_,
			ThrowFailure["catinv", args[[3]]]
		],
		categories = {}
	];
	If[Length[categories] > 0,
		If[AnyTrue[Lookup[objectInterpretation, categories] , Head[#] === Missing &],
			categories = Replace[categories, Entity["Word", w_] :> w, 1];
			If[AnyTrue[Lookup[objectInterpretation, categories] , Head[#] === Missing &],
				ThrowFailure["catinv", args[[3]]],
				categories = Intersection[Flatten[Lookup[objectInterpretation, categories]], VertexList[classifierGraph]]
			],
			categories = Intersection[Flatten[Lookup[objectInterpretation, categories]], VertexList[classifierGraph]]
		]
	];
	
	
	(*-- option parsing --*)
	
	(*manual threshold specification*)
	If[
		FilterRules[opts, RecognitionThreshold] =!= {},

		Message[ImageInstanceQ::deprec, RecognitionThreshold, AcceptanceThreshold];
		If[
			FilterRules[opts, AcceptanceThreshold]  === {},
			threshold = GetOption[RecognitionThreshold];
			thresholdMessage = "thrs2",

			threshold = GetOption[AcceptanceThreshold];
			thresholdMessage = "thrs"
		],

		threshold = GetOption[AcceptanceThreshold];
		thresholdMessage = "thrs"
	];

	If[threshold =!= Automatic,
		If[!(RealValuedNumericQ[threshold] && 0 <= threshold <=1),
			ThrowFailure[Evaluate[thresholdMessage], threshold],
			threshold = Max[threshold, 10^-5]
		],
		threshold = .5
	];

	(* prior on the queried objects *)
	recognitionprior = GetOption[RecognitionPrior];
	If[RealValuedNumericQ[recognitionprior],
		recognitionprior = Clip[recognitionprior, {0, 1}],
		recognitionprior = .5
	];
	
	(*classifier performance target*)
	performancegoal = GetOption[PerformanceGoal];
	If[!MatchQ[performancegoal, "Quality"|"Speed"|Automatic],
		ThrowFailure["mlbdpg", performancegoal]
	];

	targetDevice = GetOption[TargetDevice];
	If[!TrueQ[NeuralNetworks`TestTargetDevice[targetDevice, $FunctionName]],
		ThrowFailure[]
	];

	(*-- recognition --*)
	
	(* recognition *)
		(* NeuralNetworkImageClassifier: returns an association of leaf synsets with probabilities *)
	probabilities = imageClassification[img, performancegoal, targetDevice];

	(* Update the leaf probabilities with inbuild class priors *)
	probabilities = priorUpdate[leafClassPriors, probabilities];
	
	(* applying recognition prior *)
	With[{span = VertexOutComponent[classifierGraph, objects]},
	Module[{p1,p2},	

		p1 = probabilities[[Key/@Intersection[span, Keys@probabilities]]];
		p2 = probabilities[[Key/@Complement[Keys@probabilities, Keys@p1]]];
		
		p1 *= recognitionprior;
		p2 *= (1-recognitionprior);
		
		probabilities = Join[p1, p2];
		probabilities /= Total[probabilities];
	]];
	
	(* Obtain non-leaf synset probabilities *)
	probabilities = parentProbabilities[probabilities, descendantLeaves];
		
	(* filtering *)
	If[Length[categories] > 0,
		With[{span = VertexOutComponent[classifierGraph, categories]},
			probabilities = KeyTake[probabilities, span];
			
			(* the probabilities are renormalised within the selected categories *)
			probabilities/= Total[KeyTake[probabilities, MinimalBy[span, nodeDepth[#, "BottomDepth"]&]]]
			
		]
	];
	
	(*result*)
	(* the oject is identified if any of its children is present in the image *)
	TrueQ[
		IntersectingQ[
			Keys[Select[probabilities, GreaterThan[threshold]]],
			VertexOutComponent[classifierGraph, objects]
		]
	]
]


(****************************************************************************
    UTILITIES
****************************************************************************)


(* This function returns an association of entities (mainly concepts) and the related probability *)
imageClassification[image_, performancegoal_, targetDevice_] :=
Scope[

	net = GetNetModel["Wolfram ImageIdentify Net V1"];

	SafeNetEvaluate[
		net[image, "Probabilities", TargetDevice -> targetDevice],
		AssociationQ
	]
]

(*--- variables ---*)

(*ontologyData keys:

{"VertexCentrality", "WordNetDictionary", "WordNetReverseDictionary",
"Graph", "SynsetInterpretation", "ObjectInterpretation",
"AncestorLeafRelation", "PopularityScore", "NodeDepth",
"LeafClassPriors", "Synset2Concept", "Synset2Entity", "ScorePrior"}*)

classifierGraph      := ontologyData["Graph"];
descendantLeaves     := ontologyData["AncestorLeafRelation"];
nodeDepth            := ontologyData["NodeDepth"];
defaultParameters    := ontologyData["DefaultParameters"];

leafClassPriors      := ontologyData["LeafClassPriors"];
scorePrior           := ontologyData["ScorePrior"];

synset2Entity        := ontologyData["Synset2Entity"];
synset2Concept       := ontologyData["Synset2Concept"];
objectInterpretation := ontologyData["ObjectInterpretation"];


(*--- functions ---*)

(* nodeRank
	Inputs: association of graph nodes probabilities, required confidence level [0,1]
	Options:
	Outputs: ranking function
*)

nodeRank[probabilities_Association, confidence_?NumericQ] :=
Module[
	{
		keys, leaves, depth,
		bestleaf, bestnode, maxP, minP, maxDepth, minDepth,
		scaledProp, scaledDepth
	},

	keys = Keys[probabilities];
	leaves = KeyTake[probabilities, Keys[leafClassPriors]];
	depth = nodeDepth[#, "TopDepth"]&;
	bestleaf = FirstPosition[leaves, Max[leaves]][[1,1]];
	bestnode = First@MinimalBy[depth]@Keys@MaximalBy[Identity]@probabilities;

	maxP = probabilities[bestnode];
	minP = probabilities[bestleaf];
	maxDepth = nodeDepth[bestleaf, "TopDepth"];
	minDepth = nodeDepth[bestnode, "TopDepth"];

	If[minP === maxP,
		If[maxDepth === minDepth, 
			Return[probabilities],
			scaledProp = Values@probabilities
		],
		scaledProp = Rescale[Values@probabilities, {minP, maxP}]
	];
	scaledDepth = Clip[Rescale[Lookup[nodeDepth, keys][[All, "TopDepth"]], {minDepth, maxDepth}], {0, 1}];

	AssociationThread[
		keys,
		Lookup[scorePrior, keys, 1] * 
			(confidence * scaledProp + (1 - confidence) * scaledDepth)
	]
]

(* priorCorrect
	Inputs: association of prior probabilites, association of probabilities
	Outputs: association of update probabilities
*)	

priorUpdate[priors_Association, probabilities_Association] := Module[
	{sortedPriorKey, probs}
	,
	sortedPriorKey = KeyTake[priors, Keys@probabilities];
	probs = Values@probabilities * Values@sortedPriorKey;
	probs = probs/Total[probs];
	AssociationThread[Keys@sortedPriorKey -> probs]
]

(* parentProbabilities
	Inputs: assocation of leaf probabilities, association of lists of descendant leaves per synset
	Outputs: association of probabilities for each synset (leaf + parent)
*)

parentProbabilities[leafProbs_Association, descendantLeaves_Association] := 
	Join[Total[Lookup[leafProbs, #]] & /@ descendantLeaves, leafProbs]

