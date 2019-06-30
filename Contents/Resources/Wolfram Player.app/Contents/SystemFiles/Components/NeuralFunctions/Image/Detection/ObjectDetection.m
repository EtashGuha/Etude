
(* Functions *)
PackageExport["ImageContents"]
PackageExport["ImageBoundingBoxes"]
PackageExport["ImageCases"]
PackageExport["ImagePosition"]
PackageExport["ImageContainsQ"]

Developer`ToList;

Options[ImageContents] =
Options[ImagePosition] =
Options[ImageBoundingBoxes] =
Options[ImageCases] =
Options[ImageContainsQ] = {
	MaxFeatures -> Infinity,
	AcceptanceThreshold -> Automatic,
	MaxOverlapFraction -> Automatic,
	TargetDevice -> "CPU"
} // SortBy[ToString @* First];

hiddenOptions = {
	"NMSMode" -> "Global"
} // SortBy[ToString @* First];

Map[
	DefineFunction[#, detectionRouter, {1, 2}, "ExtraOptions" -> hiddenOptions]&,
	{ImagePosition, ImageBoundingBoxes}
]

(* ImageCases must be defined using ArgumentsWithRules to support
the category -> prop syntax *)
DefineFunction[ImageCases, detectionRouter, {1, 2},
	"ExtraOptions" -> hiddenOptions,
	"ArgumentsParser" -> System`Private`ArgumentsWithRules
]

DefineFunction[ImageContainsQ, detectionRouter, 2, "ExtraOptions" -> hiddenOptions]

DefineFunction[ImageContents, detectionRouter, {0, 3}, "ExtraOptions" -> hiddenOptions]

detectionRouter[args_, opts_] :=
Block[
	{
		img, obj, test, properties, conceptProperties,
		threshold, maxFeatures, res, ontologyData,
		classMessage, minThreshold, targetDevice
	},

	(* zero arg *)

	If[
		And[
			$FunctionName === ImageContents,
			Length[args] == 0
		],
		Return[detectionLabels]
	];

	(* image parsing *)

	If[!Image`Utilities`SetImage2D[img, args[[1]]],
		ThrowFailure["imginv", args[[1]]]
	];
	If[ImageColorSpace[img] =!= "RGB",
		img = ColorConvert[img, "RGB"]
	];
	(* This is a somewhat arbitrary assumption that transparent images
	will be seen on a white background *)
	If[Image`ImageInformation[img, "Transparency"],
		img = RemoveAlphaChannel[img, White]
	];

	(*-- check for the CNN_GraphData paclet to be loaded --*)
	ontologyData = LoadPaclet["CNN_GraphData4", "Construct.m"];


	(* object/property parsing *)

	If[
		Length[args] > 1,

		If[Head[args[[2]]] === Alternatives,
			test = ContainsAny,
			test = ContainsAll
		];
		obj = args[[2]],

		obj = All
	];

	(* property argument for ImageContents *)
	Which[
		Length[args] > 2 && $FunctionName === ImageContents,
			obj = obj -> args[[3]],

		$FunctionName === ImageContents,
			obj = obj -> Automatic
	];

	{obj, properties} = parseObj[obj];

	(* option parsing *)

	threshold = GetOption[System`AcceptanceThreshold];
	Which[
		threshold === Automatic,
		threshold = 0.5,
		RealValuedNumericQ[threshold],
		threshold = Clip[threshold, {0, 1}],
		True,
		ThrowFailure["thrs", threshold]
	];

	maxFeatures = GetOption[MaxFeatures];
	If[
		Or[
			MatchQ[maxFeatures, Alternatives[Infinity, All, Automatic]],
			Internal`RealValuedNumericQ[maxFeatures] && maxFeatures >= 0
		],
		maxFeatures = Replace[maxFeatures, {
			Automatic|All -> Infinity,
			x_ :> Ceiling[x]
		}],
		ThrowFailure["maxf", maxFeatures]
	];

	maxOverlap = GetOption[MaxOverlapFraction];
	If[RealValuedNumericQ[maxOverlap],
		maxOverlap = Clip[maxOverlap, {0, 1}],
		maxOverlap = 0.3
	];

	targetDevice = GetOption[TargetDevice];
	If[!TrueQ[NeuralNetworks`TestTargetDevice[targetDevice, $FunctionName]],
		ThrowFailure[]
	];

	(* detection *)

	res = detection[img, threshold, targetDevice];

	(* empty detection result *)
	If[obj =!= All,
		res	= Select[
			res,
			MatchQ[#Concept, Alternatives @@ ToList[obj]]&
		];

		If[Length[res] == 0,
			Return[emptyDetectionResult[$FunctionName]]
		]
	];

	(* shortcut for an empty detection, ImageContainsQ can already return its result *)
	Which[
		Length[res] == 0,
		Return[emptyDetectionResult[$FunctionName]],

		$FunctionName === ImageContainsQ,
		Return[detectionResult[$FunctionName]]
	];

	(* non max suppression *)
	Switch[GetOption["NMSMode"],
	(* class dependent *)
		"Class",
		res = Join @@ Values[
			GroupBy[
				res, #Concept &, nonMaxSuppression[maxOverlap] @* SortBy[-#Probability&]
			]
		],
	(* class independent *)
		"Global",
		res = nonMaxSuppression[maxOverlap][SortBy[res, -#Probability&]]
	];

	(* sort result *)
	With[{imageArea = Times @@ ImageDimensions[img]},
		res = SortBy[res, - #Probability + .1 Area[#BoundingBox]/imageArea &]
	];

	(* build an association with the results *)
	(* TODO: optimize by not computing unnecessary properties, especially "Image" *)
	res = Association[
		#,
		"Entity" -> synset2Entity[#Concept],
		"Image" -> ImageTrim[img, List @@ #BoundingBox],
		"Position" -> Mean[N[List @@ #BoundingBox]],
		"Dimensions" -> Abs[Subtract @@ #BoundingBox]
	] & /@ res;

	(* compute extra properties of the "Concept" entities *)
	If[ContainsAny[ToList[properties], validConceptProperties],
		conceptProperties = getConceptProperties[];
		res = Join @@@ Transpose[{res, conceptProperties}]
	];

	(* final result computation *)

	detectionResult[$FunctionName]

]

detectionPropertyQ[p_] := ValidPropertyQ[$FunctionName, p, validProperties, extraProperties]

parseObj[expr_] :=
Module[
	{classMessage, dict, classes, properties, fixList},

	classMessage[o_] := ThrowFailure[
		Evaluate@Switch[
			$FunctionName,
			ImageContainsQ, "objinv",
			_, "catinv"
		],
		o
	];

	fixList = If[ListQ[#], Alternatives @@ #, #]&;

	Switch[
		expr,

		_Rule | _RuleDelayed /; MatchQ[$FunctionName, ImageCases | ImageContents],
		classes = First[expr];
		properties = Replace[Last[expr], {
				All -> validDetectionProperties
			}
		];

		If[
			ListQ[properties],
			If[
				!AllTrue[properties, detectionPropertyQ],
				ThrowFailure[]
			],

			If[
				!detectionPropertyQ[properties],
				ThrowFailure[]
			]
		]

		,

		_,
		classes = expr;
		properties = Automatic
	];

	(* the default value for Automatic depends on the calling function *)
	If[properties === Automatic,
		properties = Switch[
			$FunctionName,
			ImageCases, "Image",
			ImagePosition, "Position",
			ImageBoundingBoxes, "BoundingBox",
			ImageContents, $ImageContentsProperties,
			_, None
		]
	];

	If[
		classes =!= All,

		dict = Join[
			objectInterpretation,
			AssociationThread[detectionLabels, detectionLabels]
		];

		classes = If[ListQ[classes], Identity, First] @ Replace[
			Developer`ToList[classes],
			{
				o_Alternatives :>
					fixList@Flatten@Lookup[dict, List @@ o, classMessage[o]],
				o_ :>
					fixList@Lookup[dict, o, classMessage[o]]
			},
			1
		],

		classes = detectionLabels
	];

	{classes, properties}
]

emptyDetectionResult[ImageContainsQ] := False
emptyDetectionResult[ImageContents] := Missing["NotRecognized"]
emptyDetectionResult[_] := {}

detectionResult[ImageCases | ImagePosition | ImageBoundingBoxes] :=
Module[
	{p, final},

	final = GroupBy[
		Take[res, UpTo[maxFeatures]],
		#Concept&,
		Lookup[properties]
	];

	If[ListQ[obj],
		final,
		Join @@ final
	]

]

detectionResult[ImageContainsQ] := test[Lookup[res, "Concept"], Flatten[{obj /. Alternatives -> List}]]

detectionResult[ImageContents] :=
	Dataset[KeyTake[Take[res, UpTo[maxFeatures]], properties]]

$ImageContentsProperties = {"Image", "Concept", "BoundingBox", "Probability"};

validDetectionProperties = {
	"Image",
	"BoundingBox",
	"Probability",
	"Dimensions",
	"Position"
} // Sort

(* "Image" is removed because we don't want to use the property of the concept
in place of the image trim *)
validConceptProperties = DeleteCases[validConceptProperties, "Image"]

validProperties = Join[
	validDetectionProperties,
	validConceptProperties
]

(* these properties are valid but should not be suggested by ValidPropertyQ *)
extraProperties = {All, Automatic, "Concept"}

(* EntityValue call to get the extra properties of the "Concept" entities *)
(* The most common messages are quieted and replaced with appropriate ones *)
getConceptProperties[] :=
Module[
	{extra, messages},
	extra = Quiet[
		Check[
			EntityValue[
				Lookup[res, "Concept"],
				Cases[ToList[properties], Alternatives @@ validConceptProperties],
				"PropertyAssociation"
			],
			messages = $MessageList;
			$Failed
		],
		{General::conopen, EntityValue::nodat}
	];

	Which[
		MemberQ[messages, HoldForm[EntityValue::conopen]],
			ThrowFailure["conopen", $FunctionName],

		FailureQ[extra] || Length[extra] =!= Length[res],
			ThrowFailure["interr", "EntityValue error"],

		True,
			extra
	]
]

(****************************************************************************
		Detection framework
****************************************************************************)

detectionLabels := detectionLabels =
	Lookup[Normal[NetExtract[GetNetModel["Wolfram ImageCases Net V1"], "Class"]], "Labels"];

detection[img_, detectionThreshold_ : .2, targetDevice_] :=
Module[
	{
		w, h, scale, coords, classes,
		boxes, padding, classProb,
		probable, finals
	},

	{coords, classes} = Values[
		SafeNetEvaluate[
			GetNetModel["Wolfram ImageCases Net V1"][
				img,
				{"BoundingBox", "Class" -> {"TopProbabilities", 1}},
				TargetDevice -> targetDevice
			],
			AssociationQ
		]
	];

	(* transform coordinates into rectangular boxes *)
	{w, h} = ImageDimensions[img];
	scale = Max[{w, h}] / 416;
	padding = 416 (1 - {w, h} / Max[{w, h}]) / 2;
	boxes = Apply[
		Rectangle @@ Transpose[{
			Clip[scale (#1 + {- #3/2, + #3/2} - padding[[1]]), {0, w}],
			Clip[scale (416 - #2 + {- #4/2, + #4/2} - padding[[2]]), {0, h}]
		}] &,
		416 * Normal[coords],
		1
	];

	(* each class probability is rescaled with the box objectivness *)
	classProb = classes[[All, 1, 2]];
	classes = classes[[All, 1, 1]];
	(* filter by probability*)
	(*very small probability are thresholded *)
	probable = Position[classProb, p_ /; p - 10^-2 > detectionThreshold];
	If[Length[probable] == 0, Return[{}]];

	(* gather the boxes of the same class and perform non-max suppression *)
	finals = Join @@ Values @ GroupBy[
			Transpose[{
				Extract[boxes, probable],
				Extract[classes, probable],
				Extract[classProb, probable]
			}],
		#[[2]] &
	];
	AssociationThread[
		{"BoundingBox", "Concept", "Probability"},
		#
	] & /@ finals
]


(* Non-max suppression *)

nonMaxSuppression[overlapThreshold_][detection_] :=
Module[
	{boxes, confidence},
	Fold[
		Function[
			{list, new},
			If[
				NoneTrue[list[[All, 1]], IoU[#, new[[1]]] > overlapThreshold &],
				Append[list, new],
				list
			]
		],
		Sequence @@ TakeDrop[Reverse@SortBy[detection, Last], 1]
	]
]

ClearAll[IoU]
IoU := IoU =
With[{c =
	Compile[
		{{box1, _Real, 2}, {box2, _Real, 2}},
	Module[
		{area1, area2, x1, y1, x2, y2, w, h, int},

			area1 = (box1[[2, 1]] - box1[[1, 1]]) (box1[[2, 2]] - box1[[1, 2]]);
			area2 = (box2[[2, 1]] - box2[[1, 1]]) (box2[[2, 2]] - box2[[1, 2]]);

			x1 = Max[box1[[1, 1]], box2[[1, 1]]];
			y1 = Max[box1[[1, 2]], box2[[1, 2]]];
			x2 = Min[box1[[2, 1]], box2[[2, 1]]];
			y2 = Min[box1[[2, 2]], box2[[2, 2]]];

			w = Max[0., x2 - x1];
			h = Max[0., y2 - y1];

			int = w * h;

			int / (area1 + area2 - int)
	],
		RuntimeAttributes -> {Listable},
		Parallelization -> True,
		RuntimeOptions -> "Speed"
	]},
	c @@ Replace[{##}, Rectangle -> List, Infinity, Heads -> True] &
]


(*--- variables ---*)


(*$OntologyData keys:

{"VertexCentrality", "WordNetDictionary", "WordNetReverseDictionary",
"Graph", "SynsetInterpretation", "ObjectInterpretation",
"AncestorLeafRelation", "PopularityScore", "NodeDepth",
"LeafClassPriors", "Synset2Concept", "Synset2Entity", "ScorePrior"}*)

synset2Entity        := ontologyData["Synset2Entity"];
synset2Concept       := ontologyData["Synset2Concept"];
objectInterpretation := ontologyData["ObjectInterpretation"];
