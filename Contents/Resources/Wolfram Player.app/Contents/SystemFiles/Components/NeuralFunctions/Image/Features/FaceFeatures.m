
PackageImport["GeneralUtilities`"]

(* Functions *)
PackageExport["FacialFeatures"]

Options[FacialFeatures] = {
	Method -> Automatic
}

$FacialFeaturesHiddenOptions = {
	"Caching" -> True,
	"OutputFormat" -> Automatic
};

$subValuesPattern = _String;

FacialFeatures[feature: $subValuesPattern][image_] :=
With[{res = FacialFeatures[image, feature]},
	res /; Head[res] =!= FacialFeatures
]

DefineFunction[
	FacialFeatures,
	iFacialFeatures,
	{1, 2},
	"ExtraOptions" -> $FacialFeaturesHiddenOptions,
	"ArgumentsPattern" -> Alternatives[
		Repeated[Blank[], {2, Infinity}],
		PatternSequence[Except[$subValuesPattern], ___]
	]
]

iFacialFeatures[args_, opts_] := Scope[

	(* Argument parsing *)

	images = None;
	If[
		!Or[
			Image`Utilities`SetImage2D[images, args[[1]]],
			Image`Utilities`SetImageList2D[images, args[[1]]]
		],
		ThrowFailure["imginv", args[[1]]]
	];
	If[!VectorQ[images], images = {images}];

	If[Length[args] > 1 && args[[2]] =!= Automatic,
		Which[
			ContainsOnly[Flatten[{args[[2]]}, 1], Image`HumanDump`$AvailableFaceFeatures],
			features = args[[2]],

			args[[2]] === All,
			features = Image`HumanDump`$AllFaceFeatures,

			args[[2]] === "Properties",
			Return[Image`HumanDump`$AvailableFaceFeatures],

			True,
			ThrowFailure["featinv", args[[2]]]
		],
		summaryQ = True;
		features = $DefaultFaceFeatures
	];

	(* Preprocessing *)

	faceBoxes = Lookup[
		Cases[
			Developer`ToList[GetOption[Method]],
			_Rule | _RuleDelayed
		],
		"FaceBoxes",
		Automatic
	];

	singleFaceQ = False;
	Switch[faceBoxes,

		Automatic,
		faceBoxes = None;
		preprocessor = Automatic,

		_?boundingBoxQ,
		singleFaceQ = True;
		faceBoxes = ConstantArray[
			{normalizeBoundingBox[faceBoxes]},
			Length[images]
		];
		preprocessor = None,

		{_?boundingBoxQ ..},
		faceBoxes = ConstantArray[
			normalizeBoundingBox /@ faceBoxes,
			Length[images]
		];
		preprocessor = None,

		None | {},
		faceBoxes = ConstantArray[{}, Length[images]];
		preprocessor = None,

		All | Full,
		singleFaceQ = True;
		faceBoxes = None;
		preprocessor = None,

		_,
		preprocessor = faceBoxes;
		faceBoxes = None
	];


	preprocessor = Replace[
		preprocessor,
		Automatic -> $defaultPreprocessor
	];
	If[faceBoxes === None,
		faceBoxes = applyPreprocessor[preprocessor, images]
	];

	dims = MapThread[
		ConstantArray[ImageDimensions[#], Length[#2]]&,
		{images, faceBoxes}
	];

	images = trimFaces[images, faceBoxes];

	faces = DeleteMissing[MapThread[
		Association[
			"Image" -> #1,
			"BoundingBox" -> #2,
			"OriginalSize" -> #3
		]&,
		{Flatten[images], Flatten[faceBoxes], Flatten[dims, 1]}
	], 1, 2];
	shape = Length /@ Replace[faceBoxes, _Missing | {_Missing} :> {}, 1];

	(* Feature computation *)
	If[Length[faces] > 0,
		$Caching = TrueQ[GetOption["Caching"]];
		computedFeatures = AssociationMap[
			Image`HumanDump`getFaceFeature[#][faces]&,
			Flatten[{features}, 1]
		];

		computedFeatures = Transpose[computedFeatures, AllowedHeads -> All],

		computedFeatures = ConstantArray[Association[], Length[images]]
	];

	format = GetOption["OutputFormat"];

	Switch[ToString @ format,

		"List" | ("Automatic" /; StringQ[features]),
		res = Lookup[computedFeatures, features, Missing["NotRecognized"]];
		res = TakeList[res, shape],

		"Dataset",
		res = KeyTake[computedFeatures, features];
		res = Dataset /@ TakeList[res, shape],

		"Association" | _,
		res = KeyTake[computedFeatures, features];
		res = TakeList[res, shape]
	];

	(* If there's no preprocessor only a single face per image is assumed *)
	(* so one level is removed. *)
	(* Note: do not use [[All, 1]] as Dataset does not like it *)
	If[singleFaceQ,
		res = First /@ res
	];

	(* If only one image is present, remove another level *)
	If[VectorQ[args[[1]]],
		res,
		First[res]
	]
]

(* Preprocessing *)

$defaultPreprocessor := Function[{img},
	With[
		{
			size = ImageDimensions[img],
			faces = FindFaces[img]
		},
		Map[
			expandRectangle[#, 10, size]&,
			faces
		]
	]
]

expandRectangle[
	Rectangle[{xmin_, ymin_}, {xmax_, ymax_}],
	pad_,
	{maxW_, maxH_}
] :=
Rectangle @@ Transpose[{
	Clip[{xmin - pad, xmax + pad}, {0, maxW}],
	Clip[{ymin - pad, ymax + pad}, {0, maxH}]
}]

applyPreprocessor[Automatic, image_] :=
	applyPreprocessor[$defaultPreprocessor, image]

applyPreprocessor[None, image_?ImageQ] :=
	{Rectangle[{0,0}, ImageDimensions[image]]}

applyPreprocessor[f_, image_?VectorQ] :=
	Map[applyPreprocessor[f, #]&, image]

applyPreprocessor[f_, image_?ImageQ] :=
Scope[
	faces = f[image];
	Which[
		MatchQ[faces, {} | {{}} | _Missing],
		{Missing[]},

		boundingBoxQ[faces],
		singleFaceQ ^= True
		{normalizeBoundingBox[faces]},

		VectorQ[faces, boundingBoxQ],
		normalizeBoundingBox /@ faces,

		True,
		ThrowFailure["prepinv", f]
	]
]

boundingBoxQ[expr_List] :=
	MatrixQ[expr, Internal`RealValuedNumericQ] && Dimensions[expr] == {2, 2}

boundingBoxQ[expr_Rectangle] :=
	boundingBoxQ[List @@ expr]

boundingBoxQ[_] := False

normalizeBoundingBox[expr_Rectangle ? boundingBoxQ] := expr
normalizeBoundingBox[expr_List ? boundingBoxQ] := Rectangle @@ expr
normalizeBoundingBox[expr_List] := $Failed

trimFaces[image_?ImageQ, box_?boundingBoxQ] := ImageTrim[image, List @@ box]
trimFaces[image_?ImageQ, All] := image
trimFaces[image_?ImageQ, _?MissingQ] := Missing[]

trimFaces[image_?ImageQ, boxes_] := Map[trimFaces[image, #]&, boxes]
trimFaces[image_?ListQ, boxes_] := MapThread[trimFaces, {image, boxes}]



(* Features *)

baseElements = {
	"Age",
	"Gender",
	"Emotion",
	"Landmarks",
	"Descriptor",
	"AgeWeights",
	"GenderWeights",
	"EmotionWeights"
(* 	"HairColor",
	"EyeColor",
	"SkinTone" *)
};

$2DFanLandmarks = {
	"OutlinePoints" -> {1, 17},
	"RightEyebrowPoints" -> {18, 22},
	"LeftEyebrowPoints" -> {23, 27},
	"NosePoints" -> {28, 36},
	"RightEyePoints" -> {37, 42},
	"LeftEyePoints" -> {43, 48},
	"MouthExternalPoints" -> {49, 60},
	"MouthInternalPoints" -> {61, 68}
};

$2DFanLanRegion = {
	"OutlinePoints" -> Line,
	"RightEyePointsbrowPoints" -> Line,
	"LeftEyePointsbrowPoints" -> Line,
	"NosePoints" -> Line@*(TakeDrop[#, 4]&),
	"RightEyePoints" -> Polygon,
	"LeftEyePoints" -> Polygon,
	"MouthExternalPoints" -> Polygon,
	"MouthInternalPoints" -> Polygon
};

$VanillaCNNLandmarks = {
	"RightEye" -> {1, 1},
	"LeftEye" -> {2, 2},
	"Mouth" -> {3, 4},
	"Nose" -> {5, 5}
};

faceElements = Keys[$2DFanLandmarks];

groupFaceElements = {
	"EyePoints" -> {"RightEyePoints", "LeftEyePoints"},
	"EyebrowPoints" -> {"RightEyePointsbrowPoints", "LeftEyePointsbrowPoints"},
	"MouthPoints" -> {"MouthExternalPoints", "MouthInternalPoints"}
}

$DefaultFaceFeatures = {
	"Image",
	"Age",
	"Gender",
	"Emotion"
}

Unprotect["Image`HumanDump`$AllFaceFeatures"];
Image`HumanDump`$AllFaceFeatures = Join[
	Image`HumanDump`$AllFaceFeatures,
	DeleteCases[baseElements, "Landmarks"],
	faceElements
] // Union;
Protect["Image`HumanDump`$AllFaceFeatures"];

Unprotect["Image`HumanDump`$AvailableFaceFeatures"];
Image`HumanDump`$AvailableFaceFeatures = Join[
	Image`HumanDump`$AvailableFaceFeatures,
	baseElements,
	faceElements,
	Keys[groupFaceElements]
] // Union;
Protect["Image`HumanDump`$AvailableFaceFeatures"];


numericAssocQ[a_?ListQ] := VectorQ[a, numericAssocQ]
numericAssocQ[a_] := And[AssociationQ[a], VectorQ[Values[a], NumericQ]]

netResult["NumericAssociation"] := numericAssocQ
netResult["String"] := Function[
	MatchQ[#, Indeterminate | _String | {(_String | Indeterminate) ..}]
]
netResult["Entity"] := Function[
	MatchQ[#, Indeterminate | _Entity | {(_Entity | Indeterminate) ..}]
]

(* Previous definitions are in StartUp/ImageProcessing/Human/Faces.m *)
Unprotect[Image`HumanDump`getRawFaceFeature];

getRawFaceFeature = Image`HumanDump`getRawFaceFeature;

(* Classify-based properties *)

getRawFaceFeature["Age"][face_] :=
	Lookup[ageDistribution[Lookup[face, "Image"]], "Mean"]
getRawFaceFeature["AgeWeights"][face_] :=
	Lookup[ageDistribution[Lookup[face, "Image"]], "Distribution"]

getRawFaceFeature["Gender"][face_] :=
	Cached[SafeNetEvaluate[
		Classify["FacialGender", Lookup[face, "Image"], TrainingProgressReporting -> None],
		netResult["String"]
	]]
getRawFaceFeature["GenderWeights"][face_] :=
	Cached[SafeNetEvaluate[
		Classify["FacialGender", Lookup[face, "Image"], "Probabilities", TrainingProgressReporting -> None],
		netResult["NumericAssociation"]
	]]

getRawFaceFeature["Emotion"][face_] :=
	Cached[SafeNetEvaluate[
		Classify["FacialExpression", Lookup[face, "Image"], TrainingProgressReporting -> None],
		netResult["Entity"]
	]]
getRawFaceFeature["EmotionWeights"][face_] :=
	Cached[SafeNetEvaluate[
		Classify["FacialExpression", Lookup[face, "Image"], "Probabilities", TrainingProgressReporting -> None],
		netResult["NumericAssociation"]
	]]

(* Other NN-based properties *)

getRawFaceFeature["Descriptor"][face_] :=
	Cached[SafeNetEvaluate[
		Normal@GetNetModel["ResNet-101 Trained on Augmented Casia WebFace Data"][Lookup[face, "Image"]],
		 ArrayQ
	]]

getRawFaceFeature[f: Alternatives @@ faceElements][face_] :=
	Lookup[getRawFaceFeature["Landmarks"][face], f]

getRawFaceFeature[f: Alternatives @@ Keys[groupFaceElements]][face_] :=
	KeyTake[
		getRawFaceFeature["Landmarks"][face],
		Replace[f, groupFaceElements]
	]

getRawFaceFeature["Landmarks"][face_] :=
	Cached[evaluationFunction[face, "2DFaceAlignmentNet"]]

Protect[Image`HumanDump`getRawFaceFeature];

(* Landmarks *)

(* Note: should the rescaling happen per image or per face? *)
rescaleTo[landmarks_?MatrixQ, dims_, offset_: {0, 0}] :=
Transpose[
	MapThread[
		Rescale[#1, {0, 1}, #2] + #3 &,
		{
			Transpose[landmarks],
			Tuples[{{0}, dims}],
			offset
		}
	]
]

getOffset[faces_List] := Lookup[faces, "BoundingBox"][[All, 1]]

(* "Vanilla CNN for Facial Landmark Regression" *)

evaluationFunction[faces_, "VanillaCNN", flip_: True] :=
Block[
	{imgList, landmarks, groups, res},

	imgList = Flatten[{Lookup[faces, "Image"]}];
	offsets = Flatten[{getOffset[faces]}, 1];

	landmarks = SafeNetEvaluate[Normal@GetNetModel[
		"Vanilla CNN for Facial Landmark Regression"][imgList],
		 ArrayQ
	];

	landmarks = MapThread[
		rescaleTo[#, ImageDimensions[#2], #3]&,
		{landmarks, imgList, offsets}
	];

	groups = Flatten[Differences /@ Values[$VanillaCNNLandmarks] + 1];

	res = AssociationThread[
		Keys[$VanillaCNNLandmarks],
		TakeList[#, groups]
	]& /@ landmarks;

	If[MatchQ[faces, _List],
		res,
		First[res]
	]
]

(* "2D Face Alignment Net Trained on 300W Large Pose Data" *)

evaluationFunction[faces_, "2DFaceAlignmentNet", flip_: True] :=
Block[
	{imgList, heatmaps, landmarks, groups, res},

	imgList = Flatten[{Lookup[faces, "Image"]}];
	offsets = Flatten[{getOffset[faces]}, 1];

	heatmaps = SafeNetEvaluate[
		Normal@GetNetModel["2D Face Alignment Net Trained on 300W Large Pose Data"][imgList],
		ArrayQ
	];

	If[flip,
		heatmaps += flippedHeatmaps[imgList, GetNetModel["2D Face Alignment Net Trained on 300W Large Pose Data"]]
	];

	landmarks = postProcess[heatmaps];

	landmarks = MapThread[
		rescaleTo[#, ImageDimensions[#2], #3]&,
		{landmarks, imgList, offsets}
	];

	groups = Flatten[Differences /@ Values[$2DFanLandmarks] + 1];

	(* right now, landmarks are returned as points but this may change *)
	(* again in the future *)
(* 	res = Merge[{
			$2DFanLanRegion,
			AssociationThread[
				Keys[$2DFanLandmarks],
				TakeList[#, groups]
			]
		},
		#[[1]][#[[2]]]&
	]& /@ landmarks; *)
	res = AssociationThread[
		Keys[$2DFanLandmarks],
		TakeList[#, groups]
	]& /@ landmarks;

	If[MatchQ[faces, _List],
		res,
		First[res]
	]
]

flippedHeatmaps[imgs_, net_] :=
Block[

	{matchedLandmarks, hm},

	matchedLandmarks = {{1, 17}, {2, 16}, {3, 15}, {4, 14}, {5, 13},
		{6, 12}, {7, 11}, {8, 10}, {18, 27}, {19, 26}, {20, 25}, {21, 24},
		{22, 23}, {37, 46}, {38, 45}, {39, 44}, {40, 43}, {42, 47},
		{41, 48}, {32, 36}, {33, 35}, {51, 53}, {50, 54}, {49, 55},
		{62, 64}, {61, 65}, {68, 66}, {60, 56}, {59, 57}
	};
	hm = SafeNetEvaluate[Normal@net[ImageReflect[#, Left] & /@ imgs],  ArrayQ];
	hm = Permute[#, Cycles[matchedLandmarks]] & /@ hm;
	Map[Reverse, hm, {3}]
]

postProcess[heatmaps_] :=
Block[
	{posFlattened, posMat},
	posFlattened = Map[
		First@Ordering[#, -1] &,
		Flatten[heatmaps, {{1}, {2}, {3, 4}}],
		{2}
	];
	posMat = QuotientRemainder[posFlattened - 1, 64] + 1;
	Map[
		Image`TransformPixelCoordinates[#, {64, 64},
			"RawIndices"->"GraphicsCoordinates"]&,
		posMat
	] / 64.
]



(* Age *)

ClearAll[ageDistribution]
SetAttributes[ageDistribution, Listable]
ageDistribution[image_, property_: Automatic] :=
Scope[
	distribution = Cached[SafeNetEvaluate[
		Classify[
			"FacialAge",
			{image, ImageReflect[image, Left]},
			"Probabilities",
			TrainingProgressReporting -> None
		],
		 netResult["NumericAssociation"]
		]
	];
	distribution = Merge[distribution, Mean];

	wd = WeightedData[Keys[distribution], Values[distribution]];

	Association[
		"Distribution" -> wd,
		"Mean" -> Round[Mean[wd]],
		"StandardDeviation" -> StandardDeviation[wd]
	]
]

getAgeWithConfidence[age_, confidence_] :=
Scope[
	best = age;
	n = 0;
	While[Max[best] < confidence,
		best = AssociationThread[
		Mean  /@ Partition[Keys[age], ++n, 1],
		Total /@ Partition[Values[age], n, 1]
		]
	];

	PlusMinus[
		Round[First[Keys[MaximalBy[best,Identity]]]],
		Round[n/2]
	]
]

