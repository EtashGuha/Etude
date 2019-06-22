(* ::Package:: *)

BeginPackage["OpenCVLink`"]
Begin["`Private`"]

$InitOpenCVLink = False;

(* This variable are saved for efficiency *)
(* Each call to FindLibrary takes roughly 0.015s *)
$OpenCVLinkLibrary := $OpenCVLinkLibrary = FindLibrary["OpenCVLink"];
$OpenCVLinkBaseDirectory := $OpenCVLinkBaseDirectory = FileNameDrop[$InputFileName, -2];
$BaseLibraryDirectory := $BaseLibraryDirectory = FileNameJoin[{$OpenCVLinkBaseDirectory, "LibraryResources", $SystemID}];

dlls["Windows"|"Windows-x86-64"] = Join[{"tbb"}, # <> "343" & /@
				{"opencv_core", "opencv_flann", "opencv_imgproc",
				"opencv_features2d", "opencv_imgcodecs", "opencv_videoio",
				"opencv_highgui", "opencv_ml", "opencv_objdetect",
				"opencv_photo","opencv_video", "opencv_shape",
				"opencv_calib3d","opencv_stitching", "opencv_superres",
				"opencv_videostab", "opencv_dnn"}];

dlls["MacOSX-x86-64" | "MacOSX-x86"] = "lib" <> #  <> ".dylib"& /@
				{"opencv_core", "opencv_flann", "opencv_imgproc", "opencv_imgcodecs",
				"opencv_videoio", "opencv_highgui", "opencv_features2d", "opencv_ml",
				"opencv_objdetect", "opencv_photo", "opencv_video", "opencv_shape",
				"opencv_calib3d", "opencv_stitching", "opencv_superres",
				"opencv_videostab", "opencv_dnn"};

dlls["Linux"|"Linux-x86-64"|"Linux-ARM"] = "lib" <> # <> ".so"& /@
				{"opencv_core", "opencv_flann", "opencv_imgproc",
				"opencv_ml","opencv_photo","opencv_imgcodecs",
				"opencv_video","opencv_videoio","opencv_highgui",
				"opencv_shape", "opencv_superres", "opencv_features2d",
				"opencv_objdetect", "opencv_calib3d", "opencv_stitching",
				"opencv_videostab", "opencv_dnn"};

dlls[___] := $Failed;

opencvfunctions =
Hold[
	{$Dilation, "opencv_Dilation", {{"Image", "Constant"}, Integer}, {"Image"}},
	{$MedianFilter, "opencv_MedianFilter", {{"Image", "Constant"}, Integer}, {"Image"}},
	{$ImagePad, "opencv_ImagePad", {{"Image", "Constant"}, Integer, Integer, Integer, Integer, "UTF8String", {_Real, 1}}, {"Image"}},
	{$ImagePerspectiveTransformation, "opencv_ImagePerspectiveTransformation", {{"Image", "Constant"}, {_Real, 2}, "UTF8String", "UTF8String", {_Real, 1}}, {"Image"}},
	{$ImageConvolve, "opencv_ImageConvolve", {{"Image", "Constant"}, {_Real, 1}, {_Real, 1}, "UTF8String"}, {"Image"}},
	{$LocalAdaptiveBinarize, "opencv_LocalAdaptiveBinarize", {{"Image", "Constant"}, Integer, {Real, 1}, "UTF8String", Real}, {"Image"}},
	{$SuperResolution, "opencv_SuperResolution", {"UTF8String", "UTF8String", Integer}, {"Image"}},
	{$PyrDown, "opencv_PyrDown", {{"Image", "Constant"}, "UTF8String", _Integer, _Integer}, {"Image"}},
	{$PyrUp, "opencv_PyrUp", {{"Image", "Constant"}, _Integer, _Integer}, {"Image"}},
	{$HoughLines, "opencv_HoughLines", {{"Image", "Constant"}, _Real, _Real, _Integer, _Real, _Real, "UTF8String"}, {_Real, 1}},
	{$HoughLinesP, "opencv_HoughLinesP", { {"Image", "Constant"}, _Real, _Real, _Integer, _Real, _Real}, {_Real, 1}},
	{$HoughCircles, "opencv_HoughCircles", {{"Image", "Constant"}, _Real, _Real, _Real, _Real, _Integer, _Integer}, {_Real, 1}},
	{$LineSegmentDetector, "opencv_LineSegmentDetector", {{"Image", "Constant"}}, {_Real, 1}},
	{$PhaseCorrelate, "opencv_PhaseCorrelate", {{"Image", "Constant"}, {"Image", "Constant"}, {_Real, _, "Constant"}}, {_Real, 1}},
	{$PyrMeanShiftFilter, "opencv_PyrMeanShiftFilter", {{"Image", "Constant"}, _Real, _Real, _Integer, _Integer}, {"Image"}},
	{$FastNlMeansDenoising, "opencv_FastNlMeansDenoising", {{"Image", "Constant"}, Real, Integer, Integer}, {"Image"}},
	{$FastNlMeansDenoisingColored, "opencv_FastNlMeansDenoisingColored", {{"Image", "Constant"}, Real, Real, Integer, Integer}, {"Image"}},
	{$FastNlMeansDenoisingMulti, "opencv_FastNlMeansDenoisingMulti", {{"Image3D", "Constant"}, Integer, Integer, Real, Integer, Integer}, {"Image"}},
	{$FastNlMeansDenoisingColoredMulti, "opencv_FastNlMeansDenoisingColoredMulti", {{"Image3D", "Constant"}, Integer, Integer, Real, Real, Integer, Integer}, {"Image"}},
	{$MergeMertens, "opencv_MergeMertens", {{"Image3D", "Constant"}}, {"Image"}},
	{$Decolor, "opencv_Decolor", {{"Image", "Constant"}}, {"Image"}},
	{$ColorBoost, "opencv_Color_Boost", {{"Image", "Constant"}}, {"Image"}},
	{$EarthMoverDistanceFlowMatrix, "opencv_EarthMoverDistanceFlowMatrix", {{Real, 2}, {Real, 1}, {Real, 2}, {Real, 1}, String, {Real, 2}}, {Real, 2}},
	{$EarthMoverDistance, "opencv_EarthMoverDistance", {{Real, 2}, {Real, 1}, {Real, 2}, {Real, 1}, String, {Real, 2}}, Real},
	{$EarthMoverDistanceWeightsFlowMatrix, "opencv_EarthMoverDistanceWeightsFlowMatrix", {{Real, 1}, {Real, 1}, {Real, 2}}, {Real, 2}},
	{$ImageCorrelate, "opencv_ImageCorrelate", {{"Image", "Constant"}, {"Image", "Constant"}, "UTF8String", "UTF8String", Real}, {"Image"}},
	{$OpticalFlowFarneback, "opencv_OpticalFlowFarneback", {{"Image3D", "Constant"}, Real, Integer, Real, Integer, Integer, Real, "UTF8String", "Boolean", {Real, 3}, Real}, {Real, 4}},
	{$OpticalFlowDualTVL1, "opencv_OpticalFlowDualTVL1", {{"Image3D", "Constant"}, Real, Integer, Real, Integer, Real, Real, Integer, "Boolean", {Real, 3}, Real}, {Real, 4}},
	{$Stylization, "opencv_Stylization", {{"Image", "Constant"}, _Real, _Real}, {"Image"}},
	{$EdgePreservingFilter, "opencv_EdgePreservingFilter", {{"Image", "Constant"}, "UTF8String", _Real, _Real}, {"Image"}},
	{$DetailEnhance, "opencv_DetailEnhance", {{"Image", "Constant"}, _Real, _Real}, {"Image"}},
	{$ImageDemosaic, "opencv_imageDemosaic", {{"Image", "Constant"}, "UTF8String"}, {Image}},
	{$MergeRobertson, "opencv_MergeRobertson", {{"Image3D", "Constant"}, {Real, 1}, {Real, 1}}, {"Image"}},
	{$CalibrateRobertson, "opencv_CalibrateRobertson", {{"Image3D", "Constant"}, {Real, 1}, Integer, Real}, {Real, 1}},
	{$RemapTensor, "opencv_RemapTensor", {{"Image", "Constant"}, "UTF8String", "UTF8String", {Real, 1}, {Real, 2}, {Real, 2}}, {"Image"}},
	{$LogPolarCV, "opencv_LogPolarCV", {{"Image", "Constant"}, {Real, 1}, Real, "UTF8String", "Boolean", "Boolean"}, {"Image"}},
	{$LogPolarRemap, "opencv_LogPolarRemap", {{"Image", "Constant"}, {Real, 1}, Real, "UTF8String", "UTF8String", {Real, 1}, "Boolean"}, {"Image"}},
	{$Tonemap, "opencv_Tonemap", {{"Image", "Constant"}, String, {Real, 1}}, {"Image"}},
	{$MergeDebevec, "opencv_MergeDebevec", {{"Image3D", "Constant"}, {Real, 1}, {Real, 1}}, {"Image"}},
	{$CalibrateDebevec, "opencv_CalibrateDebevec", {{"Image3D", "Constant"}, {Real, 1}, Integer, Real, "Boolean"}, {Real, 1}},
	{$AlignMTB, "opencv_AlignMTB", {{"Image3D", "Constant"}, Integer, Integer}, {"Image3D"}},
	{$CalibrateDebevec, "opencv_CalibrateDebevec", {{"Image3D", "Constant"}, {Real, 1}, Integer, Real, "Boolean"}, {Real, 1}},
	{$LoadImagesFromPathInto, "opencv_LoadImagesFromPathInto", {{"RawArray", "Shared"}, "UTF8String", {Integer, 1, "Constant"}, "UTF8String", Integer, {"RawArray", "Constant"}, Integer, {"RawArray", "Constant"}, Integer, True|False}, {Integer, 1}},
	{$Fast, "opencv_Fast", {{"Image", "Constant"}, "Boolean", {"Image", "Constant"}, Integer, "Boolean"}, {_Real, 2}},
	{$Agast, "opencv_Agast", {{"Image", "Constant"}, "Boolean", {"Image", "Constant"}, Integer, "Boolean"}, {_Real, 2}},
	{$Brisk, "opencv_Brisk", {{"Image", "Constant"}, "Boolean", {"Image", "Constant"}, "UTF8String", Integer, Integer, Real, {Real, 2}}, {_Real, 2}},
	{$Orb, "opencv_Orb", {{"Image", "Constant"}, "Boolean", {"Image", "Constant"}, "UTF8String", Integer, Integer, Real, Integer, Integer, Integer, "UTF8String", {Real, 2}}, {_Real, 2}},
	{$Kaze, "opencv_Kaze", {{"Image", "Constant"}, "Boolean", {"Image", "Constant"}, "UTF8String", _Real, _Integer, _Integer, "UTF8String", "Boolean", "Boolean", {_Real, 2}}, {_Real, 2}},
	{$Akaze, "opencv_Akaze", {{"Image", "Constant"}, "Boolean", {"Image", "Constant"}, "UTF8String", _Real, _Integer, _Integer, "UTF8String", "UTF8String", _Integer, _Integer, {_Real, 2}}, {_Real, 2}},
	{$DescriptorMatcher, "opencv_DescriptorMatcher", {{_Real, 2}, "Boolean", {_Integer, 2}, {_Real, 2}, "UTF8String", "UTF8String", _Real}, {_Integer, 1}},
	{$MserDetect, "opencv_MserDetect", {{"Image", "Constant"}, Integer, Integer, Integer, Real, Real, Integer, Real, Real, Integer, "UTF8String" }, {Real, 2}},
	{$MatchShapes, "opencv_MatchShapes", {{Real, 2}, {Real, 2}, "UTF8String"}, _Real},
	{$FindContours, "opencv_FindContours", {{"Image", "Constant"}, {"UTF8String"}, {"UTF8String"}, "Boolean"}, {Integer, 1}},
	{$FindFacesMultiScaleCascade, "opencv_FaceDetection", {{"Image", "Constant"}, Integer, Integer, Integer, Integer, "UTF8String", Integer, Real, "Boolean", Integer}, {_Real, 1}},
	{$FindFacesDNN, "opencv_FaceDetectionDNN", {{"Image","Constant"},  "UTF8String", "UTF8String", Real, "Boolean",Integer}, {_Real, 1}}
];

safeLibraryLoad[debug_, lib_] :=
Block[
	{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},

	Quiet[
		Check[
			LibraryLoad[lib],
			If[TrueQ[debug],
				Print["Failed to load ", lib]
			];
			Throw[$InitOpenCVLink = $Failed, "OpenCVLinkFailure"]
		]
	]
]
safeLibraryFunctionLoad[debug_, args___] :=
Block[
	{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},
	Quiet[
		Check[
			LibraryFunctionLoad[$OpenCVLinkLibrary, args],
			If[TrueQ[debug],
				Print["Failed to load the function ", First[{args}], " from ", $OpenCVLinkLibrary]
			];
			Throw[$InitOpenCVLink = $Failed, "OpenCVLinkFailure"]
		]
	]
]

SetAttributes[define, HoldAll];
define[symbol_, fun_, argin_, argout_] :=
If[
	TrueQ[debugQ],
	(* immediate assignment with debug messages *)
	symbol = safeLibraryFunctionLoad[True, fun, argin, argout]
	,
	(* deleyed assignment with no messages *)
	symbol := symbol = Catch[safeLibraryFunctionLoad[False, fun, argin, argout], "OpenCVLinkFailure"]
]

InitOpenCVLink[debug: (True | False) : False] :=
If[
	TrueQ[$InitOpenCVLink],
	$InitOpenCVLink
	,
	$InitOpenCVLink = Catch[
		If[
			dlls[$SystemID] === $Failed,
			Message[OpenCVLink::sys, "Incompatible SystemID"];
			Throw[$Failed, "OpenCVLinkFailure"]
		];
		Block[
			{debugQ = TrueQ[debug]},
			safeLibraryLoad[debug, #]& /@ Flatten[{dlls[$SystemID], $OpenCVLinkLibrary}];
			ReleaseHold[define @@@ opencvfunctions];
			True
		],
		"OpenCVLinkFailure"
	]
];

InitOpenCVLink[];

End[]
EndPackage[]

