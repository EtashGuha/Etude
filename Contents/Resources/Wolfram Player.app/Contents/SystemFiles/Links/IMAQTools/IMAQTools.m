(* ::Package:: *)

(* Created by the Wolfram Workbench Sep 27, 2013 *)

BeginPackage["IMAQTools`"]
(* Exported symbols added here with SymbolName::usage *) 

(* Device Management *)
IMAQTools`DevMgr`ListDevices::usage = "IMAQTools`DevMgr`ListDevices[] discovers cameras attached to the system.";
IMAQTools`DevMgr`GetConnectedDevices::usage = "IMAQTools`DevMgr`GetConnectedDevices[] lists the cameras currently connected to IMAQTools framework.";

(* Device Control and Configuration *)
IMAQTools`OmniStream`Open::usage = "IMAQTools`OmniStream`Open[index, frameRate., width, height, captureCallback] opens the camera corresponding to the nonnegative integer index provided. captureCallback should be a function of the form f[asyncObj_, eventType_, eventData_], or None.";
IMAQTools`OmniStream`Start::usage = "IMAQTools`OmniStream`Start[] starts the currently opened camera.";
IMAQTools`OmniStream`Stop::usage = "IMAQTools`OmniStream`Stop[] stops the currently opened camera.";
IMAQTools`OmniStream`Close::usage = "IMAQTools`OmniStream`Close[] closes the currently opened camera.";
IMAQTools`OmniStream`SetResolution::usage = "IMAQTools`OmniStream`SetResolution[width, height] sets the resolution of the currently opened camera.";
IMAQTools`OmniStream`GetResolution::usage = "IMAQTools`OmniStream`GetResolution[] returns the resolution {width, height} of the currently opened camera.";
IMAQTools`OmniStream`GetSupportedResolutions::usage = "IMAQTools`OmniStream`GetSupportedResolutions[] returns a list of the available native resolutions of the currently opened camera.";
IMAQTools`OmniStream`SetSoftFrameRate::usage = "IMAQTools`OmniStream`SetSoftFrameRate[frameRate.] sets the rate at which Images will be buffered and/or returned asynchronously for the currently opened camera (does not affect the hardware FrameRate).";
IMAQTools`OmniStream`GetSoftFrameRate::usage = "IMAQTools`OmniStream`GetSoftFrameRate[] returns the rate at which Images will be buffered and/or returned asynchronously for the currently opened camera.";
IMAQTools`OmniStream`UpdateFrontend::usage = "IMAQTools`OmniStream`UpdateFrontend[] should be called by any Dynamics consuming the asynchronous image stream to prevent the stream from going idle.";
IMAQTools`OmniStream`KeepAlive::usage = "IMAQTools`OmniStream`KeepAlive[] prevents the currently opened camera from automatically closing after the timeout period returned by IMAQTools`OmniStream`GetShutoffTime.";
(* acquire images *)
IMAQTools`OmniStream`GrabFrame::usage = "IMAQTools`OmniStream`GrabFrame[] returns the latest Image from the currently opened camera.";
IMAQTools`OmniStream`GrabNFrames::usage = "IMAQTools`OmniStream`GrabNFrames[n] returns the n latest Images from the currently opened camera.";
(* throttling configuration *)
IMAQTools`OmniStream`SetUseThrottling::usage = "IMAQTools`OmniStream`SetUseThrottling[True|False] sets whether throttling functionality is enabled.";
IMAQTools`OmniStream`GetUseThrottling::usage = "IMAQTools`OmniStream`GetUseThrottling[] returns whether throttling functionality is enabled.";
IMAQTools`OmniStream`SetThrottlingMethod::usage = "IMAQTools`OmniStream`SetThrottlingMethod[method] selects the throttling algorithm to use corresponding to the nonnegative integer index provided.";
IMAQTools`OmniStream`GetThrottlingMethod::usage = "IMAQTools`OmniStream`GetThrottlingMethod[] returns the currently selected throttling method.";
IMAQTools`OmniStream`ThrottlingParams::usage = "IMAQTools`OmniStream`ThrottlingParams[] lists the available throttling parameters.\nIMAQTools`OmniStream`ThrottlingParams[param] returns the value of specified parameter.\nIMAQTools`OmniStream`ThrottlingParams[param->value] sets the value of the specified parameter.";
IMAQTools`OmniStream`SetUseFrontendFpsExp::usage = "IMAQTools`OmniStream`SetUseFrontendFpsExp[True|False] selects whether to use experimental method to calculate frontend update rate (i.e., frequency of calls to IMAQTools`OmniStream`UpdateFrontend).";
IMAQTools`OmniStream`GetUseFrontendFpsExp::usage = "IMAQTools`OmniStream`GetUseFrontendFpsExp[] returns whether experimental method is used to calculate frontend update rate (i.e., frequency of calls to IMAQTools`OmniStream`UpdateFrontend).";
IMAQTools`OmniStream`SetPrintStatStream::usage = "IMAQTools`OmniStream`SetPrintStatStream[True|False] selects whether throttling statistics will be printed while camera is running. FEATURE IS CURRENTLY DISABLED DUE TO MSVS BUILD CONFLICT.";
IMAQTools`OmniStream`GetPrintStatStream::usage = "IMAQTools`OmniStream`GetPrintStatStream[] returns whether throttling statistics will be printed while camera is running. FEATURE IS CURRENTLY DISABLED DUE TO MSVS BUILD CONFLICT.";
(* image buffering configuration *)
IMAQTools`OmniStream`SetMaxBufferSize::usage = "IMAQTools`OmniStream`SetMaxBufferSize[maxBufferSize] sets the maximum number of images buffered for the currently opened camera.";
IMAQTools`OmniStream`GetMaxBufferSize::usage = "IMAQTools`OmniStream`GetMaxBufferSize[] returns the maximum number of images buffered for the currently opened camera.";
IMAQTools`OmniStream`GetGrabFrameTimeout::usage = "IMAQTools`OmniStream`GetGrabFrameTimeout[]";
IMAQTools`OmniStream`SetGrabFrameTimeout::usage = "IMAQTools`OmniStream`SetGrabFrameTimeout[timeout.]";
IMAQTools`OmniStream`SetBufferMode::usage = "IMAQTools`OmniStream`SetBufferMode[mode] sets a buffering function to be applied to the images before they are buffered.";
IMAQTools`OmniStream`GetBufferMode::usage = "IMAQTools`OmniStream`GetBufferMode[] returns the current buffering mode.";
IMAQTools`OmniStream`GetBufferModes::usage = "IMAQTools`OmniStream`GetBufferModes[] returns a list of modes which can be passed to IMAQTools`OmniStream`SetBufferMode, in addition to arbitrary functions.";
(* autoShutoff configuration *)
IMAQTools`OmniStream`SetShutoffTime::usage = "IMAQTools`OmniStream`SetShutoffTime[timeout] sets the period after which the currently opened camera will automatically close, to the integer number of seconds provided. A value of 0 disables the AutoShtuoff feature.";
IMAQTools`OmniStream`GetShutoffTime::usage = "IMAQTools`OmniStream`GetShutoffTime[] returns the period after which the currently opened camera will automatically close.";
(* configuration for handling poorly-exposed images *)
IMAQTools`OmniStream`AutoExposureParams::usage = "IMAQTools`OmniStream`AutoExposureParams[] lists the available parameters for handling images that are insufficiently exposed due to AutoExposure-hunting by some devices.\nIMAQTools`OmniStream`AutoExposureParams[param] returns the value of specified parameter.\nIMAQTools`OmniStream`AutoExposureParams[param->value] sets the value of the specified parameter.";


IMAQTools::grabframetimeout = "Timed out acquiring image for device with ID `1`";
IMAQTools::noformats = "Unable to get supported resolutions for device with ID `1`.";
IMAQTools::nodevice = "Device with ID `1` is not connected.";

Begin["`Private`"]
(* Implementation of the package *)

$pacletVersion = 0.03;
$packageFile = $InputFileName;

$libName = Switch[ $OperatingSystem,
	"MacOSX", "IMAQTools.dylib",
	"Windows", "IMAQTools.dll",
	"Unix", "IMAQTools.so",
	_, $Failed
];

$adapterLib = FileNameJoin[{FileNameTake[$packageFile, {1,-2}], "LibraryResources", $SystemID, $libName}];

$adapterInitialized;

loadAdapter[]:= If[ !ValueQ[$adapterInitialized],
	If[ !$CloudEvaluation || TrueQ[CloudSystem`FeatureEnabledQ["EnableServerImagingDevice"]],
	ulfLoadAdapter2 = LibraryFunctionLoad[ $adapterLib, "LoadAdapter2", {}, "Void"];
	lfLoadAdapter2[args___] :=  PreemptProtect[ulfLoadAdapter2[ args]];
	ulfGetAvailableDevices2 = LibraryFunctionLoad[ $adapterLib, "GetAvailableDevices2", LinkObject, LinkObject];
	lfGetAvailableDevices2[args___] :=  PreemptProtect[ulfGetAvailableDevices2[ args]];
	ulfGetConnectedDevices2 = LibraryFunctionLoad[$adapterLib, "GetConnectedDevices2", LinkObject, LinkObject];
	lfGetConnectedDevices2[args___] :=  PreemptProtect[ulfGetConnectedDevices2[ args]];
	ulfLayeredCameraConnect = LibraryFunctionLoad[$adapterLib, "ConnectDevice2", {Integer}, Integer];
	lfLayeredCameraConnect[args___] :=  PreemptProtect[ulfLayeredCameraConnect[ args]];
	ulfLayeredCameraDisconnect = LibraryFunctionLoad[$adapterLib, "DisconnectDevice2", {Integer}, Integer];
	lfLayeredCameraDisconnect[args___] :=  PreemptProtect[ulfLayeredCameraDisconnect[ args]];

	ulfOmniCameraOpen = LibraryFunctionLoad[$adapterLib, "OmniCameraOpen", {Integer,Real,Integer,Integer}, Integer];
	lfOmniCameraOpen[args___] :=  PreemptProtect[ulfOmniCameraOpen[ args]];
	ulfOmniCameraOpenWithAutoshutoff = LibraryFunctionLoad[$adapterLib, "OmniCameraOpenWithAutoshutoff", {Integer,Real,Integer,Integer,Integer}, Integer];
	lfOmniCameraOpenWithAutoshutoff[args___] :=  PreemptProtect[ulfOmniCameraOpenWithAutoshutoff[ args]];
	ulfOmniCameraStart = LibraryFunctionLoad[$adapterLib, "OmniCameraStart", {Integer}, Integer];
	lfOmniCameraStart[args___] :=  PreemptProtect[ulfOmniCameraStart[ args]];
	ulfOmniCameraStop = LibraryFunctionLoad[$adapterLib, "OmniCameraStop", {Integer}, Integer];
	lfOmniCameraStop[args___] :=  PreemptProtect[ulfOmniCameraStop[ args]];
	ulfOmniCameraClose = LibraryFunctionLoad[$adapterLib, "OmniCameraClose", {Integer}, Integer];
	lfOmniCameraClose[args___] :=  PreemptProtect[ulfOmniCameraClose[ args]];
	ulfOmniCameraSetUseThrottling = LibraryFunctionLoad[$adapterLib, "OmniCameraSetUseThrottling", {Integer,"Boolean"}, Integer];
	lfOmniCameraSetUseThrottling[args___] :=  PreemptProtect[ulfOmniCameraSetUseThrottling[ args]];
	ulfOmniCameraSetThrottlingMethod = LibraryFunctionLoad[$adapterLib, "OmniCameraSetThrottlingMethod", {Integer,Integer}, Integer];
	lfOmniCameraSetThrottlingMethod[args___] :=  PreemptProtect[ulfOmniCameraSetThrottlingMethod[ args]];
	ulfOmniCameraSetThrottlingParam = LibraryFunctionLoad[$adapterLib, "OmniCameraSetThrottlingParam", {Integer,Integer,Real}, Integer];
	lfOmniCameraSetThrottlingParam[args___] :=  PreemptProtect[ulfOmniCameraSetThrottlingParam[ args]];
	ulfOmniCameraGetThrottlingParam = LibraryFunctionLoad[$adapterLib, "OmniCameraGetThrottlingParam", {Integer,Integer}, Real];
	lfOmniCameraGetThrottlingParam[args___] :=  PreemptProtect[ulfOmniCameraGetThrottlingParam[ args]];
	ulfOmniCameraUpdateFrontendInterval = LibraryFunctionLoad[$adapterLib, "OmniCameraUpdateFrontendInterval", {Integer,Integer}, Integer];
	lfOmniCameraUpdateFrontendInterval[args___] :=  PreemptProtect[ulfOmniCameraUpdateFrontendInterval[ args]];
	ulfOmniCameraSetResolution = LibraryFunctionLoad[$adapterLib, "OmniCameraSetResolution", {Integer,Integer,Integer}, Integer];
	lfOmniCameraSetResolution[args___] :=  PreemptProtect[ulfOmniCameraSetResolution[ args]];
	ulfOmniCameraGetResolution = LibraryFunctionLoad[$adapterLib, "OmniCameraGetResolution", {Integer}, {Integer,_}];
	lfOmniCameraGetResolution[args___] :=  PreemptProtect[ulfOmniCameraGetResolution[ args]];
	ulfOmniCameraGetSupportedResolutions = LibraryFunctionLoad[$adapterLib, "OmniCameraGetSupportedResolutions", {Integer}, {Integer,_}];
	lfOmniCameraGetSupportedResolutions[args___] :=  PreemptProtect[ulfOmniCameraGetSupportedResolutions[ args]];
	ulfOmniCameraSetMaxBufferSize = LibraryFunctionLoad[$adapterLib, "OmniCameraSetMaxBufferSize", {Integer,Integer}, Integer];
	lfOmniCameraSetMaxBufferSize[args___] :=  PreemptProtect[ulfOmniCameraSetMaxBufferSize[ args]];
	ulfOmniCameraGetMaxBufferSize = LibraryFunctionLoad[$adapterLib, "OmniCameraGetMaxBufferSize", {Integer}, Integer];
	lfOmniCameraGetMaxBufferSize[args___] :=  PreemptProtect[ulfOmniCameraGetMaxBufferSize[ args]];
	ulfOmniCameraGrabFrame = LibraryFunctionLoad[$adapterLib, "OmniCameraGrabFrame", {Integer}, {Integer, _}];
	lfOmniCameraGrabFrame[args___] :=  PreemptProtect[ulfOmniCameraGrabFrame[ args]];
	ulfOmniCameraGrabAutoExposedFrame = LibraryFunctionLoad[$adapterLib, "OmniCameraGrabAutoExposedFrame", {Integer}, {Integer, _}];
	lfOmniCameraGrabAutoExposedFrame[args___] :=  PreemptProtect[ulfOmniCameraGrabAutoExposedFrame[ args]];
	ulfOmniCameraKeepAlive = LibraryFunctionLoad[$adapterLib, "OmniCameraKeepAlive", {Integer}, Integer];
	lfOmniCameraKeepAlive[args___] :=  PreemptProtect[ulfOmniCameraKeepAlive[ args]];
	ulfOmniCameraSetShutoffTime = LibraryFunctionLoad[$adapterLib, "OmniCameraSetShutoffTime", {Integer,Integer}, Integer];
	lfOmniCameraSetShutoffTime[args___] :=  PreemptProtect[ulfOmniCameraSetShutoffTime[ args]];
	ulfOmniCameraGetShutoffTime = LibraryFunctionLoad[$adapterLib, "OmniCameraGetShutoffTime", {Integer}, Integer];
	lfOmniCameraGetShutoffTime[args___] :=  PreemptProtect[ulfOmniCameraGetShutoffTime[ args]];
	ulfOmniCameraUpdateFrontendIntervalExp = LibraryFunctionLoad[$adapterLib, "OmniCameraUpdateFrontendIntervalExp", {Integer}, Integer];
	lfOmniCameraUpdateFrontendIntervalExp[args___] :=  PreemptProtect[ulfOmniCameraUpdateFrontendIntervalExp[ args]];
	ulfOmniCameraSetUseFrontendFpsExp = LibraryFunctionLoad[$adapterLib, "OmniCameraSetUseFrontendFpsExp", {Integer,"Boolean"}, Integer];
	lfOmniCameraSetUseFrontendFpsExp[args___] :=  PreemptProtect[ulfOmniCameraSetUseFrontendFpsExp[ args]];
	ulfOmniCameraGrabNFrames = LibraryFunctionLoad[ $adapterLib, "OmniCameraGrabNFrames", {Integer, Integer}, {Integer, _}];
	lfOmniCameraGrabNFrames[args___] :=  PreemptProtect[ulfOmniCameraGrabNFrames[ args]];
	ulfOmniCameraGrabAutoExposedNFrames = LibraryFunctionLoad[ $adapterLib, "OmniCameraGrabAutoExposedNFrames", {Integer, Integer}, {Integer, _}];
	lfOmniCameraGrabAutoExposedNFrames[args___] :=  PreemptProtect[ulfOmniCameraGrabAutoExposedNFrames[ args]];
	ulfOmniCameraRetrieveImage = LibraryFunctionLoad[ $adapterLib, "OmniCameraRetrieveImage", {Integer}, Image];
	lfOmniCameraRetrieveImage[args___] :=  PreemptProtect[ulfOmniCameraRetrieveImage[ args]];
	ulfOmniCameraRetrieveMetaInformation = LibraryFunctionLoad[ $adapterLib, "OmniCameraRetrieveMetaInformation", {Integer}, "RawArray"];
	lfOmniCameraRetrieveMetaInformation[args___] :=  PreemptProtect[ulfOmniCameraRetrieveMetaInformation[ args]];
	ulfOmniCameraSetPrintStatStream = LibraryFunctionLoad[$adapterLib, "OmniCameraSetPrintStatStream", {Integer,"Boolean"}, Integer];
	lfOmniCameraSetPrintStatStream[args___] :=  PreemptProtect[ulfOmniCameraSetPrintStatStream[ args]];
	ulfOmniCameraSetAutoExposeMaxIterations = LibraryFunctionLoad[$adapterLib, "OmniCameraSetAutoExposeMaxIterations", {Integer,Integer}, Integer];
	lfOmniCameraSetAutoExposeMaxIterations[args___] :=  PreemptProtect[ulfOmniCameraSetAutoExposeMaxIterations[ args]];
	ulfOmniCameraGetAutoExposeMaxIterations = LibraryFunctionLoad[$adapterLib, "OmniCameraGetAutoExposeMaxIterations", {Integer}, Integer];
	lfOmniCameraGetAutoExposeMaxIterations[args___] :=  PreemptProtect[ulfOmniCameraGetAutoExposeMaxIterations[ args]];
	ulfOmniCameraSetAutoExposeNumStableFrames = LibraryFunctionLoad[$adapterLib, "OmniCameraSetAutoExposeNumStableFrames", {Integer,Integer}, Integer];
	lfOmniCameraSetAutoExposeNumStableFrames[args___] :=  PreemptProtect[ulfOmniCameraSetAutoExposeNumStableFrames[ args]];
	ulfOmniCameraGetAutoExposeNumStableFrames = LibraryFunctionLoad[$adapterLib, "OmniCameraGetAutoExposeNumStableFrames", {Integer}, Integer];
	lfOmniCameraGetAutoExposeNumStableFrames[args___] :=  PreemptProtect[ulfOmniCameraGetAutoExposeNumStableFrames[ args]];
	ulfOmniCameraSetAutoExposeImageMeanThreshold = LibraryFunctionLoad[$adapterLib, "OmniCameraSetAutoExposeImageMeanThreshold", {Integer,Real}, Integer];
	lfOmniCameraSetAutoExposeImageMeanThreshold[args___] :=  PreemptProtect[ulfOmniCameraSetAutoExposeImageMeanThreshold[ args]];
	ulfOmniCameraGetAutoExposeImageMeanThreshold = LibraryFunctionLoad[$adapterLib, "OmniCameraGetAutoExposeImageMeanThreshold", {Integer}, Real];
	lfOmniCameraGetAutoExposeImageMeanThreshold[args___] :=  PreemptProtect[ulfOmniCameraGetAutoExposeImageMeanThreshold[ args]];
	ulfOmniCameraSetAutoExposeImageMeanStabilityThreshold = LibraryFunctionLoad[$adapterLib, "OmniCameraSetAutoExposeImageMeanStabilityThreshold", {Integer,Real}, Integer];
	lfOmniCameraSetAutoExposeImageMeanStabilityThreshold[args___] :=  PreemptProtect[ulfOmniCameraSetAutoExposeImageMeanStabilityThreshold[ args]];
	ulfOmniCameraGetAutoExposeImageMeanStabilityThreshold = LibraryFunctionLoad[$adapterLib, "OmniCameraGetAutoExposeImageMeanStabilityThreshold", {Integer}, Real];
	lfOmniCameraGetAutoExposeImageMeanStabilityThreshold[args___] :=  PreemptProtect[ulfOmniCameraGetAutoExposeImageMeanStabilityThreshold[ args]];
	ulfOmniCameraSetAutoExposeTimeout = LibraryFunctionLoad[$adapterLib, "OmniCameraSetAutoExposeTimeout", {Integer,Real}, Integer];
	lfOmniCameraSetAutoExposeTimeout[args___] :=  PreemptProtect[ulfOmniCameraSetAutoExposeTimeout[ args]];
	ulfOmniCameraGetAutoExposeTimeout = LibraryFunctionLoad[$adapterLib, "OmniCameraGetAutoExposeTimeout", {Integer}, Real];
	lfOmniCameraGetAutoExposeTimeout[args___] :=  PreemptProtect[ulfOmniCameraGetAutoExposeTimeout[ args]];
	ulfOmniCameraGetUseThrottling = LibraryFunctionLoad[$adapterLib, "OmniCameraGetUseThrottling", {Integer}, "Boolean"];
	lfOmniCameraGetUseThrottling[args___] :=  PreemptProtect[ulfOmniCameraGetUseThrottling[ args]];
	ulfOmniCameraGetThrottlingMethod = LibraryFunctionLoad[$adapterLib, "OmniCameraGetThrottlingMethod", {Integer}, Integer];
	lfOmniCameraGetThrottlingMethod[args___] :=  PreemptProtect[ulfOmniCameraGetThrottlingMethod[ args]];
	ulfOmniCameraSetSoftFrameRate = LibraryFunctionLoad[$adapterLib, "OmniCameraSetSoftFrameRate", {Integer,Real}, Integer];
	lfOmniCameraSetSoftFrameRate[args___] :=  PreemptProtect[ulfOmniCameraSetSoftFrameRate[ args]];
	ulfOmniCameraGetSoftFrameRate = LibraryFunctionLoad[$adapterLib, "OmniCameraGetSoftFrameRate", {Integer}, Real];
	lfOmniCameraGetSoftFrameRate[args___] :=  PreemptProtect[ulfOmniCameraGetSoftFrameRate[ args]];
	ulfOmniCameraGetUseFrontendFpsExp = LibraryFunctionLoad[$adapterLib, "OmniCameraGetUseFrontendFpsExp", {Integer}, "Boolean"];
	lfOmniCameraGetUseFrontendFpsExp[args___] :=  PreemptProtect[ulfOmniCameraGetUseFrontendFpsExp[ args]];
	ulfOmniCameraGetPrintStatStream = LibraryFunctionLoad[$adapterLib, "OmniCameraGetPrintStatStream", {Integer}, "Boolean"];
	lfOmniCameraGetPrintStatStream[args___] :=  PreemptProtect[ulfOmniCameraGetPrintStatStream[ args]];
	ulfOmniCameraGetGrabFrameTimeout = LibraryFunctionLoad[$adapterLib, "OmniCameraGetGrabFrameTimeout", {Integer, "Boolean"}, Real];
	lfOmniCameraGetGrabFrameTimeout[args___] :=  PreemptProtect[ulfOmniCameraGetGrabFrameTimeout[ args]];
	ulfOmniCameraSetGrabFrameTimeout = LibraryFunctionLoad[$adapterLib, "OmniCameraSetGrabFrameTimeout", {Integer, Real, "Boolean"}, Integer];
	lfOmniCameraSetGrabFrameTimeout[args___] :=  PreemptProtect[ulfOmniCameraSetGrabFrameTimeout[ args]];
	ulfOmniCameraGetBufferMode = LibraryFunctionLoad[$adapterLib, "OmniCameraGetBufferMode", {Integer}, String];
	lfOmniCameraGetBufferMode[args___] := PreemptProtect[ulfOmniCameraGetBufferMode[ args]];
	ulfOmniCameraSetBufferMode = LibraryFunctionLoad[$adapterLib, "OmniCameraSetBufferMode", {Integer, String}, Integer];
	lfOmniCameraSetBufferMode[args___] := PreemptProtect[ulfOmniCameraSetBufferMode[ args]];
	lfOmniCameraAddWLBufferFunctionFrame = LibraryFunctionLoad[$adapterLib, "OmniCameraAddWLBufferFunctionFrame", {Integer, Image}, Integer];
	lfOmniCameraRemoveWLBufferFunction = LibraryFunctionLoad[$adapterLib, "OmniCameraRemoveWLBufferFunction", {Integer}, Integer];
	lfOmniCameraSetWLBufferFunction = LibraryFunctionLoad[$adapterLib, "OmniCameraSetWLBufferFunction", {Integer}, Integer];
	lfOmniCameraSetUseWLBufferFunction = LibraryFunctionLoad[$adapterLib, "OmniCameraSetUseWLBufferFunction", {Integer, "Boolean"}, Integer];
	lfOmniCameraGetUseWLBufferFunction = LibraryFunctionLoad[$adapterLib, "OmniCameraGetUseWLBufferFunction", {Integer}, "Boolean"];

	lfLoadAdapter2[];
	$adapterInitialized = True;
	,
	$adapterInitialized = False
	]
];

SetAttributes[tryload, HoldAll];
tryload[expr_] := (
	loadAdapter[];
	If[TrueQ[$adapterInitialized], expr, $Failed]
)

(* ::Section:: *)
(* Context Variables *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

$cells = Association[];
addCellKey[] := Module[{n},
	n = "CellSerialNumber"/.Developer`CellInformation@EvaluationCell[];
	If[!MemberQ[Keys[$cells],n],
		AssociateTo[$cells,n -> Length[$cells]];
	];
]
getCellKey[] := ($cells["CellSerialNumber" /. Developer`CellInformation@EvaluationCell[]] /. Except[x_Integer] -> -1)

$throttleParams = {
	"ERRSUM_SteadyStateCountThresh",
	"ERRSUM_FrontendFpsResetThresh",
	"ERRSUM_nErrorSamples",
	"ERRSUM_ThrottleIncr",
	"ERRSUM_SignsThresh",
	"ERRSUM_SteadyStateIncreaseIncr",
	"ERRSUM_FrontendTargetErrorThreshFactor",
	"ERRSUM_FrontendThrottledErrorThresh",
	"PID_SteadyStateCountThresh",
	"PID_FrontendFpsResetThresh",
	"PID_P",
	"PID_I",
	"PID_D",
	"PID_nI",
	"PID_FrontendDropBelowThresh",
	"PID_IncrSteadyStateErrThresh",
	"PID_SteadyStateIncreaseFactor",
	"PID_DiffStabilityCounterCountThresh",
	"PID_DiffStabilityCounterDiffThresh"
};
$throttleParamsAssoc = AssociationThread[#, Range[Length[#]]-1]& @ $throttleParams;

$autoExposureParams = {
	"MaxIterations",
	"NumStableFrames",
	"ImageMeanThreshold",
	"ImageMeanStabilityThreshold",
	"Timeout"
};
$autoExposureParamsAssoc = AssociationThread[#, Range[Length[#]]-1]& @ $autoExposureParams;

$omniBufferModes = Sort@{"LeftRightReversed", "TopBottomReversed", "Rotate90", "Rotate180", "Rotate270", "Default"};

enumDups[list_]:=Module[{dupIndices,mods,outList=list},
	dupIndices=Replace[DeleteCases[GatherBy[Partition[Riffle[list,Table[i,{i,Length@list}]],2],Part[#,1]&],_List?((Length[#]<=1)&)],{a_,b_}:>b,{2}];
	mods=Join@@(indexList/@dupIndices);
	(outList[[#[[1]]]]=list[[#[[1]]]]<>" ("<>ToString[#[[2]]]<>")")&/@mods;
	outList
]

$omniCamIndex;
Internal`SetValueNoTrack[$omniCamIndex, True];
$omniCamIndex = -1;

streamData = <|
	"AsyncObject" -> Null, 
	"UseFrontendFpsExp" -> True, 
	"UseThrottling" -> True, 
	"WLBufferFunctionAsyncObject" -> Null
|>;

$omniStreams = <||>;
Internal`SetValueNoTrack[$omniStreams, True];

addOmniAsyncObj[camIndex_Integer, obj_] := If[MemberQ[Keys[$omniStreams], camIndex], 
		$Failed
		, 
		AssociateTo[$omniStreams, camIndex -> streamData]; 
		$omniStreams[camIndex, "AsyncObject"] = obj;
	]
getOmniAsyncObj[camIndex_Integer] := If[MemberQ[Keys[$omniStreams], camIndex], 
		$omniStreams[camIndex, "AsyncObject"]
		, 
		streamData["AsyncObject"]
	]
removeOmniAsyncObj[camIndex_Integer] := If[MemberQ[Keys[$omniStreams], camIndex], 
		If[AsynchronousTaskObject === Head[#], Quiet@System`RemoveAsynchronousTask[#]]& @ $omniStreams[camIndex, "AsyncObject"]; 
		$omniStreams = Delete[$omniStreams, Key[camIndex]];
		,
		$Failed
	]

addOmniBufferFunAsyncObj[camIndex_Integer, obj_] := If[MemberQ[Keys[$omniStreams], camIndex],
		$omniStreams[camIndex, "WLBufferFunctionAsyncObject"] = obj;
		,
		$Failed
	]
getOmniBufferFunAsyncObj[camIndex_Integer] := If[MemberQ[Keys[$omniStreams], camIndex], 
		$omniStreams[camIndex, "WLBufferFunctionAsyncObject"]
		, 
		streamData["WLBufferFunctionAsyncObject"]
	]
removeOmniBufferFunAsyncObj[camIndex_Integer] := If[MemberQ[Keys[$omniStreams], camIndex], 
		If[AsynchronousTaskObject === Head[#], 
			omniStreamSetUseWLBufferFunction[camIndex, False];
			omniStreamRemoveWLBufferFunction[camIndex];
			Quiet@System`RemoveAsynchronousTask[#];
		]& @ $omniStreams[camIndex, "WLBufferFunctionAsyncObject"];
		$omniStreams[camIndex, "WLBufferFunctionAsyncObject"] = streamData["WLBufferFunctionAsyncObject"];
		,
		$Failed
	]

getOmniUseFrontendFpsExp[camIndex_Integer] := If[MemberQ[Keys[$omniStreams], camIndex], 
		$omniStreams[camIndex, "UseFrontendFpsExp"]
		, 
		streamData["UseFrontendFpsExp"]
	]
setOmniUseFrontendFpsExp[camIndex_Integer, useFrontendFpsExp:(True|False)] := If[MemberQ[Keys[$omniStreams], camIndex], 
		$omniStreams[camIndex, "UseFrontendFpsExp"] = useFrontendFpsExp;
		, 
		$Failed
	]

getOmniUseThrottling[camIndex_Integer] := If[MemberQ[Keys[$omniStreams], camIndex], 
		$omniStreams[camIndex, "UseThrottling"]
		, 
		streamData["UseThrottling"]
	]
setOmniUseThrottling[camIndex_Integer, useThrottling:(True|False)] := If[MemberQ[Keys[$omniStreams], camIndex], 
		$omniStreams[camIndex, "UseThrottling"] = useThrottling;
		, 
		$Failed
	]

IMAQTools`DevMgr`GetConnectedDevices[] := tryload@Module[{res},
	res = Quiet[lfGetConnectedDevices2[]];
	If[ListQ[res], res, $Failed]
]
IMAQTools`DevMgr`ListDevices[]:= tryload@(
	If[MemberQ[{"Windows", "MacOSX", "Unix"}, $OperatingSystem], 
		lfGetAvailableDevices2[]
		,
		$Failed
	]
)

(* ::Section:: *)
(* Device Control and Configuration *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

Options[IMAQTools`OmniStream`Open] = {"AutoShutoffTimeout" -> Infinity};

IMAQTools`OmniStream`Open[camIndex_Integer, captureCallback_, opts:OptionsPattern[]] := IMAQTools`OmniStream`Open[camIndex, 12., 320, 240, captureCallback, opts]

IMAQTools`OmniStream`Open[camIndex_Integer, frameRate_Real, width_Integer, height_Integer, captureCallback_, opts:OptionsPattern[]] :=
tryload@Module[{index, obj, shutoffTimeout},
	index = Quiet[lfLayeredCameraConnect[camIndex]] /. Except[_Integer] -> -1;
	If[index < 0, Return@$Failed];
	shutoffTimeout = OptionValue@"AutoShutoffTimeout" /. {Infinity -> 0, (x_Real /; x >= 0) -> Ceiling[x], Except[_Integer] -> 0, (x_ /; x < 0) -> 0};
	obj = If[shutoffTimeout == 0,
		Quiet@Internal`CreateAsynchronousTask[lfOmniCameraOpen, {index, frameRate, width, height}, captureCallback, "UserData" -> <|"StreamIndex" -> index|>]
		,
		Quiet@Internal`CreateAsynchronousTask[lfOmniCameraOpenWithAutoshutoff, {index, frameRate, width, height, shutoffTimeout}, captureCallback, "UserData" -> <|"StreamIndex" -> index|>]
	];
	If[obj === $Failed, IMAQTools`OmniStream`Close[index]; Return[$Failed]];
	addOmniAsyncObj[index, obj];
	$omniCamIndex = index;
	lfOmniCameraSetUseFrontendFpsExp[index, getOmniUseFrontendFpsExp[index]];
	lfOmniCameraSetUseThrottling[index, getOmniUseThrottling[index]];
	obj
]

IMAQTools`OmniStream`Start[] := IMAQTools`OmniStream`Start[$omniCamIndex]
IMAQTools`OmniStream`Start[camIndex_Integer] := tryload@Module[{res},
	res = lfOmniCameraStart[camIndex];
 	Return@If[res == 0, Null, $Failed]
 ]

IMAQTools`OmniStream`Stop[] := IMAQTools`OmniStream`Stop[$omniCamIndex]
IMAQTools`OmniStream`Stop[camIndex_Integer] := tryload@Module[{res},
	res = lfOmniCameraStop[camIndex];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`Close[] := IMAQTools`OmniStream`Close[$omniCamIndex]
IMAQTools`OmniStream`Close[camIndex_Integer] := tryload@Module[{res1, res2},
	removeOmniBufferFunAsyncObj[camIndex];
	res1 = lfOmniCameraClose[camIndex];
	res2 = lfLayeredCameraDisconnect[camIndex];
	removeOmniAsyncObj[camIndex];
	$omniCamIndex = If[Length[Keys[$omniStreams]] == 0, -1, First[Keys[$omniStreams]]];
	Return@If[res1 == 0 && res2 == 0, Null, $Failed, $Failed]
]

IMAQTools`OmniStream`SetUseThrottling[useThrottling:(True|False)] := IMAQTools`OmniStream`SetUseThrottling[$omniCamIndex, useThrottling]
IMAQTools`OmniStream`SetUseThrottling[camIndex_Integer, useThrottling:(True|False)] := tryload@Module[{res},
	res = lfOmniCameraSetUseThrottling[camIndex, useThrottling];
	Return@If[res == 0, setOmniUseThrottling[camIndex, useThrottling]; Null, $Failed]
]

IMAQTools`OmniStream`SetThrottlingMethod[method_Integer] := IMAQTools`OmniStream`SetThrottlingMethod[$omniCamIndex, method]
IMAQTools`OmniStream`SetThrottlingMethod[camIndex_Integer, method_Integer] := tryload@Module[{res},
	res = lfOmniCameraSetThrottlingMethod[camIndex, method];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`UpdateFrontend[] := IMAQTools`OmniStream`UpdateFrontend[$omniCamIndex]
IMAQTools`OmniStream`UpdateFrontend[camIndex_Integer] := tryload@Module[{res},
	res = If[!getOmniUseFrontendFpsExp[camIndex],
		addCellKey[];
		lfOmniCameraUpdateFrontendInterval[camIndex, getCellKey[]]
		,
		lfOmniCameraUpdateFrontendIntervalExp[camIndex]
	];
	
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`SetUseFrontendFpsExp[useFrontendFpsExp:(True|False)] := IMAQTools`OmniStream`SetUseFrontendFpsExp[$omniCamIndex, useFrontendFpsExp]
IMAQTools`OmniStream`SetUseFrontendFpsExp[camIndex_Integer, useFrontendFpsExp:(True|False)] := tryload@Module[{res},
	res = lfOmniCameraSetUseFrontendFpsExp[camIndex, useFrontendFpsExp];
	Return@If[res == 0, setOmniUseFrontendFpsExp[camIndex, useFrontendFpsExp]; Null, $Failed]
]

IMAQTools`OmniStream`SetResolution[width_Integer, height_Integer] := IMAQTools`OmniStream`SetResolution[$omniCamIndex, width, height]
IMAQTools`OmniStream`SetResolution[camIndex_Integer, width_Integer, height_Integer] := tryload@Module[{res},
	res = lfOmniCameraSetResolution[camIndex, width, height];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`GetResolution[] := IMAQTools`OmniStream`GetResolution[$omniCamIndex]
IMAQTools`OmniStream`GetResolution[camIndex_Integer] := tryload@Module[{res},
	res = Quiet[lfOmniCameraGetResolution[camIndex]];
	Return@If[ListQ[res], res, $Failed];
]

IMAQTools`OmniStream`GetSupportedResolutions[] := IMAQTools`OmniStream`GetSupportedResolutions[$omniCamIndex]
IMAQTools`OmniStream`GetSupportedResolutions[camIndex_Integer] := tryload@Module[{res},
	res = Quiet[lfOmniCameraGetSupportedResolutions[camIndex]];
	Return@If[ListQ[res], res, If[Head[res] === LibraryFunctionError, If[First[res] === "LIBRARY_NUMERICAL_ERROR", Message[IMAQTools::noformats, camIndex];, Message[IMAQTools::nodevice, camIndex];];]; $Failed]
]

IMAQTools`OmniStream`SetSoftFrameRate[frameRate_Real] := IMAQTools`OmniStream`SetSoftFrameRate[$omniCamIndex, frameRate]
IMAQTools`OmniStream`SetSoftFrameRate[camIndex_Integer, frameRate_Real] := tryload@Module[{res},
	res = lfOmniCameraSetSoftFrameRate[camIndex, frameRate];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`GetSoftFrameRate[] := IMAQTools`OmniStream`GetSoftFrameRate[$omniCamIndex]
IMAQTools`OmniStream`GetSoftFrameRate[camIndex_Integer] := tryload@Module[{res},
	Return@(Quiet@lfOmniCameraGetSoftFrameRate[camIndex] /. Except[_Real] -> $Failed)
]

IMAQTools`OmniStream`SetMaxBufferSize[maxBufferSize_Integer] := IMAQTools`OmniStream`SetMaxBufferSize[$omniCamIndex, maxBufferSize]
IMAQTools`OmniStream`SetMaxBufferSize[camIndex_Integer, maxBufferSize_Integer] := tryload@Module[{res},
	res = lfOmniCameraSetMaxBufferSize[camIndex, maxBufferSize];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`GetMaxBufferSize[] := IMAQTools`OmniStream`GetMaxBufferSize[$omniCamIndex]
IMAQTools`OmniStream`GetMaxBufferSize[camIndex_Integer] := tryload@(
	Return@(Quiet[lfOmniCameraGetMaxBufferSize[camIndex]] /. Except[_Integer] -> $Failed);
)

createImageWithMetaInformation[{imgKey_, metaKey_}] := Module[{img, meta, dims, n, v, k, metaAssoc = <||>},
	img = Quiet[lfOmniCameraRetrieveImage[imgKey]];
	If[!ImageQ[img], Return[$Failed]];
	meta = Quiet[lfOmniCameraRetrieveMetaInformation[metaKey]];
	If[Head[meta] =!= LibraryFunctionError,
		dims = ImageDimensions[img];
		meta = SplitBy[Normal[meta], SameQ[#, 0]&][[1;; ;;2]];
		While[Length[meta] > 0,
			n = meta[[1, 1]];
			k = FromCharacterCode[meta[[2]]];
			v = FromCharacterCode[Take[meta, {3, n + 2}]];
			v = Replace[If[StringMatchQ[#, NumberString], ToExpression[#], #] & /@ v, {x_} :> x];
			meta = Drop[meta, n + 2];
			AssociateTo[metaAssoc, k -> v];
		];
		If[!MissingQ[metaAssoc["DateTime"]], metaAssoc["DateTime"] = DateObject[ToExpression /@ StringSplit[metaAssoc["DateTime"], " "]]];
		AssociateTo[metaAssoc, {"PixelXDimension" -> First[dims], "PixelYDimension" -> Last[dims], "TimeZoneOffset" -> $TimeZone, "Software" -> System`ConvertersDump`Utilities`$signature}];
		Image[img, MetaInformation -> <|"Exif" -> metaAssoc|>]
		,
		img
	]
]

Options[IMAQTools`OmniStream`GrabFrame] = {"CheckAutoExposure" -> False};

IMAQTools`OmniStream`GrabFrame[opts:OptionsPattern[]] := IMAQTools`OmniStream`GrabFrame[$omniCamIndex, opts]
IMAQTools`OmniStream`GrabFrame[camIndex_Integer, opts:OptionsPattern[]] := tryload@Module[{keys},
	keys = If[TrueQ[OptionValue@"CheckAutoExposure"], 
		Quiet@lfOmniCameraGrabAutoExposedFrame[camIndex]
		, 
		Quiet@lfOmniCameraGrabFrame[camIndex]
	];
	If[!(ListQ[keys] && Length[keys] == 2), If[Head[keys] === LibraryFunctionError && First[keys] === "LIBRARY_NUMERICAL_ERROR", Message[IMAQTools::grabframetimeout, $omniCamIndex];]; Return[$Failed];];
	createImageWithMetaInformation[keys]
]

Options[IMAQTools`OmniStream`GrabNFrames] = {"CheckAutoExposure" -> False};

IMAQTools`OmniStream`GrabNFrames[numFrames_Integer, opts:OptionsPattern[]] := IMAQTools`OmniStream`GrabNFrames[$omniCamIndex, numFrames, opts]
IMAQTools`OmniStream`GrabNFrames[camIndex_Integer, numFrames_Integer, opts:OptionsPattern[]] := tryload@Module[{keys},
	keys = If[TrueQ[OptionValue@"CheckAutoExposure"],
		Quiet@lfOmniCameraGrabAutoExposedNFrames[camIndex, numFrames]
		,
		Quiet@lfOmniCameraGrabNFrames[camIndex, numFrames]
	];
	If[!(ListQ[keys] && Dimensions[keys] === {numFrames, 2}), If[Head[keys] === LibraryFunctionError && First[keys] === "LIBRARY_NUMERICAL_ERROR", Message[IMAQTools::grabframetimeout, $omniCamIndex];]; Return[$Failed]];
	createImageWithMetaInformation /@ keys
]

IMAQTools`OmniStream`KeepAlive[] := IMAQTools`OmniStream`KeepAlive[$omniCamIndex]
IMAQTools`OmniStream`KeepAlive[camIndex_Integer] := tryload@Module[{res},
	res = lfOmniCameraKeepAlive[camIndex];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`SetShutoffTime[shutoffTimeout_Integer] := IMAQTools`OmniStream`SetShutoffTime[$omniCamIndex, shutoffTimeout]
IMAQTools`OmniStream`SetShutoffTime[camIndex_Integer, shutoffTimeout_Integer] := tryload@Module[{res},
	res = lfOmniCameraSetShutoffTime[camIndex, shutoffTimeout];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`GetShutoffTime[] := IMAQTools`OmniStream`GetShutoffTime[$omniCamIndex]
IMAQTools`OmniStream`GetShutoffTime[camIndex_Integer] := tryload@(
	Return@(Quiet@lfOmniCameraGetShutoffTime[camIndex] /. Except[_Integer] -> $Failed)
)

IMAQTools`OmniStream`ThrottlingParams[] := $throttleParams

IMAQTools`OmniStream`ThrottlingParams[param_String -> value_] := IMAQTools`OmniStream`ThrottlingParams[$omniCamIndex, param -> value]
IMAQTools`OmniStream`ThrottlingParams[camIndex_Integer, param_String -> value_] := tryload@Module[{res},
	res = lfOmniCameraSetThrottlingParam[camIndex, $throttleParamsAssoc[param], N@value];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`ThrottlingParams[param_String] := IMAQTools`OmniStream`ThrottlingParams[$omniCamIndex, param]
IMAQTools`OmniStream`ThrottlingParams[camIndex_Integer, param_String] := tryload@Module[{res},
	Return@(Quiet@lfOmniCameraGetThrottlingParam[camIndex, $throttleParamsAssoc[param]] /. Except[r_Real] -> $Failed);
]

IMAQTools`OmniStream`SetPrintStatStream[print:(True|False)] := IMAQTools`OmniStream`SetPrintStatStream[$omniCamIndex, print]
IMAQTools`OmniStream`SetPrintStatStream[camIndex_Integer, print:(True|False)] := tryload@Module[{res},
	res = lfOmniCameraSetPrintStatStream[camIndex, print];
	Return@If[res == 0, Null, $Failed]
]

IMAQTools`OmniStream`AutoExposureParams[] := $autoExposureParams

IMAQTools`OmniStream`AutoExposureParams[param_String -> value_] := IMAQTools`OmniStream`AutoExposureParams[$omniCamIndex, param -> value]
IMAQTools`OmniStream`AutoExposureParams[camIndex_Integer, param_String -> value_] := tryload@Module[{res},
	res = Switch[param,
		"MaxIterations", If[!MatchQ[value, _Integer], $Failed, lfOmniCameraSetAutoExposeMaxIterations[camIndex, value]],
		"NumStableFrames", If[!MatchQ[value, _Integer], $Failed, lfOmniCameraSetAutoExposeNumStableFrames[camIndex, value]],
		"ImageMeanThreshold", If[!MatchQ[value, _Real], $Failed, lfOmniCameraSetAutoExposeImageMeanThreshold[camIndex, value]],
		"ImageMeanStabilityThreshold", If[!MatchQ[value, _Real], $Failed, lfOmniCameraSetAutoExposeImageMeanStabilityThreshold[camIndex, value]],
		"Timeout", If[!MatchQ[value, _Real], $Failed, lfOmniCameraSetAutoExposeTimeout[camIndex, value]],
		_, $Failed
	];
	Return@If[res === 0, Null, $Failed]
]

IMAQTools`OmniStream`AutoExposureParams[param_String] := IMAQTools`OmniStream`AutoExposureParams[$omniCamIndex, param]
IMAQTools`OmniStream`AutoExposureParams[camIndex_Integer, param_String] := tryload@Module[{res},
	Return@Switch[param,
		"MaxIterations", Quiet@lfOmniCameraGetAutoExposeMaxIterations[camIndex] /. Except[i_Integer] -> $Failed,
		"NumStableFrames", Quiet@lfOmniCameraGetAutoExposeNumStableFrames[camIndex] /. Except[i_Integer] -> $Failed,
		"ImageMeanThreshold", Quiet@lfOmniCameraGetAutoExposeImageMeanThreshold[camIndex] /. Except[i_Real] -> $Failed,
		"ImageMeanStabilityThreshold", Quiet@lfOmniCameraGetAutoExposeImageMeanStabilityThreshold[camIndex] /. Except[i_Real] -> $Failed,
		"Timeout", Quiet@lfOmniCameraGetAutoExposeTimeout[camIndex] /. Except[_Real] -> $Failed,
		_, $Failed
	]; 
]

Options[IMAQTools`OmniStream`SetGrabFrameTimeout] = {"FirstFrame" -> Automatic, "General" -> Automatic};

IMAQTools`OmniStream`SetGrabFrameTimeout[opts:OptionsPattern[]] := IMAQTools`OmniStream`SetGrabFrameTimeout[$omniCamIndex, opts]
IMAQTools`OmniStream`SetGrabFrameTimeout[camIndex_Integer, opts:OptionsPattern[]] := tryload@Module[{t1, t2, res},
	t1 = N[OptionValue["FirstFrame"] /. Default -> -1];
	t2 = N[OptionValue["General"] /. Default -> -1];
	If[MatchQ[t1, _Real],
		res = lfOmniCameraSetGrabFrameTimeout[camIndex, t1, True];
		If[ res != 0, Return[$Failed];];
	];
	If[MatchQ[t2, _Real],
		res = lfOmniCameraSetGrabFrameTimeout[camIndex, t2, False];
		If[ res != 0, Return[$Failed];];
	];
	Return[Null];
]

IMAQTools`OmniStream`GetGrabFrameTimeout[] := IMAQTools`OmniStream`GetGrabFrameTimeout[$omniCamIndex]
IMAQTools`OmniStream`GetGrabFrameTimeout[camIndex_Integer] := tryload@Module[{res1, res2},
	res1 = Quiet@lfOmniCameraGetGrabFrameTimeout[camIndex, True] /. Except[t_Real] -> $Failed;
	res2 = Quiet@lfOmniCameraGetGrabFrameTimeout[camIndex, False] /. Except[t_Real] -> $Failed;
	Return@If[res1 === $Failed || res2 === $Failed, $Failed, Association@MapThread[Rule, {First /@ Options[IMAQTools`OmniStream`SetGrabFrameTimeout], {res1, res2}}]]
]

IMAQTools`OmniStream`GetUseThrottling[] := IMAQTools`OmniStream`GetUseThrottling[$omniCamIndex]
IMAQTools`OmniStream`GetUseThrottling[camIndex_Integer] := tryload@(
	Return@(Quiet@lfOmniCameraGetUseThrottling[camIndex] /. Except[b:(True|False)] -> $Failed)
)

IMAQTools`OmniStream`GetThrottlingMethod[] := IMAQTools`OmniStream`GetThrottlingMethod[$omniCamIndex]
IMAQTools`OmniStream`GetThrottlingMethod[camIndex_Integer] := tryload@(
	Return@(Quiet@lfOmniCameraGetThrottlingMethod[camIndex] /. Except[_Integer] -> $Failed)
)

IMAQTools`OmniStream`GetUseFrontendFpsExp[] := IMAQTools`OmniStream`GetUseFrontendFpsExp[$omniCamIndex]
IMAQTools`OmniStream`GetUseFrontendFpsExp[camIndex_Integer] := tryload@(
	Return@(Quiet@lfOmniCameraGetUseFrontendFpsExp[camIndex] /. Except[b:(True|False)] -> $Failed)
)

IMAQTools`OmniStream`GetPrintStatStream[] := IMAQTools`OmniStream`GetPrintStatStream[$omniCamIndex]
IMAQTools`OmniStream`GetPrintStatStream[camIndex_Integer] := tryload@(
	Return@(Quiet@lfOmniCameraGetPrintStatStream[camIndex] /. Except[b:(True|False)] -> $Failed)
)

IMAQTools`OmniStream`GetBufferModes[] := $omniBufferModes

IMAQTools`OmniStream`SetBufferMode[mode:(_String|_Function|_Composition)] := IMAQTools`OmniStream`SetBufferMode[$omniCamIndex, mode]
IMAQTools`OmniStream`SetBufferMode[camIndex_Integer, mode:(_String|_Function|_Composition)] := 
Module[{res, obj, enabled = True},
	If[StringQ[mode],
		omniStreamSetUseWLBufferFunction[camIndex, False];
		omniStreamSetNativeBufferMode[camIndex, mode]
		,
		omniStreamSetNativeBufferMode[camIndex, "Default"];
		If[(mode) === (Identity[#]&),
			omniStreamSetUseWLBufferFunction[camIndex, False]
			,
			If[AsynchronousTaskObject =!= Head@(obj = getOmniBufferFunAsyncObj[camIndex])
					|| (mode) =!= (Lookup["UserData" /. Options[obj], "BufferFunction"]),
				removeOmniBufferFunAsyncObj[camIndex];
				If[AsynchronousTaskObject === Head@(obj = omniStreamSetWLBufferFunction[camIndex, mode]),
					addOmniBufferFunAsyncObj[camIndex, obj];
					,
					enabled = False;
				];
			];
			omniStreamSetUseWLBufferFunction[camIndex, enabled];
			If[enabled, Null, $Failed]
		]
	]
]

IMAQTools`OmniStream`GetBufferMode[] := IMAQTools`OmniStream`GetBufferMode[$omniCamIndex]
IMAQTools`OmniStream`GetBufferMode[camIndex_Integer] := Module[{res},
	If[(AsynchronousTaskObject === Head@(res = getOmniBufferFunAsyncObj[camIndex]))
			&& TrueQ[omniStreamGetUseWLBufferFunction[camIndex]],
		Lookup["UserData" /. Options[res], "BufferFunction"]
		,
		omniStreamGetNativeBufferMode[camIndex]
	]
]

wlBufferFunProcessImage[asyncObj_, eventTag_, eventData_] := Module[{userData, img, resImg},
	userData = "UserData" /. Options[asyncObj];
	img = First[eventData];
	resImg = Quiet[CheckAbort[userData["BufferFunction"][img], Null]];
	lfOmniCameraAddWLBufferFunctionFrame[userData["StreamIndex"], Image[If[ImageQ[resImg], resImg, img], "Byte"]];
]

(* Library Function wrappers to check return values: *)

omniStreamSetNativeBufferMode[camIndex_Integer, mode_String] := tryload@Module[{res},
	res = lfOmniCameraSetBufferMode[camIndex, mode];
	Return@If[res == 0, Null, $Failed]
]

omniStreamGetNativeBufferMode[camIndex_Integer] := tryload@(
	Return@(Quiet@lfOmniCameraGetBufferMode[camIndex] /. Except[_String] -> $Failed /. "" -> "Default")
)

omniStreamSetWLBufferFunction[camIndex_Integer, f:(_Function|_Composition)] := tryload@Module[{res},
	res = Quiet@Internal`CreateAsynchronousTask[lfOmniCameraSetWLBufferFunction, {camIndex}, 
					(wlBufferFunProcessImage[#1, #2, #3]&), "UserData" -> <|"BufferFunction" -> f, "StreamIndex" -> camIndex|>];
	If[Head[res] === AsynchronousTaskObject, res, $Failed]
]

omniStreamRemoveWLBufferFunction[camIndex_Integer] := tryload@Module[{res},
	res = lfOmniCameraRemoveWLBufferFunction[camIndex];
	Return@If[res == 0, Null, $Failed]
]

omniStreamSetUseWLBufferFunction[camIndex_Integer, enabled:(True|False)] := tryload@Module[{res, obj},
	res = lfOmniCameraSetUseWLBufferFunction[camIndex, enabled];
	If[res == 0, 
		If[AsynchronousTaskObject === Head@(obj = getOmniBufferFunAsyncObj[camIndex]),
			Quiet@If[enabled, System`StartAsynchronousTaskObject[obj], System`StopAsynchronousTaskObject[obj]];
		];
		Null
		,
		$Failed
	]
]

omniStreamGetUseWLBufferFunction[camIndex_Integer] := tryload@(
	Return@(Quiet@lfOmniCameraGetUseWLBufferFunction[camIndex] /. Except[b:(True|False)] -> $Failed)
)

End[]

EndPackage[]

