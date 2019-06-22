BeginPackage["SoundFileTools`"]
Begin["`Private`"]

$InitSoundFileTools = False;

$ThisDirectory = FileNameDrop[$InputFileName, -1];
$BaseLibraryDirectory = FileNameJoin[{$ThisDirectory, "LibraryResources", $SystemID}];
$SoundFileToolsLibrary = "SoundFileTools";
Get[FileNameJoin[{$ThisDirectory, "LibraryResources", "LibraryLinkUtilities.wl"}]];
dlls["MacOSX-x86-64"] = {"libsndfile-tools.dylib"};
dlls["Linux"|"Linux-x86-64"] = {"libsndfile.so"};
dlls["Linux-ARM"] = {};
dlls["Windows"|"Windows-x86-64"] = {"libsndfile-1.dll", "libmad.dll"};
dlls[___] := $Failed;

InitSoundFileTools[debug_:False] :=
	If[ TrueQ[$InitSoundFileTools],
		$InitSoundFileTools
		,
		$InitSoundFileTools = Catch[
			If[ dlls[$SystemID] === $Failed,
				Message[SoundFileTools::sys, "Incompatible SystemID"];
				Throw[$Failed]
			];
			Block[{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},

				SetPacletLibrary[$SoundFileToolsLibrary];

				SafeLibraryLoad /@ Flatten[{dlls[$SystemID], $SoundFileToolsLibrary}];

				$ErrorDescription = SafeLibraryFunctionLoad["ErrorDescription", {}, "UTF8String"];

				(* Import *)
				$ImportSoundFile = SafeLibraryFunctionLoad["ImportSoundFile", {"UTF8String"}, "RawArray"];
				$ImportSoundIntegerMetadata = SafeLibraryFunctionLoad["ImportSoundIntegerMetadata", {"UTF8String"}, {_Integer,_}];
				If[$SystemID =!= "Linux-ARM",
					lf$ImportSoundIntegerMetadataList = SafeLibraryFunctionLoad["ImportSoundIntegerMetadataList", {Integer, {"RawArray", "Constant"}}, {_Integer, _}];
				];
				$ImportSoundStringMetadata = SafeLibraryFunctionLoad["ImportSoundStringMetadata", {"UTF8String"}, "UTF8String"];
				$ImportSoundSingleStringMetadata = SafeLibraryFunctionLoad["ImportSoundSingleStringMetadata", {"UTF8String", Integer}, "UTF8String"];

				(* Export *)
				$CheckFormatEncoding = SafeLibraryFunctionLoad["CheckFormatEncoding", {Integer, Integer, Integer, Integer}, Integer];
				$ExportSoundFile = SafeLibraryFunctionLoad["ExportSoundFile", {"UTF8String", {"RawArray", "Constant"}, Integer, Integer, Integer, {_Integer, 1}, "UTF8String", Real}, "Void"];
				$ExportSoundMetadata = SafeLibraryFunctionLoad["ExportSoundMetadata", {"UTF8String", {_Integer, 1}, "UTF8String"}, "Void"];

				(* Parallel Import for NetEncoder *)
				If[$SystemID =!= "Linux-ARM",
					lf$LoadNetEncoderFeaturesFromPaths = SafeLibraryFunctionLoad["LoadNetEncoderFeaturesFromPaths",
						{Integer, 					(* number of threads *)
						{Integer, 1, "Constant"}, 	(* feature enums *)
						{Integer, 2, "Constant"}, 	(* feature params *)
						{Integer, 1, "Constant"}, 	(* feature param lengths *)
						{"RawArray", "Constant"}, 	(* additional feature param data ("Real32") *)
						{"RawArray", "Constant"}, 	(* additional feature param paths ('\0'-separated, UTF8-encoded) *)
						{"RawArray", "Constant"} 	(* flat input audio paths ('\0'-separated, UTF8-encoded) *)
						}, "DataStore"]; 			(* flat feature data *)
					lf$LoadNetEncoderFeaturesFromData = SafeLibraryFunctionLoad["LoadNetEncoderFeaturesFromData",
						{Integer, 					(* number of threads *)
						{Integer, 1, "Constant"}, 	(* feature enums *)
						{Integer, 2, "Constant"}, 	(* feature params *)
						{Integer, 1, "Constant"}, 	(* feature param lengths *)
						{"RawArray", "Constant"}, 	(* additional feature param data ("Real32") *)
						{"RawArray", "Constant"}, 	(* additional feature param paths ('\0'-separated, UTF8-encoded) *)
						{"RawArray", "Constant"}, 	(* flat input audio data *)
						{Integer, 1, "Constant"}, 	(* flat input audio SampleRates *)
						{Integer, 2, "Constant"} 	(* input audio Dimensions *)
						}, "DataStore"]; 			(* flat feature data *)
					lf$NetEncoderFeatures = SafeLibraryFunctionLoad["NetEncoderFeatures", {}, "UTF8String"];
					$NetEncoderFeatures = Quiet[Check[StringSplit[lf$NetEncoderFeatures[], "\n"], 
													If[TrueQ[debug], Print["Failed to set $NetEncoderFeatures"]]; Throw[$Failed]]];
				];
				True
			]
		]
	]

$ImportSoundIntegerMetadataList[numThreads_, paths_] :=
With[{pathsConcat = RawArray["UnsignedInteger8", Flatten[Append[Riffle[ToCharacterCode[paths, "UTF8"], 0], 0]]]},
	lf$ImportSoundIntegerMetadataList[numThreads, pathsConcat]
]

Options[$LoadFeaturesFromPaths] := SoundFileTools`Private`NetEncoderDump`optionsForFeature["MFCC"]

(* For the following functions, 'opts' should contain the full set of options required for the requested features, with all values specified (i.e., no Automatic values). *)

(* Assumptions: (SoundFileTools`Private`NetEncoderDump`validPaths[paths] && SoundFileTools`Private`NetEncoderDump`validFeatures[features]) *)
$LoadFeaturesFromPaths[UNUSED:(True|False):False, paths_?ListQ, features_, opts:OptionsPattern[{$LoadFeaturesFromPaths, SoundFileTools`Private`NetEncoderDump`netEncoderExec2}]] /; (paths =!= {} && StringQ[First[paths]]) :=
(
	SoundFileTools`Private`NetEncoderDump`netEncoderExec2["File", paths, features, {opts}, $LoadFeaturesFromPaths, FilterRules[{opts}, Options[SoundFileTools`Private`NetEncoderDump`netEncoderExec2]]]
)

Options[$LoadFeaturesFromData] := Options[$LoadFeaturesFromPaths];

(* Assumptions: (SoundFileTools`Private`NetEncoderDump`validAudios[audios] && SoundFileTools`Private`NetEncoderDump`validFeatures[features]) *)
$LoadFeaturesFromData[UNUSED:(True|False):False, audios_?ListQ, features_, opts:OptionsPattern[{$LoadFeaturesFromData, SoundFileTools`Private`NetEncoderDump`netEncoderExec2}]] /; (audios =!= {} && AudioQ[First[audios]]) :=
(
	SoundFileTools`Private`NetEncoderDump`netEncoderExec2["Raw", Audio`InternalAudioData /@ audios, Audio`Utilities`AudioSampleRate /@ audios, features, {opts}, $LoadFeaturesFromData, FilterRules[{opts}, Options[SoundFileTools`Private`NetEncoderDump`netEncoderExec2]]]
)

(* Assumptions: (SoundFileTools`Private`NetEncoderDump`validData[rawArrays] && SoundFileTools`Private`NetEncoderDump`validRates[sampleRates] && SoundFileTools`Private`NetEncoderDump`validFeatures[features]) *)
$LoadFeaturesFromData[UNUSED:(True|False):False, rawArrays_?ListQ, sampleRates_, features_, opts:OptionsPattern[{$LoadFeaturesFromData, SoundFileTools`Private`NetEncoderDump`netEncoderExec2}]] /; (rawArrays =!= {} && Developer`RawArrayQ[First[rawArrays]]) :=
(
	SoundFileTools`Private`NetEncoderDump`netEncoderExec2["Raw", rawArrays, sampleRates, features, {opts}, $LoadFeaturesFromData, FilterRules[{opts}, Options[SoundFileTools`Private`NetEncoderDump`netEncoderExec2]]]
)

Begin["`NetEncoderDump`"]

$formatOptions = SortBy[#, ToString]& @
{
	(* True|False|None.
	When applied to "AudioData":
		If True, result dimensions for each signal are {conformedLength, nchannels*}
		If False, result dimensions for each signal are {nchannels, conformedLength}
		If None, result dimensions for each signal are {conformedLength}
		*Currently, all input signals are downmixed to mono; having an Interleaving option for
		"AudioData" was done in case there is ever an option to extract features per-channel.
	When applied to imaginary "STFT" data:
		If True, result dimensions for each signal are {NumberOfFrames, WindowSize, 2}
		If False, result dimensions for each signal are {NumberOfFrames, 2, WindowSize}
		If None, result dimensions for each signal are {2, NumberOfFrames, WindowSize}
	The only other feature besides "STFT" this affects is "AudioData", where the Interleaving
	 *)
	Interleaving -> Automatic
}

optionsForFeature["AudioData"] = SortBy[#, ToString]& @
Join[$formatOptions,
{
	(* Integer representing number of threads to use when extracting features.
		If > 0, specifies an explicit number of threads to use (does not check whether this is > NumAvailableHardwareThreads (HWT))
		If < 0, will use HWT
		If == 0, will use HWT / 2 + 1
		If <= 0 and information about HWT is unavailable, defaults to a suitable choice *)
	"NumberOfThreads" -> Automatic, 
	(* Positive Integer representing the target rate to resample input audio to before extracting features. *)
	SampleRate -> Automatic,
	(* Non-zero Integer representing target length (in Samples!) of input signals be fore computing features.
			If < 0, will use full length of signals (this could create ragged output)
			If > 0, will use only upto the specified target length from each signal (padding if necessary) *)
	"Duration" -> Automatic,
	(* Integer or vector of Integers (length == NumInputSignals) representing (in Samples!) a TimeShift specification.
		Note: Randomization of parameter values is no longer performed in SoundFileTools. For parameters which support
		randomization, use a single value to achieve No Randomization, or a list of values (length == NumInputSignals) to
		manually achieve Randomization by assigning to each input the parameter value drawn from its corresponding index in the list. *)
	"StartPosition" -> Automatic,
	(* A rule "<type:{"RMS","Max"}>" -> Real, or None representing any normalization to be applied to input signals.
		If "Max" -> val, val is a target maximum amplitude for each signal.
		If "RMS" -> val, val is a target rms amplitude value for each signal. *)
	"Normalize" -> Automatic,
	(* A Real, vector of Reals (length == NumInputSignals), or None representing level of noise to be added to inputs. *)
	"NoiseLevel" -> Automatic,
	(* Vector of Reals, RawArray, "<filepath>", or None representing the type of noise to apply to the inputs.
		If None, noise will be white noise drawn from a Uniform Distribution. *)
	"NoiseData" -> Automatic,
	(* Real or None representing coefficient for pre-emphasis filter to be applied to inputs.
		If Real, a pre-emphasis filter with coefficients {1, -abs(val)} will be used. *)
	"PreEmphasis" -> Automatic,
	(* Real, vector of Reals (length == NumInputSignals), or None representing mix level of convolution.
		If enabled, controls the level of convolution as follows: res <- (1-level)*orig + level*res *)
	"ConvolutionLevel" -> Automatic,
	(* Vector of Reals, RawArray, "<filepath>", or None representing the signal to convolve with the inputs.
		May not be None if convolution is enabled by "ConvolutionLevel". *)
	"ConvolutionData" -> Automatic,
	(* Real, vector or Reals (length == NumInputSignals), or None representing an amount by which to amplify inputs.
		If enabled, amplification is performed prior to normalization (if applicable). *)
	"VolumePerturbation" -> Automatic
}];
optionsForFeature["Spectrogram"] = SortBy[#, ToString]& @
Join[optionsForFeature["AudioData"],
	{
	(* Non-zero Integer representing target length of features.
		If < 0, the feature length will depend on the length of each input signal (or the length specified by "Duration").
		It really shouldn't be used in combination with a positive "Duration", but if it is, then "Duration" should be
		>= the number of samples needed to fulfill "NumberOfFrames"; otherwise SoundFileTools will return $Failed. *)
	"NumberOfFrames" -> Automatic,
	(* Positive integer representing (in Samples!) the size of the DiscreteFourierTransform to use, or -1 if it
		should be inferred from the WindowSize. If "DFTSize" is specified and is less than "WindowSize", then
		each window will be truncated. If "DFTSize" is specified and is greater than "WindowSize", then each
		window will be padded with zeros prior to computing the transform. *)
	"DFTSize" -> None, (* this should be set back to Automatic once NetEncoder code supports this option. *)
	(* Positive Integer representing (in Samples!) the size of window to use for computing the STFT upon which most features are based. *)
	"WindowSize" -> Automatic,
	(* Positive Integer representing (in Samples!) the stride size to use for computing the STFT. *)
	"Offset" -> Automatic,
	(* Vector of Reals, RawArray, "<ippString>" or None representing a smoothing window to apply to each partition of the STFT.
		Available strings are {"IPPHammingWindow", "IPPHannWindow", "IPPBartlettWindow", "IPPBlackmanWindow"*, "IPPKaiserWindow"*}.
			*must be specified as a rule "<ippString>" -> Real to supply a value for the window parameter. *)
	"Window" -> Automatic
	}
];
optionsForFeature["STFT"] = SortBy[#, ToString]& @
Join[optionsForFeature["Spectrogram"],
	(* True|False representing whether imaginary data is returned as Re & Im or as Abs & Arg components. *)
	{"ReIm" -> Automatic}
];
optionsForFeature["MelSpectrogram"] = SortBy[#, ToString]& @
Join[optionsForFeature["Spectrogram"],
	(* Positive Integer representing lower band edge (in Hertz!). Must not be greater than "HighFrequency". *)
	{"LowFrequency" -> Automatic, 
	(* Positive Integer representing upper band edge (in Hertz!). Must be less than "SampleRate"/2. *)
	"HighFrequency" -> Automatic, 
	(* Positive Integer representing number of bands to use when filtering the Spectrogram. *)
	"NumberOfFilters" -> Automatic,
	(* True|False representing whether to use the PowerSpectrum or MagnitudeSpectrum *)
	"UsePowerSpectrum" -> True, (* this should be set back to Automatic once NetEncoder code supports this option. *)
	(* A 3-element vector of Reals or None representing warping of the filter bands.
		If enabled, must specify {warpFactor, lowBandwidth, highBandwidth} *)
	"VTLP" -> Automatic}
];
optionsForFeature["MFCC"] = SortBy[#, ToString]& @
Join[optionsForFeature["MelSpectrogram"],
	(* Positive Integer representing number of coefficients to keep from DCT. Must not be greater than "NumberOfFilters". *)
	{"NumberOfCoefficients" -> Automatic}
];

Options[netEncoderExecReturn] = {
	"ReduceSingleFeature" -> False, (* If True and Length[features] == 1, will remove the redundant outer list *)
	"ReturnLibraryParams" -> False (* If True, will return the actual parameters (not including data and result lengths) passed in Sequence to the library function *)
};

Options[netEncoderExec2] = Options[netEncoderExecReturn];

netEncoderExec2["File", paths_, features_, opts_, head_, retOpts:OptionsPattern[]] :=
Module[{pathsConcat, res, params, lfparams, featureNames, numThreads},
	Catch[
		{featureNames, params, numThreads, lfparams} = netEncoderMultiFeatureParams[head, features, opts, retOpts];
		{pathsConcat} = netEncoderPaths[paths];
		res = SoundFileTools`Private`lf$LoadNetEncoderFeaturesFromPaths[numThreads, Sequence@@lfparams, pathsConcat];
		netEncoderExecReturn[res, params, lfparams, retOpts]
	]
]

netEncoderExec2["Raw", rawArrays_, sampleRates_, features_, opts_, head_, retOpts:OptionsPattern[]] :=
Module[{dataDims, flattenedData, res, params, lfparams, featureNames, numThreads},
	Catch[
		{featureNames, params, numThreads, lfparams} = netEncoderMultiFeatureParams[head, features, opts, retOpts];
		{flattenedData, dataDims} = netEncoderData[rawArrays];
		res = SoundFileTools`Private`lf$LoadNetEncoderFeaturesFromData[numThreads, Sequence@@lfparams, flattenedData, sampleRates, dataDims];
		netEncoderExecReturn[res, params, lfparams, retOpts]
	]
]

netEncoderExecReturn[resExpr_, paramsExpr_, lfparams_, opts:OptionsPattern[]] :=
Module[{res = resExpr, params = paramsExpr, depth, libraryParams = Nothing},
	If[Head@res =!= Developer`DataStore, Throw[$Failed]];
	If[TrueQ[OptionValue["ReturnLibraryParams"]], libraryParams = ("LibraryParameters" -> lfparams)];
	If[TrueQ[OptionValue["ReduceSingleFeature"]] && Length@res == 1, res = First@res];
	depth = ArrayDepth[res];
	Association["Data" -> Replace[Apply[List, res, {0, depth - 1}], {} -> $Failed, {depth}], libraryParams]
]

rawarrayFromPaths[paths_] := (RawArray["UnsignedInteger8", Flatten[Append[Riffle[ToCharacterCode[paths, "UTF8"], 0], 0]]]);

rawarraysFromRealParams[params_] := (RawArray["Real32", #]& /@
	Replace[Cases[Lookup[params, $real32Params, Nothing], (_List|_?NumericArrayQ|_Rule|_?Internal`RealValuedNumericQ)],
	{r_Rule :> {Last[r]}, r_?Internal`RealValuedNumericQ :> {r}, r_List :> Flatten[r]}, 1]);

netEncoderPaths[paths_] :=
{
	rawarrayFromPaths[paths]
}

netEncoderData[rawArrays_] :=
{
	Join[Sequence@@(Flatten[Developer`RawArrayConvert[#, "Real32"]]& /@ rawArrays)],
	Replace[Dimensions /@ rawArrays, {x_?IntegerQ} :> {1, x}, 1]
}

(* Utilities to handle conversion from strings to C enum values *)

$windowEnum = {None, "IPPHammingWindow", "IPPHannWindow", "IPPBartlettWindow", "IPPBlackmanWindow", "IPPKaiserWindow", "Data"};
$normEnum = {None, "Max", "RMS"};

fixOption["Normalize", optval_] :=
	({First@First@Position[$normEnum, First@#],
		Switch[Last@#, None, 0, _?Internal`RealValuedNumericQ, 1, _, Length@Last@#]}& @ If[optval === None, optval -> None, optval])

fixOption["NoiseLevel"
		|"ConvolutionLevel"
		|"VolumePerturbation", optval_] :=
	Switch[optval, None, 0, _?Internal`RealValuedNumericQ, 1, _, Length@optval]

fixOption["PreEmphasis", optval_] := (optval =!= None)

fixOption["StartPosition", optval_] :=
	{Switch[optval, None, 0, _?Internal`RealValuedNumericQ, 1, _, Length@optval], If[optval === None, Nothing, optval]}

fixOption["VTLP", optval_] := (If[optval === None, 0, If[ListQ@First@optval, Length@First@optval, 1]])

fixOption["NoiseData"
		|"ConvolutionData", optval_] :=
	{StringQ@optval, Switch[optval, _String|None, 0, _?Internal`RealValuedNumericQ, 1, _, Length@optval]}

fixOption["Window", optval_String -> _] := First@First@Position[$windowEnum, optval]
fixOption["Window", optval_] := First@First@Position[$windowEnum, Replace[optval, Except[_String|None] -> "Data"]]

fixOption[_, optval_] := optval

(* Takes a list of features (which could potentially be rules themselves with feature-specific options) and a list of global options (to apply to all features)
and creates a parameter association with proper params for all features that can be passed to the library link function as well as parameters that can 
be returned from the top-level function. *)
netEncoderMultiFeatureParams[head_, features_, opts_, retOptsExpr_] :=
Module[{featuresExpr, retOpts, counts, pos, params, paramArray, featureArray, paramLengths, numThreads, maxParamLen, intParams, realParams, fileParams, i = 0, featureNames},
	(* make sure all features in the list of features are rules of the form: "Feature" -> {feature options}, and check for incorrectly formatted features. *)
	featuresExpr = Replace[features, {f_String :> f -> {}, Rule[f_,l_?AssociationQ] :> f->Normal[l]}, {1}];
	If[!(++i; ListQ[Last[#]]), 
		Message[head::erropts, Last[#], "\""<>First[#]<>"\" (Feature "<>ToString[i]<>")"]; 
		Throw[$Failed]
	]& /@ featuresExpr;
	(* collect non-computation-related options for each feature. Feature-specific options should overwrite global options. *)
	retOpts = (First[#] -> Association[FilterRules[Join[FilterRules[retOptsExpr, Complement[First/@retOptsExpr, First/@Last[#]]], Last[#]], Options[netEncoderExecReturn]]])& /@ featuresExpr;
	(* populate each feature with global options, then overwrite with feature-specific options. It is being kept as a list to support multiple instances of the same feature type. *)
	featuresExpr = (First[#] -> FilterRules[Join[Last[#], FilterRules[opts, Except[Last[#]]]], optionsForFeature[First[#]]])& /@ featuresExpr;
	params = (First[#] -> netEncoderParams[Last[#], head, optionsForFeature[First[#]]])& /@ featuresExpr;
	(* features are given unique names for the output feature-param association *)
	counts = Table[0, Length@SoundFileTools`Private`$NetEncoderFeatures];
	params = Association@First@Last@Reap[Scan[(pos = First@First@Position[SoundFileTools`Private`$NetEncoderFeatures, First[#]]; 
												Sow@If[++counts[[pos]] > 1, First[#]<>"$"<>ToString[counts[[pos]]] -> Last[#], #]
												)&, params]];
	(* 'NumberOfThreads' is not included in the feature param-list for each feature, so get the max value to pass to the library function *)
	numThreads = Max[Values[Lookup[#, "NumberOfThreads", 1]& /@ params]];
	params = Association[Normal[#] /. Rule["NumberOfThreads", _] -> Rule["NumberOfThreads", numThreads]]& /@ params;
	(* list of named features *)
	featureNames = (First /@ featuresExpr);
	(* feature enum values. SoundFileTools`Private`$NetEncoderFeatures is set by a library function during InitSoundFileTools with strings positionally corresponding to the C feature-enum. *)
	featureArray = Flatten[Position[SoundFileTools`Private`$NetEncoderFeatures, #]& /@ featureNames];
	(* remove 'NumberOfThreads' from individual feature lists, and convert Reals so that feature params can be passed as a Tensor of mints *)
	intParams = Association[Normal[Delete[#, Key["NumberOfThreads"]]] /. {x:(True|False) :> Boole[x], None -> -1}]& /@
					(MapIndexed[fixOption[First@First@#2, #1]&, #]& /@ params);
	(* get param-list lengths for each feature *)
	maxParamLen = Max[(paramLengths = Length[Flatten[#]]&/@ Values[Values /@ intParams])];
	(* pad the feature param-lists to fit into Tensor of all feature params *)
	paramArray = ArrayPad[Flatten[#], {0, maxParamLen - Length[Flatten[#]]}, -1]& /@ Values[Values /@ intParams];
	(* get additional param data *)
	realParams = Replace[Join[Sequence@@Flatten[rawarraysFromRealParams /@ Values[params], 1]], {} :> RawArray["Real32", {0.}]];
	fileParams = rawarrayFromPaths[Cases[Lookup[#, $filepathParams, Nothing]& /@ Values[params], _String, 2]];
	(* first 3 values are for returning useful information along with the flat result data; the final list is for passing in Sequence to the library function *)
	{featureNames, params, numThreads, {featureArray, paramArray, paramLengths, realParams, fileParams}}
]

$netEncoderPositionedParams = {
	"NumberOfThreads",
	"Interleaving",
	"SampleRate",
	"Duration",
	"StartPosition",
	"Normalize",
	"NoiseLevel",
	"NoiseData",
	"PreEmphasis",
	"ConvolutionLevel",
	"ConvolutionData",
	"VolumePerturbation",
	"NumberOfFrames",
	"DFTSize",
	"WindowSize",
	"Offset",
	"Window",
	"ReIm",
	"LowFrequency",
	"HighFrequency",
	"NumberOfFilters",
	"UsePowerSpectrum",
	"VTLP",
	"NumberOfCoefficients"
};

$real32Params = Select[$netEncoderPositionedParams, 
	MemberQ[{"Normalize",
			"Window",
			"NoiseLevel",
			"NoiseData",
			"PreEmphasis",
			"ConvolutionLevel",
			"ConvolutionData",
			"VolumePerturbation",
			"VTLP"}, #]&]; (* to preserve positioned ordering *)

$filepathParams = Select[$netEncoderPositionedParams,
	MemberQ[{"NoiseData",
			"ConvolutionData"}, #]&]; (* to preserve positioned ordering *)

netEncoderParams[opts_, f_Symbol] := netEncoderParams[opts, f, Options[f]]
netEncoderParams[opts_, f_Symbol, fopts_List] := Module[{n},
	Check[
		n = ToString[First[#]]& /@ fopts;
		AssociationThread[#, OptionValue[fopts, {opts}, #]]& @ Select[$netEncoderPositionedParams, MemberQ[n, #]&]
		,
		Throw[$Failed]
	]
]

validPaths[paths_] := (paths =!= {} && VectorQ[paths, StringQ])
validAudios[audios_] := (audios =!= {} && VectorQ[audios, AudioQ])
validData[rawArrays_] := (rawArrays =!= {} && VectorQ[rawArrays, Developer`RawArrayQ])
validRates[sampleRates_] := (sampleRates =!= {} && VectorQ[sampleRates, Internal`NonNegativeIntegerQ])
validFeatures[features_] := (features =!= {} && VectorQ[features, MemberQ[SoundFileTools`Private`$NetEncoderFeatures, If[MatchQ[#, Rule[_, _?AssociationQ]], First[#], #]]&])

End[] (* NetEncoderDump` *)

End[]
EndPackage[]
