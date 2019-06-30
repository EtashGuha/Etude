Package["NeuralNetworks`"]

(*----------------------------------------------------------------------------*)
(* Audio encoder utilities

Three types of Audio: File["something.mp3"], Audio[File[...]], Audio[NumericArray[...]].
audioFeatureExtract is common to all feature extraction methods that correctly
handles all audio cases

*)

LoadSFT := Block[{$ContextPath = $ContextPath}, 
	Needs["SoundFileTools`"];
	If[!SoundFileTools`Private`InitSoundFileTools[],
		Panic["Cannot load SoundFileTools"]
	];
	Clear[LoadSFT];
];

PackageScope["audioFeatureExtract"]

audioFeatureExtract[{}, _, _] := {}

(* audioFeatureExtract fast path for a pure list of files *)
audioFeatureExtract[audio:{__File}, featType_String, params_Association] := Scope[
	LoadSFT;
	out = fileAudioFeatureExtract[audio[[All, 1]], featType, params];
	(* check for failures *)
	audioEncFailureCheck[out, audio]
]

(* audioFeatureExtract general case *)
audioFeatureExtract[audio_List, featType_String, params_Association] := Scope[
	LoadSFT;
	canonicalAudio = audioCanonicalize /@ audio;
	audioPos = Flatten @ Position[canonicalAudio, _Audio];
	filePos = Complement[Range[Length @ audio], audioPos];
	(* compute file features *)
	fileArrays = fileAudioFeatureExtract[canonicalAudio[[filePos]], featType, params];
	audioArrays = objectAudioFeatureExtract[canonicalAudio[[audioPos]], featType, params];
	(* construct output array *)
	out = Range[Length @ audio];
	out[[filePos]] = fileArrays;
	out[[audioPos]] = audioArrays;
	audioEncFailureCheck[out, audio]
]

symbolicDistributionToVector[params_, length_] :=
Replace[params,
	{
		dist_?UnivariateDistributionQ :> RandomVariate[dist, length],
		Round[Times[x_, dist_?UnivariateDistributionQ]] :> Round[Times[x, RandomVariate[dist, length]]]
	},
	{1, 2}
]

computeStartPositions[params_, lengths_] := Scope[
	p = params;
	Which[
		MatchQ[p["StartPosition"], Scaled[{_, _}]],
			scaledRanges = {- # + 1, - # + 1} & /@ lengths;
			scaledRanges = Map[- First[p["StartPosition"]] * # &, scaledRanges];
			scaledRanges = Round[scaledRanges];
			p["StartPosition"] = RandomInteger /@ scaledRanges;
		,
		(* no randomization case *)
		MatchQ[p["StartPosition"], Scaled[_?NumberQ]],
			p["StartPosition"] = First[p["StartPosition"]] * lengths;
	];
	p
]

invertStartPosition[params_] := If[params["StartPosition"] === None, params, MapAt[Minus, params, Key["StartPosition"]]]

prepareParameters[params_, length_] := invertStartPosition[symbolicDistributionToVector[params, length]]
prepareParameters[params_, length_, lengths_] :=
invertStartPosition[
	symbolicDistributionToVector[
		computeStartPositions[params, lengths],
		length
	]
]

positiveUnivariateDistributionQ[dist_] :=
	realDomainDistributionQ[dist] && DistributionDomain[dist][[1, 1]] >= 0.

realDomainDistributionQ[dist_] :=
	VectorQ[#, Internal`RealValuedNumericQ[#] || Abs[#] === Infinity &] & @@ DistributionDomain[dist]


PackageScope["AudioEncoderSequenceLengths"]

AudioEncoderSequenceLengths[params_, input_] /; AnyTrue[input, mp3Q] := $Failed
AudioEncoderSequenceLengths[params_, input_] :=
Scope[
	{sampleRate, offset} = Lookup[params, {"SampleRate", "Offset"}, None];
	computeAudioLength[input, sampleRate, offset]
]

mp3Q[s_String] := StringEndsQ[s, {"mp3", "mpg"}, IgnoreCase -> True]
mp3Q[File[s_String]] := mp3Q[s]
mp3Q[a_?Audio`AudioStreamQ] := mp3Q@Audio`AudioInformation[a, "ResourcePath"]
mp3Q[_] := False

ClearAll[computeAudioLength];
computeAudioLength[in_Audio, targetSampleRate_, None] :=
	Round[Divide @@ Audio`AudioInformation[in, {"Length", "SampleRate"}] * targetSampleRate]
computeAudioLength[in_File, targetSampleRate_, None] :=
	Round[targetSampleRate * #Length / #SampleRate & @ Audio`Utilities`ExtractFileMetadata[First[in]]]
computeAudioLength[in:Except[_List], targetSampleRate_, offset_] :=
	Ceiling[computeAudioLength[in, targetSampleRate, None] / offset]
computeAudioLengthNonMP3[in:{__File}, targetSampleRate_, offset_]:=
Scope[
	res = Round[
		targetSampleRate * Apply[
			Divide,
			SoundFileTools`Private`$ImportSoundIntegerMetadataList[- 1, in[[All, 1]]][[All, {3, 2}]],
			{1}
		]
	];
	If[offset =!= None,
		res = Ceiling[res / offset];
	];
	res
]
computeAudioLength[in:{__Audio}, targetSampleRate_, offset_] :=
	Map[computeAudioLength[#, targetSampleRate, offset]&, in]
computeAudioLength[in:{__}, targetSampleRate_, offset_]:=
Scope[
	audioPos = Flatten@Position[in, _Audio];
	mp3Pos = Flatten@Position[in, File[_?mp3Q]];
	filePos = Complement[Range[Length[in]], Join[audioPos, mp3Pos]];

	res = ConstantArray[0, Length[in]];
	res[[audioPos]] = computeAudioLength[in[[audioPos]], targetSampleRate, offset];
	res[[filePos]] = computeAudioLengthNonMP3[in[[filePos]], targetSampleRate, offset];
	res[[mp3Pos]] = computeAudioLength[#, targetSampleRate, offset] & /@ in[[mp3Pos]];

	res
]
computeAudioLength[___] := $Failed

(* call file-based feature extractor *)
fileAudioFeatureExtract[files_List, featType_String, params_Association] /; Length[files] > 4000 :=
	Flatten[
		Map[
			fileAudioFeatureExtract[#, featType, params]&,
			Partition[files, UpTo[4000], 4000]
		],
		1
	]
fileAudioFeatureExtract[files_List, featType_String, params_Association] /; Length[files] > 0 := Scope[
	(* check that files exist *)
	If[!Quiet@TrueQ@FileExistsQ[#], EncodeFail["path `` does not exist", #]]& /@ files;
	out = SoundFileTools`Private`$LoadFeaturesFromPaths[
		True,
		files,
		{featType},
		Sequence @@ Normal[
			prepareParameters[
				Sequence @@ {
					params,
					Length[files],
					If[params["DurationQ"],
						computeAudioLength[File /@ files, params["SampleRate"], None],
						Nothing
					]
				}
			]
		]
	];
	(* if no files could be loaded in a batch, $LoadFeaturesFromPaths returns 
		$Failed. We will assume something has gone very wrong and fail.
	*)
	If[FailureQ[out], EncodeFail["Could not load any audio files in the batch"]];
	First @ out["Data"]
]

(* call Audio[..]-based feature extractor *)
objectAudioFeatureExtract[audios_List, featType_String, params_Association] /; Length[audios] > 4000 := 
	Flatten[
		Map[
			objectAudioFeatureExtract[#, featType, params]&,
			Partition[audios, UpTo[4000], 4000]
		],
		1
	]
objectAudioFeatureExtract[audios_List, featType_String, params_Association] /; Length[audios] > 0 := (
	(* check for valid audio objects *)
	If[!AudioQ[#], EncodeFail["`` is an invalid Audio object"], #]& /@ audios;
	First @ SoundFileTools`Private`$LoadFeaturesFromData[
		True,
		audios,
		{featType},
		Sequence @@ Normal[
			prepareParameters[
				Sequence @@ {
					params,
					Length[audios],
					If[params["DurationQ"],
						computeAudioLength[audios, params["SampleRate"], None],
						Nothing
					]
				}
			]
		]
	]["Data"]
)

(* convert File[..] and stream Audio[..] objects to string filename, 
	and keep non-stream Audio[..] as Audio[..] *)
audioCanonicalize[audio_?AudioQ] :=
Which[
	# === None,
		audio,
	TrueQ[FileExistsQ[#]],
		#,
	True,
		FindFile[#]
] & @ Audio`AudioInformation[audio, "ResourcePath"]
audioCanonicalize[path_File] := First[path];
audioCanonicalize[___] := EncodeFail["input is neither an Audio object or a File"]

(* failure handling *)
NetEncoder::audioimprt = "Cannot load ``. Using random audio instead.";

(* no failure case *)
audioEncFailureCheck[arrays:{___NumericArray}, _] := arrays
(* at least one failure *)
NetEncoder::audioimprt = "Cannot load ``. Using random audio from batch instead."
audioEncFailureCheck[arrays_List, inputs_List] := Scope[
	pos = Flatten @ Position[arrays, $Failed];
	If[Length[pos] === Length[inputs],
		EncodeFail["Could not load any audio files in the batch"];
	];
	Message[NetEncoder::audioimprt, #] & /@ inputs[[pos]];
	sampleArray = RandomSample[Select[arrays,  NumericArrayQ], 1][[1]];
	Replace[arrays, $Failed -> sampleArray, {1}]
]

PackageScope["checkAudioNormalization"]

checkAudioNormalization[e_] :=
	Which[
		MatchQ[e, Automatic|True|"Max"|"RMS"|None],
			e
		,
		MatchQ[e, {"Max"|"RMS", x_?Internal`RealValuedNumericQ}],
			If[e[[2]] <= 0,
				FailCoder["`1` is not a valid value for \"Normalization\". It should be a positive value.", e[[2]]]
				,
				e
			]
		,
		True,
			FailCoder["`` is not a valid value for \"Normalization\". Possible values are None, True, \"Max\" or \"RMS\".", e]
	];


PackageScope["CheckWindowFunction"]

CheckWindowFunction[windowFunction_, winsize_] :=
If[
	(!ListQ[windowFunction] && !MatchQ[windowFunction, None|Automatic] &&
	!VectorQ[Quiet@Array[windowFunction, 3, {-.5, .5}], Internal`RealValuedNumericQ]) ||
	(ListQ[windowFunction] && Length[windowFunction] =!= winsize)
	,
	FailCoder[
		"`1` is not a valid value for \"WindowFunction\". It should be None, a window function or a list of length `2`.",
		windowFunction, winsize
	]
]

WindowFunctionToArray[windowFunction_, windowSize_, "Audio"] := None
WindowFunctionToArray[windowFunction_, windowSize_, _] :=
Which[
	ListQ[windowFunction] || windowFunction === None,
		windowFunction
	,
	windowFunction === Automatic,
		0.54 - 0.46 Cos[((2*Pi*(Range[-.5, .5, 1./(windowSize - 1)] + .5)))]
	,
	VectorQ[windowFunction[Range[-.5, .5, .5]], Internal`RealValuedNumericQ],
		windowFunction[Range[-.5, .5, 1./(windowSize - 1)]]
	,
	True,
		Array[windowFunction, windowSize, {-.5, .5}]
]


PackageScope["GenerateSFTParameters"]

GenerateSFTParameters[params_, type_, feat_] := Scope[
	augmentionPars = StripVP[Lookup[params, "Augmentation", None]];
	If[augmentionPars === None,
		augmentionPars = AssociationThread[{"TimeShift", "Volume", "Noise", "Convolution", "VTLP"} -> None];
	];
	wSize = Lookup[params, "WindowSize", None];
	If[wSize =!= None,
		wSize = TimeToSamples[wSize, params["SampleRate"]]
	];
	offset = Lookup[params, "Offset", None];
	If[offset =!= None,
		offset = TimeToSamples[offset, params["SampleRate"]]
	];
	Association@Flatten@{
		"NumberOfFrames" -> If[feat === "Audio", -1, OutTypeLen[type, -1]],
		"Duration" -> If[feat === "Audio", OutTypeLen[type, -1], -1],
		"StartPosition" -> With[{ts = augmentionPars["TimeShift"]}, If[!MatchQ[ts, None|All|Scaled[_]], Round[params["SampleRate"]*ts], ts]],
		"RandomizeStart" -> !MatchQ[augmentionPars["TimeShift"], None],
		"DurationQ" -> MatchQ[augmentionPars["TimeShift"], All|Scaled[_]],
		"VolumePerturbation" -> augmentionPars["Volume"],
		With[
			{n = augmentionPars["Noise"]},
			Switch[
				n,
				None,
					{"NoiseLevel" -> None, "NoiseData" -> None},
				_?NumericQ|_?UnivariateDistributionQ,
					{"NoiseLevel" -> n, "NoiseData" -> None},
				_File|_Audio,
					{
						"NoiseLevel" -> 1,
						"NoiseData" -> convertToAudioData[n, params["SampleRate"]]
					},
				{_, _},
					{
						"NoiseLevel" -> First[n],
						"NoiseData" -> convertToAudioData[Last[n], params["SampleRate"]]
					}
			]
		],
		With[
			{c = augmentionPars["Convolution"]},
			Switch[
				c,
				None,
					{"ConvolutionLevel" -> None, "ConvolutionData" -> None}
				,
				_File|_Audio,
					{
						"ConvolutionLevel" -> 1.,
						"ConvolutionData" -> convertToAudioData[c, params["SampleRate"]]
					}
				,
				_,
					{
						"ConvolutionLevel" -> First[c],
						"ConvolutionData" -> convertToAudioData[Last[c], params["SampleRate"]]
					}
			]
		],
		"VTLP" -> If[Lookup[augmentionPars, "VTLP", None] === None,
			None
			,
			{
				augmentionPars["VTLP"][[1]],
				augmentionPars["VTLP"][[2]] /. Automatic -> Lookup[params, "MinimumFrequency", None],
				augmentionPars["VTLP"][[3]] /. Automatic -> Min[4500, .9*Lookup[params, "MaximumFrequency", None]]
			}
		],
		"SampleRate" -> params["SampleRate"],
		"Normalize" -> Replace[
				StripVP[params["Normalization"]],
				{
					(Automatic|True|"Max") -> ("Max" -> 1.),
					(* this is -20 dB *)
					"RMS" -> ("RMS" -> .1),
					{x_, y_} -> (x -> y)
				}
			],
		(* This lets SoundFileTools compute the number of threads using thread::hardware_concurrency *)
		"NumberOfThreads" -> -1,
		"WindowSize" -> wSize,
		"Offset" ->  offset,
		"Window" -> WindowFunctionToArray[Lookup[params, "WindowFunction", Automatic], wSize, feat],
		"Interleaving" -> If[MatchQ[feat, "Audio"|"AudioSTFT"], True, False],
		"NumberOfCoefficients" -> Lookup[params, "NumberOfCoefficients", None],
		"NumberOfFilters" -> Lookup[params, "NumberOfFilters", None],
		"LowFrequency" -> Lookup[params, "MinimumFrequency", None],
		"HighFrequency" -> Lookup[params, "MaximumFrequency", None],
		"ReIm" -> True,
		"PreEmphasis" -> Lookup[params, "$PreEmphasis", None]
	}
]

convertToAudioData[in_, sampleRate_] := Scope[
	audio = in;
	If[Head[audio] === File, audio = Import[audio]];
	NumericArray[
		Mean@AudioData[
			If[Audio`Utilities`AudioSampleRate[audio] === sampleRate,
				audio
				,
				AudioResample[audio, sampleRate]
			]
		],
		"Real32"
	]
]

toUniformDistributionPatterns =
{
	range:{_?NumericQ, _?NumericQ} :> Chop[UniformDistribution[range]],
	{range:{_?NumericQ, _?NumericQ}, rest___} :> {Chop[UniformDistribution[range]], rest}
};

PackageScope["CheckAudioAugmentation"]

CheckAudioAugmentation[None, _] := None
CheckAudioAugmentation[{}, _] := None
CheckAudioAugmentation[aug:{(_String->_)..}, feat_] :=
Join[
	AssociationThread[{"TimeShift", "Volume", "Noise", "Convolution", If[MatchQ[feat, "AudioMelSpectrogram"|"AudioMFCC"], "VTLP", Nothing]} -> None],
	Association@Map[CheckAudioAugmentation[#, feat]&, aug]
]

CheckAudioAugmentation["TimeShift" -> x_, _] := Scope[
	res = Replace[x, First[toUniformDistributionPatterns]];
	Which[
		Or[
			MatchQ[
				res,
				Alternatives[
					None,
					All,
					Scaled[{t1_, t2_}] /; - Infinity < t1 < t2 < Infinity,
					Scaled[t_] /; - Infinity < t < Infinity
				]
			],
			Internal`RealValuedNumericQ[res],
			UnivariateDistributionQ[res]
		],
			"TimeShift" -> Replace[res, All -> Scaled[{-1., 0.}]]
		,
		True,
			FailCoder[
				"`` is not a valid value for \"TimeShift\". Possible values are None, a \
number, a list of two numbers, a Scaled value or a univariate distribution.",
				x
			]
	]
]

CheckAudioAugmentation["Volume" -> x_, _] := Scope[
	res = Replace[x, First[toUniformDistributionPatterns]];
	Which[
		Or[
			res === None,
			positiveUnivariateDistributionQ[res]
		],
			"Volume" -> res
		,
		True,
			FailCoder[
				"`` is not a valid value for \"Volume\". Possible values are None, \
a list of two positive numbers or a univariate distribution.",
				x
			]
	]
]

validNoiseOrConvDataQ[data_] /; VectorQ[data, Internal`RealValuedNumericQ] := False
validNoiseOrConvDataQ[data_?AudioQ] := True
validNoiseOrConvDataQ[File[data_String]] := FileExistsQ[FindFile[data]]

CheckAudioAugmentation["Noise" -> x_, _] := Scope[
	res = Replace[x, toUniformDistributionPatterns];
	Which[
		Or[
			res === None,
			Internal`RealValuedNumericQ[res] && res > 0,
			MatchQ[res, {v_?Internal`RealValuedNumericQ, data_?validNoiseOrConvDataQ} /; v > 0],
			positiveUnivariateDistributionQ[res],
			MatchQ[res, data_?validNoiseOrConvDataQ],
			MatchQ[res, {v_?positiveUnivariateDistributionQ, data_?validNoiseOrConvDataQ}]
		],
			"Noise" -> res (* Replace[x, a_?Audio`AudioStreamQ :> File[Audio`AudioInformation[a, "ResourcePath"]], 2] *)
		,
		True,
			FailCoder[
				"`` is not a valid value for \"Noise\". Possible values include None, \
a univariate distribution, a list of a univariate distribution and an Audio object.",
				x /. _Audio ->"Audio[<>]"
			]
	]
]

CheckAudioAugmentation["Convolution" -> x_, _] := Scope[
	res = Replace[x, Last[toUniformDistributionPatterns]];
	Which[
		Or[
			res === None,
			MatchQ[res, data_?validNoiseOrConvDataQ],
			MatchQ[res, {v_?Internal`RealValuedNumericQ, data_?validNoiseOrConvDataQ} /; v > 0],
			MatchQ[res, {v_?positiveUnivariateDistributionQ, data_?validNoiseOrConvDataQ}]
		],
			"Convolution" -> res (* Replace[x, a_?Audio`AudioStreamQ :> File[Audio`AudioInformation[a, "ResourcePath"]], 2], *)
		,
		True,
			FailCoder[
				"`` is not a valid value for \"Convolution\". Possible values include \
None and a list of a univariate distribution and an Audio object.",
				x /. _Audio ->"Audio[<>]"
			]
	]
]

CheckAudioAugmentation["VTLP" -> x_, "AudioMelSpectrogram"|"AudioMFCC"] := Scope[
	res = Replace[x, First[toUniformDistributionPatterns]];
	Which[
		Or[
			res === None,
			MatchQ[res, {a1_, a2_, a3_} /; VectorQ[{a1, a2, a3}, Internal`RealValuedNumericQ] && a3 > a2 >= 0 && a1 >= 0],
			MatchQ[res, {a1_?positiveUnivariateDistributionQ, a3_, a4_} /; VectorQ[{a3, a4}, Internal`RealValuedNumericQ] && a4 > a3 >= 0]
		],
			"VTLP" -> res
		,
		Or[
			Internal`RealValuedNumericQ[res],
			positiveUnivariateDistributionQ[res]
		],
			"VTLP" -> {res, Automatic, Automatic}
		,
		True,
			FailCoder[
				"`` is not a valid value for \"VTLP\". Possible values include None, \
a real positive number or a univariate distribution.",
				x
			]
	]
]
CheckAudioAugmentation["VTLP" -> _, feat_] :=
	FailCoder["\"VTLP\" is not a valid augmentation for the `` NetEncoder.", feat]

CheckAudioAugmentation[x_ -> y_, _] := FailCoder["`` is not a valid augmentation.", x]
CheckAudioAugmentation[x___] := FailCoder["`` is not a valid augmentation.", x]