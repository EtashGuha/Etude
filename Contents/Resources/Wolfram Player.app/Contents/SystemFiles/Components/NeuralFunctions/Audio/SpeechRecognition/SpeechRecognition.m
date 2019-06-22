(* ::Package:: *)

PackageExport["SpeechRecognize"]


(* ::Section:: *)
(*SpeechRecognize*)


Options[SpeechRecognize] =
SortBy[
	{
		Masking -> All,
		TargetDevice -> "CPU"
	},
	ToString
];


SpeechRecognizeHiddenOptions ={
	"Caching" -> False,
	"SplitEvaluation" -> True,
	"LanguageModel" -> True,
	"BeamSize" -> Automatic,
	"ProgressReporting" -> Automatic
};


DefineFunction[
	SpeechRecognize,
	iSpeechRecognize,
	1,
	"ExtraOptions" -> SpeechRecognizeHiddenOptions
]


iSpeechRecognize[args_, opts_]:=
Module[
	{
		a, level, prop, masking, lm, beamSize, splitQ, progressReportingQ, targetDevice,
		res
	},
	(*-- argument parsing --*)
	(*
		(* support for batched eval of list of audios *)
		If[ListQ[args[[1]]],
			a = Audio`Utilities`checkAudio[#, SpeechRecognize]& /@ args[[1]];
			,
			a = Audio`Utilities`checkAudio[args[[1]], SpeechRecognize];
		];
		(* Support for input AudioStream *)
	*)
	a = Catch@Audio`Utilities`checkAudio[args[[1]], SpeechRecognize];
	If[a === $Failed,
		ThrowFailure[];
	];
	level = Automatic;
	prop = {"Text"};
	progressReportingQ = MatchQ[GetOption["ProgressReporting"], Automatic|True];

	(*-- option parsing --*)
	masking = GetOption[Masking];
	$Caching = TrueQ[GetOption["Caching"]];
	splitQ = TrueQ[GetOption["SplitEvaluation"]];
	lm = TrueQ[GetOption["LanguageModel"]];
	beamSize = GetOption["BeamSize"];
	If[!(Internal`PositiveIntegerQ[beamSize] || beamSize === Automatic),
		ThrowFailure["beam", beamSize];
	];
	targetDevice = GetOption["TargetDevice"];
	If[!NeuralNetworks`TestTargetDevice[targetDevice, SpeechRecognize], ThrowFailure[];];

	a = maskAudio[a, masking];
	splitQ = splitQ && Audio`Utilities`Duration[a] >= $splitSize;

	(*
		prepare the net. This way the download process
		is not included in the recognition progress bar
	*)
	getDeepSpeech[];

	(* Cached *)
	res = oSpeechRecognize[a, level, prop, lm, beamSize, targetDevice, splitQ, progressReportingQ];

	If[Length[prop] === 1,
		res[prop[[1]]]
		,
		Transpose[Lookup[res, prop]]
	]
]


(* ::Subsection::Closed:: *)
(*oSpeechRecognize*)


(* ::Subsubsection::Closed:: *)
(*No splitting*)


oSpeechRecognize[a_, level_, prop_, lm_, beamSize_, targetDevice_, False, progressReportingQ_] :=
Cached@Module[
	{t, res},

	t = 1.1 * estimateEvaluationTime[Audio`Utilities`Duration[a], targetDevice];

	res = showRecognitionProgress[a, t, targetDevice, lm, beamSize, progressReportingQ];

	postProcessResult[res, None, prop, level, Audio`Utilities`Duration[a]]
]


(* ::Subsubsection::Closed:: *)
(*With splitting*)


oSpeechRecognize[a_, level_, prop_, lm_, beamSize_, targetDevice_, True, progressReportingQ_] :=
Cached@Module[
	{
		windowSize, offset, length, sampleRate,
		t, res
	},

	sampleRate = Audio`Utilities`AudioSampleRate[a];
	windowSize = Round[sampleRate * $splitSize];
	offset = Round[sampleRate * ($splitSize - $splitOverlap)];
	length = Audio`Utilities`AudioLength[a];
	t = 1.2 * (Floor[Max[0, (length - windowSize)] / offset] + 2) * estimateEvaluationTime[$splitSize, targetDevice];

	res = showRecognitionProgress[a, t, targetDevice, lm, beamSize, progressReportingQ, windowSize, offset, length, sampleRate, True];

	postProcessResult[res, None, prop, level, Audio`Utilities`Duration[a]]
]


(* ::Subsection::Closed:: *)
(*Progress panel*)


showRecognitionProgress[a_, t_, targetDevice_, lm_, beamSize_, progressReportingQ_] :=
	showRecognitionProgress[a, t, targetDevice, lm, beamSize, progressReportingQ, None, None, None, None, False]


showRecognitionProgress[a_, t_, targetDevice_, lm_, beamSize_, progressReportingQ_, windowSize_, offset_, length_, sampleRate_] :=
	showRecognitionProgress[a, t, targetDevice, lm, beamSize, progressReportingQ, windowSize, offset, length, sampleRate, True]


showRecognitionProgress[a_, t_, targetDevice_, lm_, beamSize_, progressReportingQ_, windowSize_, offset_, length_, sampleRate_, chunkedQ_] /;
	t > 2 && $Notebooks && progressReportingQ && TrueQ[GeneralUtilities`$ComputeWithProgressEnabled] :=
Module[
	{acousticModelResult, res, tempText},
	DynamicModule[
		{languageModel = False, lmIndex = 0, text = ""},
		Monitor[
			If[chunkedQ,
				acousticModelResult = getAcousticModelResultChunked[text, a, windowSize, offset, length, sampleRate, targetDevice];
				,
				acousticModelResult = getAcousticModelResult[preprocessAudio[a], targetDevice];
			];
			languageModel = True;
			res = decodeResult[(lmIndex = N@#) &, lm, acousticModelResult, beamSize, targetDevice, None];
			,
			tempText = If[chunkedQ, text, None];
			makePanel[t, languageModel, lmIndex, tempText]
		]
	];
	res
]


showRecognitionProgress[a_, t_, targetDevice_, lm_, beamSize_, progressReportingQ_, windowSize_, offset_, length_, sampleRate_, chunkedQ_] /;
	t > 2 && !$Notebooks && progressReportingQ && TrueQ[GeneralUtilities`$ComputeWithProgressEnabled] :=
Module[
	{acousticModelResult, text},
	If[chunkedQ,
		tempRes = "";
		acousticModelResult = GeneralUtilities`ComputeWithProgress[
			Function[
				pf,
				getAcousticModelResultChunked[text, a, windowSize, offset, length, sampleRate, targetDevice, pf]
			],
			"Running the speech recognizer...",
			"ReportingForm" -> "Print",
			"MinimumProgressTime" -> 0,
			"UpdateInterval" -> 5,
			"MaximumIndicatorTime" -> 0,
			"MinimumDetailsTime" -> Infinity
		];
		,
		Print["Running the speech recognizer..."];
		acousticModelResult = getAcousticModelResult[preprocessAudio[a], targetDevice];
	];
	GeneralUtilities`ComputeWithProgress[
		decodeResult[#, lm, acousticModelResult, beamSize, targetDevice, None]&,
		"Finalizing the language output...",
		"ReportingForm" -> "Print",
		"MinimumProgressTime" -> 0,
		"UpdateInterval" -> 5,
		"MaximumIndicatorTime" -> 0,
		"MinimumDetailsTime" -> Infinity
	]
]


showRecognitionProgress[a_, t_, targetDevice_, lm_, beamSize_, progressReportingQ_, windowSize_, offset_, length_, sampleRate_, chunkedQ_] /;
	t <= 2 || !progressReportingQ || !TrueQ[GeneralUtilities`$ComputeWithProgressEnabled] :=
Module[
	{acousticModelResult, text},
	If[chunkedQ,
		acousticModelResult = getAcousticModelResultChunked[text, a, windowSize, offset, length, sampleRate, targetDevice];
		,
		acousticModelResult = getAcousticModelResult[preprocessAudio[a], targetDevice];
	];
	decodeResult[None, lm, acousticModelResult, beamSize, targetDevice, None]
]


(* ::Subsubsection::Closed:: *)
(*makePanel*)


SetAttributes[makePanel, HoldAll];
makePanel[estimate_, languageModel_, lmIndex_, text_:None] :=
DynamicModule[
	{estimatedTime = estimate, max = estimate * 20, t = 0},
	Overlay[{
		GeneralUtilities`InformationPanel[
			"Recognition Progress",
			{
				Left :> PaneSelector[
					{
						True -> "Finalizing language output...",
						False -> "Running speech recognizer..."
					},
					Dynamic[languageModel, TrackedSymbols :> {languageModel}]
				],
				Center :> PaneSelector[
					{
						True -> ProgressIndicator[Dynamic[lmIndex, TrackedSymbols :> {lmIndex}], {0, 1}, ImageSize -> Scaled[.97]],
						False -> ProgressIndicator[Dynamic[t, TrackedSymbols :> {t}], {0, estimatedTime}, ImageSize -> Scaled[.97]]
					},
					Dynamic[languageModel, TrackedSymbols :> {languageModel}]
				],
				"time elapsed" :> Dynamic[
					RawBoxes[
						FEPrivate`Which[
							t >= 3600, RowBox[{FEPrivate`Floor[t / 3600], "h", FEPrivate`Mod[FEPrivate`Floor[t / 60], 60], "m"}],
							t >= 300, RowBox[{FEPrivate`Floor[t / 60], "m"}],
							t >= 60, RowBox[{FEPrivate`Floor[t / 60], "m", FEPrivate`Round[FEPrivate`Mod[t, 60]], "s"}],
							t < 60,RowBox[{FEPrivate`Round[t, 1], "s"}],
							t == 0., RowBox[{0, "s"}]
						]
					](* ,
					TrackedSymbols :> {t} *)
				],
				If[text =!= None,
					"raw result" :> Dynamic[Style[text, Italic, Gray], TrackedSymbols :> {text}],
					Nothing
				]
			},
			ColumnsEqual -> False,
			UpdateInterval-> 1,
			ColumnWidths -> {7.5, 28},
			LineBreakWithin -> Automatic
		],
		Animator[
			Dynamic[t],
			{0, max, .5},
			DefaultDuration -> max,
			AnimationRepetitions -> 1,
			AppearanceElements -> {},
			ImageSize -> {1, 1}
		]
	}]
]


showTemporaryText["Snippet", sentence_] :=
If[sentence === "",
	"..."
	,
	"..." <>
	StringTake[sentence, - Min[StringLength[sentence], 200]] <>
	"..."
]
showTemporaryText["Full", sentence_] :=
If[sentence === "",
	"..."
	,
	sentence <> "..."
]


(* ::Subsection::Closed:: *)
(*getAcousticModelResult*)


getDeepSpeech[] :=
getDeepSpeech[] =
NetReplacePart[
	GetNetModel["Deep Speech 2 Trained on Baidu English Data"],
	{
		"Output" -> None,
		"Input" -> None
	}
];


getAcousticModelResult[features_, targetDevice_] :=
Cached@With[
	{
		res = AbsoluteTiming[
			SafeNetEvaluate[
				getDeepSpeech[][NumericArray[features, "Real32"], TargetDevice -> targetDevice],
				NumericArrayQ
			]
		]
	},
	saveTiming[targetDevice, {Length[features], res[[1]]}];
	Last[res]
]


SetAttributes[getAcousticModelResultChunked, HoldFirst];
getAcousticModelResultChunked[text_, a_, windowSize_, offset_, length_, sampleRate_, targetDevice_, pf_:None] :=
Module[
	{acousticModelResult, tempRes = "", currentChunk = 0, numberOfChunks},
	numberOfChunks = Floor[Max[0, (length - windowSize)] / offset] + 2;
	acousticModelResult = Audio`Developer`AudioBlockMap[
		With[
			{r = getAcousticModelResult[preprocessAudio[Audio[#, SampleRate -> sampleRate]], targetDevice]},
			pf[(currentChunk += 1)/numberOfChunks];
			tempRes = tempRes <> " " <> StringJoin[decoder[10][r]];
			text = showTemporaryText["Snippet", tempRes];
			r
		] &,
		a,
		windowSize,
		offset,
		None,
		"Channels" -> "Mean",
		Padding -> None
	];
	If[(Length[acousticModelResult] - 1) * offset + windowSize < length,
		acousticModelResult = Append[
			acousticModelResult,
			getAcousticModelResult[
				preprocessAudio[
					Audio[
						Audio`InternalAudioData[a, {((Length[acousticModelResult] - 1) * offset) + offset + 1, length}],
						SampleRate -> sampleRate
					]
				],
				targetDevice
			]
		];
	];
	pf[(currentChunk += 1)/numberOfChunks];
	tempRes = tempRes <> " " <> StringJoin[decoder[10][Last[acousticModelResult]]];
	text = showTemporaryText["Snippet", tempRes];
	NumericArray[
		joinProcessedChunks[
			Normal@acousticModelResult,
			34 * $splitOverlap,
			trans[#, .2]&
		],
		"Real32"
	]
]


(* ::Subsection::Closed:: *)
(*decodeResult*)


decodeResult[pf_, lmQ_, acousticModelResult_, beamSize_, targetDevice_, tempRes_] :=
Module[
	{temporaryDecoding, res},
	res = If[lmQ,
		With[
			{languageModel = getLanguageModel[]},
			resetCounter[];
			getLanguageModelResult[pf, Length[acousticModelResult], targetDevice, languageModel, acousticModelResult, beamSize]
		]
		,
		pf[0.1];
		StringJoin[decoder[beamSize][acousticModelResult]]
	];
	res
];


(* ::Subsection::Closed:: *)
(*CTC decoder*)


$alphabet = Characters@"' abcdefghijklmnopqrstuvwxyz";


decoder[beamSize_] :=
	decoder[beamSize] =
		NetDecoder[{"CTCBeamSearch", $alphabet, If[beamSize =!= Automatic, "BeamSize" -> beamSize, Nothing]}]


(* ::Subsection::Closed:: *)
(*getLanguageModelResult*)


getLanguageModelResult[pf_, length_, targetDevice_, languageModel_, acousticModelResult_, beamSize_] :=
Cached@StringJoin[
	applyLanguageModel[pf, length, targetDevice, languageModel, acousticModelResult, beamSize /. Automatic -> 10]
]


(* ::Subsubsection::Closed:: *)
(*getLanguageModel*)


getLanguageModel[] := getLanguageModel[] = Block[
	{net},
	net = GetNetModel["SpeechRecognizeCharactersLevelLanguageModelTrainedOnTeds_v1"];
	net = NetGraph[
		NetExtract[net,{{1, 1}, {1, 2}, {1, 3}, {2}, {3}, {4}}],
		{
			1 -> 2 -> 3 -> 4 -> 5 -> 6,
			NetPort[ "State2In"] -> NetPort[2, "State"],
			NetPort[ "State3In"] -> NetPort[3, "State"],
			NetPort[2, "State"] -> NetPort["State2Out"],
			NetPort[3, "State"] -> NetPort["State3Out"]
		}
		];
	net
];


(* ::Subsubsection::Closed:: *)
(*applyLanguageModel*)


applyLanguageModel[pf_, length_, targetDevice_, languageModel_, likelihood_, beamSize_, postProcessoOptions_:1] :=
Module[
	{res},
	Needs["NumericArrayUtilities`"];
	res = Quiet@NumericArrayUtilities`CTCBeamSearchDecode2[
		likelihood,
		False,
		-1,
		<|
			"LanguageModelFun" -> languageModelRestrictedFunction[pf, length, targetDevice, languageModel],
			"EmptyString" -> 62,
			"MixtureFun" -> Function[{str, probCTC, probLM}, probCTC + probLM * .4],
			"MixtureLevel" -> .4
		|>,
		"BeamSize" -> beamSize,
		"PostProcessOptions" -> <|"UpTo"-> postProcessoOptions|>
	][[1, "Text"]];
	If[VectorQ[res, StringQ],
		res
		,
		ThrowFailure["interr", "The CTC decoding failed"];
	]
]


(* ::Subsubsection::Closed:: *)
(*restrictedVocabulary*)


restrictedVocabulary[probabilities_] :=
Join[
	probabilities[[1 ;; 2]],
	probabilities[[3 ;; 28]] + probabilities[[29 ;; 54]]
]/(1 - Total[probabilities[[55;;]]])


(* ::Subsubsection::Closed:: *)
(*languageModelRestrictedFunction*)


Module[
	{counter = 0},
	resetCounter[] := (counter = 0);
	languageModelRestrictedFunction[pf_, length_, targetDevice_, languageModel_][beamsStates_, beamsStrUpdate_]:=
		Map[
			{
				Log@restrictedVocabulary[#Output],
				KeyMap[
					<|
					"State2Out" -> "State2In"
					,"State3Out"-> "State3In"
					|>,
					KeyDrop[#, "Output"]
				]
			} &,
			Transpose[
				Map[
					Normal,
					languageModel[
						(
							pf[(counter += 1)/length];
							<|
								"Input" -> beamsStrUpdate,
								Transpose[
									beamsStates /.
										None -> <|"State2In" -> ConstantArray[0., 256],"State3In"-> ConstantArray[0., 256]|>,
									AllowedHeads -> All
								]
							|>
						),
						{"Output", "State2Out", "State3Out"},
						TargetDevice -> targetDevice
					]
				],
				AllowedHeads -> All
			]
		]
]


(* ::Subsection::Closed:: *)
(*postProcessResult*)


postProcessResult[in_, acousticModelResult_, {"Text"}, Automatic, _] := <|"Text" -> in|>
postProcessResult[in_, acousticModelResult_, prop_, level_, dur_] :=
Module[
	{	wordBoundaries,
		res = <|"Text" -> None, "Interval"-> None, (* "Strength" -> None, *) "Audio" -> None|>
	},
	res["Text"] = in;
	If[MemberQ[prop, "Audio"|"Interval"],
		wordBoundaries = getWordBoundaries[acousticModelResult, res["Text"], Audio`Utilities`Duration[in]]
	];
	Which[
		level === Automatic,
			If[MemberQ[prop, "Interval"],
				res["Interval"] = {wordBoundaries[[1, 1]], wordBoundaries[[-1, 2]]};
			];
			If[MemberQ[prop, "Audio"],
				res["Audio"] = in;
			];
		,
		level === "Word",
			res["Text"] = TextCases[Evaluate[res["Text"]], "Word"];
			res["Interval"] = wordBoundaries;
			If[MemberQ[prop, "Audio"],
				res["Audio"] = AudioTrim[in, res["Interval"]];
			];
	];
	res
]


(* ::Subsection::Closed:: *)
(*findSplittingPoints*)


findSilences[features_, threshold_:80] :=
Module[
	{amp, silences},
	amp = Thread[{Range[0, Length@features-1], Map[Total, features]}];
	silences = Select[
		SplitBy[Map[{#[[1]], If[#[[2]] < threshold, 1, 0]}&, amp], Last],
		#[[1, 2]] === 1&
	];
	{#[[1, 1]], #[[-1, 1]]} & /@silences
]


findSplittingPoints[silences_, length_] := Module[
	{splittingPoints},
	splittingPoints = {};
	Quiet@If[-Subtract@@# > 50 && (#[[1]] > splittingPoints[[-1, 2]] + 300 || #[[1]] > 100),
		AppendTo[splittingPoints,#]
	]& /@ silences;
	If[splittingPoints =!= {},
		splittingPoints = Partition[Append[Prepend[Round[Mean /@ splittingPoints], 1], length], 2, 1];
	];
	splittingPoints
]



(* ::Subsection::Closed:: *)
(*Word Boundaries*)


indexToTime[i_] := (i - 1) * .03


timeToIndex[t_] := Round[t/.03 + 1]


getWordBoundaries[likelihood_, transcription_, dur_:Infinity] :=
Module[{spaces},
	spaces = Partition[
		Append[
			Prepend[
				Map[
					Mean[indexToTime[#]] &,
					Select[
						SplitBy[
							Thread[{Range[Length@#], #} &[First@Ordering[#, -1] & /@ likelihood]],
							Last
						],
						#[[1, 2]] === 2 &
					][[All, All, 1]]
				],
				0.
			],
			indexToTime[Length@likelihood]
		],
		2, 1
	];
	spaces
]


(* ::Subsection::Closed:: *)
(*Utilities*)


(* ::Subsubsection::Closed:: *)
(*maskAudio*)


maskAudio[a_, masking_] :=
Module[
	{maskedAudio = a},
	If[!MatchQ[masking, All|Automatic],
		Quiet[
			maskedAudio = AudioTrim[maskedAudio, masking];
			If[ListQ[maskedAudio],
				maskedAudio = AudioJoin[maskedAudio];
			];
		];
		If[!AudioQ[maskedAudio],
			ThrowFailure["msk", masking];
		];
	];
	maskedAudio
]



(* ::Subsubsection::Closed:: *)
(*preprocessAudio*)


preprocessAudio[audio_Audio] :=
Module[
	{a = audio, power, res},
	Quiet[
		If[AudioChannels[a] =!= 1,
			a = Mean[AudioChannelSeparate[a]];
		];
		If[Audio`Utilities`AudioSampleRate[a] =!= 16000,
			a = AudioResample[a, 16000, Resampling -> {"Hermite", 2}];
		];
		With[
			{length = Audio`Utilities`AudioLength[a]},
			If[length < 320,
				a = AudioPad[a, (320 - length)/16000]
			];
		];
		power = AudioMeasurements[a, "Power"];
		If[power =!= .01,
			a = AudioAmplify[
				a,
				Sqrt[.01/Max[power, 10.^-7]]
			];
		];
		res = Internal`AbsSquare[SpectrogramArray[a, 320, 160, HannWindow][[All, ;; 161]]];
	];
	If[!Developer`PackedArrayQ[res],
		ThrowFailure["interr", "the preprocessing was not able to produce a valid Audio object"];
		,
		res
	]
]


(* ::Subsubsection::Closed:: *)
(*Estimate time and size*)


getCachedTimings[dev_] /; MatchQ[dev, "CPU"|"GPU"|{"GPU", _}] :=
With[
	{data = Quiet[Get @ LocalObject["SpeechRecognize/Performance" <> ToString[dev]]]},
	If[data === $Failed,
		{}
		,
		data
	]
]


Module[
	{counter = <|"CPU" -> 0, "GPU" -> 0|>},
	saveTiming[dev_, {length_?Internal`PositiveIntegerQ, time_?Internal`RealValuedNumericQ}] /;
		MatchQ[dev, "CPU"|"GPU"|{"GPU", _}] :=
	(
		counter[dev /. {"GPU", _} -> "GPU"] += 1;
		If[counter[dev /. {"GPU", _} -> "GPU"] > 1,
			With[
				{data = getCachedTimings[dev]},
				Put[
					Append[
						If[Length[data] > 10000,
							RandomSample[data, 10000],
							data
						],
						{length, time}
					],
					LocalObject["SpeechRecognize/Performance" <> ToString[dev]]
				]
			]
		]
	)
]


estimateEvaluationTime[dur_, dev_]  /; MatchQ[dev, "CPU"|"GPU"|{"GPU", _}] :=
Module[
	{data, fit, x},
	data = getCachedTimings[dev];
	If[Length[DeleteDuplicates @ data[[All, 1]]] > 3,
		fit = LinearModelFit[data, x, x];
		Max[Replace[Normal[fit], x -> dur*100, Infinity], .1]
		,
		dur / 2.
	]
]


estimateInputByteCount[dur_] := dur * 16100 * 8


(* ::Subsubsection::Closed:: *)
(*Constants*)


PackageScope["SpeechRecognize`$SetChunkingSizeAndOverlap"]

SpeechRecognize`$SetChunkingSizeAndOverlap[] := (
	$splitSize = 3 * 60;
	$splitOverlap = 3;
	{$splitSize, $splitOverlap}
);

SpeechRecognize`$SetChunkingSizeAndOverlap[size_?Internal`RealValuedNumericQ] /;
	size > $splitOverlap + 2 :=
(
	$splitSize = size;
	{$splitSize, $splitOverlap}
)
SpeechRecognize`$SetChunkingSizeAndOverlap[size_?Internal`RealValuedNumericQ, overlap_?Internal`RealValuedNumericQ] /;
	size > overlap + 2 && overlap >= 1 :=
(
	$splitSize = size;
	$splitOverlap = overlap;
	{$splitSize, $splitOverlap}
)
SpeechRecognize`$SetChunkingSizeAndOverlap[___] := $Failed

(* Initialize $splitSize and $splitOverlap to the default values *)
SpeechRecognize`$SetChunkingSizeAndOverlap[];


(* ::Subsubsection::Closed:: *)
(*validPropertyQ*)


validProperties = {"Text", "Interval", (* "Strength", *) "Audio"};


validPropertyQ[Alternatives@@validProperties] := True
validPropertyQ[l_List] := VectorQ[l, validPropertyQ]
validPropertyQ[___] := False


(* ::Subsubsection::Closed:: *)
(*validLevelQ*)


validLevels = {Automatic, "Word", "Sentence"};


validLevelQ[Alternatives@@validLevels] := True
validLevelQ[___] := False
