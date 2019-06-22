Output: SequenceT[$$TargetLength, VectorT[$$Size]]

Parameters:
	$SampleRate: Defaulting[PosIntegerT, 16000]
	$Normalization: AudioNormalizationT
	$WindowSize: Defaulting[EitherT[{PosIntegerT, DurationT}], HeldDefault @ Quantity[25, "Milliseconds"]]
	$Offset: Defaulting[EitherT[{PosIntegerT, DurationT}], HeldDefault @ Quantity[25/3., "Milliseconds"]]
	$WindowFunction: WindowFunctionT
	$MinimumFrequency: Defaulting[PosIntegerT, Automatic]
	$MaximumFrequency: Defaulting[PosIntegerT, Automatic]
	$NumberOfFilters: Defaulting[PosIntegerT, 40]
	$Augmentation: ValidatedParameterT[CheckAudioAugmentation[#, "AudioMelSpectrogram"]&, None]
	$$PreEmphasis: Defaulting[Nullable[IntervalScalarT[0, 1]], None]

	$TargetLength: AudioTargetLengthT
	$$TargetLength: InternalAudioTargetLengthT[$TargetLength]
	$$Size: ComputedType[SizeT, $NumberOfFilters]


AllowBypass: Function[True]

PostInferenceFunction: Function @ Scope[
	winsize = TimeToSamples[$WindowSize, $SampleRate];
	CheckWindowFunction[$WindowFunction, winsize];
	If[!IntegerQ[$MinimumFrequency], PostSet[$MinimumFrequency, Ceiling[$SampleRate / winsize]]];
	If[!IntegerQ[$MaximumFrequency], PostSet[$MaximumFrequency, Round @ Min[8000, $SampleRate / 2.]]];

	(* validation *)
	If[$MaximumFrequency > $SampleRate/2,
		FailValidation[NetEncoder, "the \"MaximumFrequency\" must be smaller than half the \"SampleRate\"."]
	];
	If[$MinimumFrequency >= $MaximumFrequency,
		FailValidation[NetEncoder, "the \"MinimumFrequency\" must be smaller than the \"MaximumFrequency\"."]
	];
	If[$NumberOfFilters >= winsize,
		FailValidation[NetEncoder, "the \"WindowSize\" cannot be smaller than the \"NumberOfFilters\"."]
	];
	If[QuantityQ[$TargetLength],
		length = Ceiling[TimeToSamples[$TargetLength, $SampleRate]/TimeToSamples[$Offset, $SampleRate]];
		PostSet[$$TargetLength, length];
	];

	RestartInference[];
]

ToEncoderFunction: Function @ ModuleScope[
	encParams = GenerateSFTParameters[#1, #2, "AudioMelSpectrogram"];
	audioFeatureExtract[#, "MelSpectrogram", encParams]&
]

MLType: Function["Expression"]