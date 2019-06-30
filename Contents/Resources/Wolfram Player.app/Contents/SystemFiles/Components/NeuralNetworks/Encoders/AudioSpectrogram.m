Output: SequenceT[$$TargetLength, VectorT[$$Size]]

Parameters:
	$SampleRate: Defaulting[PosIntegerT, 16000]
	$Normalization: AudioNormalizationT
	$WindowSize: Defaulting[EitherT[{PosIntegerT, DurationT}], HeldDefault @ Quantity[25, "Milliseconds"]]
	$Offset: Defaulting[EitherT[{PosIntegerT, DurationT}], HeldDefault @ Quantity[25/3., "Milliseconds"]]
	$WindowFunction: WindowFunctionT
	$Augmentation: ValidatedParameterT[CheckAudioAugmentation[#, "AudioSpectrogram"]&, None]
	$$PreEmphasis: Defaulting[Nullable[IntervalScalarT[0, 1]], None]

	$TargetLength: AudioTargetLengthT
	$$TargetLength: InternalAudioTargetLengthT[$TargetLength]
	$$Size: SizeT

AllowBypass: Function[True]

PostInferenceFunction: Function @ Scope[
	winsize = TimeToSamples[$WindowSize, $SampleRate];
	CheckWindowFunction[$WindowFunction, winsize];
	PostSet[$$Size, Floor[(winsize / 2.) + 1]];
	If[QuantityQ[$TargetLength],
		length = Ceiling[TimeToSamples[$TargetLength, $SampleRate]/TimeToSamples[$Offset, $SampleRate]];
		PostSet[$$TargetLength, length];
	];
	RestartInference[];
]

ToEncoderFunction: Function @ ModuleScope[
	encParams = GenerateSFTParameters[#1, #2, "AudioSpectrogram"];
	audioFeatureExtract[#, "Spectrogram", encParams]&
]

MLType: Function["Expression"]

