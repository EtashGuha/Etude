Output: SequenceT[$$TargetLength, VectorT[1]]

Parameters:
	$SampleRate: Defaulting[PosIntegerT, 16000]
	$Normalization: AudioNormalizationT
	$Augmentation: ValidatedParameterT[CheckAudioAugmentation[#, "Audio"]&, None]
	$$PreEmphasis: Defaulting[Nullable[IntervalScalarT[0, 1]], None]

	$TargetLength: AudioTargetLengthT
	$$TargetLength: InternalAudioTargetLengthT[$TargetLength]

AllowBypass: Function[True]

PostInferenceFunction: Function @ Scope[
	If[QuantityQ[$TargetLength],
		length = TimeToSamples[$TargetLength, $SampleRate];
		PostSet[$$TargetLength, length];
	];
	RestartInference[];
]

ToEncoderFunction: Function @ ModuleScope[
	encParams = GenerateSFTParameters[#1, #2, "Audio"];
	audioFeatureExtract[#, "AudioData", encParams]&
]

MLType: Function["Expression"]