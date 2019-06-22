Output: InterleavingSwitchedT[$Interleaving, $ColorChannels, ComputedType[SizeListT[], Reverse @ $ImageSize]]

Parameters:
	$ImageSize: NormalizedT[SizeListT[2], rep2d, {128, 128}]
	$ColorSpace: ComputedType[ColorSpaceT, toColorSpace[$ColorSpace, $ColorChannels], {$ColorChannels}]
	$ColorChannels: ComputedType[SizeT, toImageChannelCount[$ColorSpace]]
	$Interleaving: Defaulting[BooleanT, False]
	$MeanImage: Defaulting[Nullable[EitherT[{ScalarT, ListT[$ColorChannels, ScalarT], ImageT[]}]]]
	$VarianceImage: Defaulting[Nullable[EitherT[{ScalarT, ListT[$ColorChannels, ScalarT], ImageT[]}]]]

Upgraders: {
	"11.2.0" -> Append["VarianceImage" -> None],
	"11.3.9" -> Append["Interleaving" -> False]
}

PostInferenceFunction: Function @ Scope[
	If[Not @ IntegerQ[$ColorChannels],
		PostSet[$ColorSpace, "RGB"];
		PostSet[$ColorChannels, 3];
		imgSize = $ImageSize;
		validateVarianceImage[$VarianceImage, imgSize, "RGB", 3, $Interleaving];
		imgDims = toImageOutputDims[$Interleaving, 3, imgSize];
		PostSet[NetPath["Output"], TensorT[imgDims]];
		,
		validateVarianceImage[$VarianceImage, $ImageSize, $ColorSpace, $ColorChannels, $Interleaving];
	];
]

rep2d[i_Integer] := {i, i};
rep2d[i:{_Integer, _Integer}] := i;
rep2d[_] := $Failed;

AllowBypass: Function[True]

ToEncoderFunction: Function[
	makeOpenCVImageEncoderFunction[#, makeWLImageEncoderFunction @ #]
]

TypeRandomInstance: Function[
	RandomImage[{0, 1}, ImageSize -> #ImageSize, ColorSpace -> #ColorSpace]
]

MLType: Function["Image"]

EncoderToDecoder: Function[
	{"Image",
		"ColorSpace" -> #ColorSpace, 
		"Interleaving" -> #Interleaving,
		"MeanImage" -> #MeanImage, 
		"VarianceImage" -> #VarianceImage,
		"$Channels" -> #ColorChannels, 
		"$Dimensions" -> Reverse[#ImageSize]
	}
]