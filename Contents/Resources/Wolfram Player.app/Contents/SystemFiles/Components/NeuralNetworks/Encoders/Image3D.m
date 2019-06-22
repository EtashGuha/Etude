Output: InterleavingSwitchedT[$Interleaving, $ColorChannels, ComputedType[SizeListT[], Reverse @ $ImageSize]]

Parameters:
	$ImageSize: NormalizedT[SizeListT[3], rep3d, {128, 128, 128}]
	$ColorSpace: ComputedType[ColorSpaceT, toColorSpace[$ColorSpace, $ColorChannels], {$ColorChannels}]
	$ColorChannels: ComputedType[SizeT, toImageChannelCount[$ColorSpace]]
	$Interleaving: Defaulting[BooleanT, False]
	$MeanImage: Defaulting[Nullable[EitherT[{ScalarT, ListT[$ColorChannels, ScalarT], Image3DT[]}]]]
	$VarianceImage: Defaulting[Nullable[EitherT[{ScalarT, ListT[$ColorChannels, ScalarT], Image3DT[]}]]]

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
		PostSet[NetPath["Output"], TensorT @ imgDims];
		,
		validateVarianceImage[$VarianceImage, $ImageSize, $ColorSpace, $ColorChannels, $Interleaving];
	];
]

rep3d[i_Integer] := {i, i, i};
rep3d[i:{_Integer,_Integer, _Integer}] := i;
rep3d[_] := $Failed;

AllowBypass: Function[True]

ToEncoderFunction: Function[
	Map @ makeWLImageEncoderFunction[#]
]

TypeRandomInstance: Function[
	RandomImage[{0, 1}, ImageSize -> #ImageSize, ColorSpace -> #ColorSpace]
]

MLType: Function["Expression"]

EncoderToDecoder: Function[
	{"Image3D",
		"ColorSpace" -> #ColorSpace, "MeanImage" -> #MeanImage, 
		"VarianceImage" -> #VarianceImage,
		"$Channels" -> #ColorChannels, 
		"$Dimensions" -> Reverse[#ImageSize]
	}
]