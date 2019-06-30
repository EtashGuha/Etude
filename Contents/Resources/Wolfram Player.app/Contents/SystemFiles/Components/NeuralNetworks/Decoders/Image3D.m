Input: InterleavingSwitchedT[$Interleaving, $$Channels, $$Dimensions]

Parameters:
	$ColorSpace: Defaulting[ColorSpaceT, "RGB"]
	$Interleaving: Defaulting[BooleanT, False]
	$MeanImage: Defaulting[Nullable[EitherT[{ScalarT, ListT[SizeT, ScalarT], Image3DT[]}]]]
	$VarianceImage: Defaulting[Nullable[EitherT[{ScalarT, ListT[SizeT, ScalarT], Image3DT[]}]]]
	$$Dimensions: SizeListT[3]
	$$Channels: SizeT

Upgraders: {
	"11.2.0" -> Append["VarianceImage" -> None],
	"11.3.9" -> Append["Interleaving" -> False]
}

DecoderToEncoder: Function @ Scope[
	dims = TDimensions[#2];
	{"Image3D", 
		"ImageSize" -> Reverse[Rest[dims]], "ColorChannels" -> First[dims],
		"Interleaving" -> #Interleaving,
		"ColorSpace" -> #ColorSpace, "MeanImage" -> #MeanImage,
		"VarianceImage" -> #VarianceImage
	}
]

ArrayDepth: 4

ToDecoderFunction: Function @ Scope @ With[
	{cspace = #ColorSpace, mean = #MeanImage, var = #VarianceImage, inter = #Interleaving},
	meaner = Switch[mean, 
		None, Identity,
		_Image3D, ImageAdd[#, ImageResize[mean, ImageDimensions[#]]]&,
		_, ImageAdd[#, mean]&
	];
	func = Function[input, 
		If[cspace === "RGB" && Length[input] =!= 4,
			ColorConvert[Image3D[input, Interleaving -> inter], "RGB"],
			Image3D[input, ColorSpace -> cspace, Interleaving -> inter]
		]
	] /* meaner;
	Map[func]
]


