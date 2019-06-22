Input: InterleavingSwitchedT[$Interleaving, $$Channels, $$Dimensions]

Parameters:
	$ColorSpace: Defaulting[ColorSpaceT, "RGB"]
	$Interleaving: Defaulting[BooleanT, False]
	$MeanImage: Defaulting[Nullable[EitherT[{ScalarT, ListT[SizeT, ScalarT], ImageT[]}]]]
	$VarianceImage: Defaulting[Nullable[EitherT[{ScalarT, ListT[SizeT, ScalarT], ImageT[]}]]]
	$$Dimensions: SizeListT[2]
	$$Channels: SizeT

Upgraders: {
	"11.2.0" -> Append["VarianceImage" -> None],
	"11.3.9" -> Append["Interleaving" -> False]
}

DecoderToEncoder: Function @ Scope[
	dims = TDimensions[#2];
	If[!VectorQ[dims, IntegerQ], Return[$Failed, Block]];
	{"Image", 
		"ImageSize" -> Reverse[Rest[dims]], "ColorChannels" -> First[dims],
		"Interleaving" -> #Interleaving,
		"ColorSpace" -> #ColorSpace, "MeanImage" -> #MeanImage,
		"VarianceImage" -> #VarianceImage
	}
]

ArrayDepth: 3

ToDecoderFunction: Function @ Scope @ With[
	{
		cspace = #ColorSpace,
		mean = #MeanImage,
		var = #VarianceImage,
		inter = #Interleaving,
		getNumChannel = If[#Interleaving, Last @* Dimensions, Length],
		expectedNumChannel = Switch[cspace,
			"Grayscale", 1|2,
			"CMYK", 4|5,
			Automatic, _,
			_, 3|4 (* RGB, ... *)
		]
	},
	meaner = Switch[mean, 
		None, Identity,
		_Image, ImageAdd[#, ImageResize[mean, ImageDimensions[#]]]&,
		_, ImageAdd[#, mean]&
	];
	func = Function[input, 
		If[!MatchQ[getNumChannel[input], expectedNumChannel],
			ColorConvert[Image[input, Interleaving -> inter], cspace],
			Image[input, ColorSpace -> cspace, Interleaving -> inter]
		]
	] /* meaner;
	Map[func]
]


