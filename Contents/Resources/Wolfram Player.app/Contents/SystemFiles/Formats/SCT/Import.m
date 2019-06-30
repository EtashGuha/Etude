(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterImport[
	"SCT",
	System`Convert`SCTDump`ImportSCT,
	{
		"Data" -> ElementsToRasterData,
		"Graphics" -> ElementsToRaster,
		"Image" -> (GraphicToImage[ElementsToRaster[##]]&),
		"RGBColorArray" -> ElementsToColorData[RGBColor, Heads -> True],
		"GrayLevels" :> ElementsToColorData[GrayLevel, Heads -> False],
        "Channels" :> System`Convert`SCTDump`GetChannels,
        "Summary" :> System`Convert`SCTDump`CreateSummary
	},
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> {"BitDepth", "ColorSpace", "Data", "DataType", "Graphics", "Channels", 
			"GrayLevels", "Image", "ImageResolution", "ImageSize", "RGBColorArray", "Summary"},
	"DefaultElement" -> "Image",
	"Options" -> {"ColorSpace", "ImageSize"},
	"BinaryFormat" -> True
]


End[]
