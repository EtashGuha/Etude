(* ::Package:: *)

Begin["System`Convert`PNMDump`"]


ImportExport`RegisterImport[
	"PNM",
	ImportPNM["PNM"],
	{
		"Graphics" -> ImportPNMGraphics,
		"Image" -> ImportPNMImage,
		"BitDepth" -> ImportPNMBitDepth,
		"GrayLevels" -> ImportPNMGrayLevels,
		"RGBColorArray" -> ImportPNMRGBColorArray,
        "Channels" :>  GetChannels,
        "Summary" :> CreateSummary
	},
	"AvailableElements" -> {"BitDepth", "ColorSpace", "Data",
			"DataType", "Graphics", "GrayLevels", "Image", "ImageSize",
			"RGBColorArray", "Summary", "Channels"},
	"DefaultElement" -> "Image",
	"FunctionChannels"  -> {"Streams"},
	"Sources" -> ImportExport`DefaultSources["PNM"],
	"Options" -> {"DataType", "BitDepth", "ColorSpace", "ImageSize"},
	"BinaryFormat" -> True
]


End[]
