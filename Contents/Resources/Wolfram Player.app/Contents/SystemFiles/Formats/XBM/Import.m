(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterImport["XBM",
{
	System`Convert`BitmapDump`ImportXBitmap
},
{
	"Graphics" -> ElementsToRaster,
	"Image" -> (GraphicToImage[ElementsToRaster[##]]&),
	"Data" -> ElementsToRasterData,
	"RGBColorArray" -> ElementsToColorData[RGBColor, Heads -> True],
	"GrayLevels" :> ElementsToColorData[GrayLevel, Heads -> False],
    "Channels"  :> System`Convert`BitmapDump`GetChannels,
    "Summary" :> System`Convert`BitmapDump`CreateSummary
},
	"Sources" -> ImportExport`DefaultSources["Bitmap"],
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> {"Channels", "BitDepth", "ColorSpace", "Data", "Data", "DataType", "Graphics", "GrayLevels", "Image", "ImageSize", "RGBColorArray", "Summary"},
	"DefaultElement" -> "Image",
	"Options" -> {"DataType", "BitDepth", "ColorSpace", "ImageSize"}
]


End[]

