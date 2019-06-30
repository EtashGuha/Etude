(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterImport[
  "TGA",
  System`Convert`TGADump`ImportTGA,
  {
	"Data" -> ElementsToRasterData,
	"Graphics" -> ElementsToRaster,
	"Image" -> (GraphicToImage[ElementsToRaster[##]]&),
	"RGBColorArray" -> ElementsToColorData[RGBColor, Heads -> True],
	"GrayLevels" :> ElementsToColorData[GrayLevel, Heads -> False],
    "Channels" :> System`Convert`TGADump`GetChannels,
    "Summary" :> System`Convert`TGADump`CreateSummary
  },
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"BitDepth", "ColorMap", "ColorSpace", "Data", "DataType",
			"Graphics", "GrayLevels", "Image", "ImageSize", "RawData", "Channels",
			"RGBColorArray", "Summary"},
  "DefaultElement" -> "Image",
  "Options" -> {"DataType", "BitDepth", "ColorSpace", "ImageSize"},
  "BinaryFormat" -> True,
  "AlphaChannel"->True
]


End[]
