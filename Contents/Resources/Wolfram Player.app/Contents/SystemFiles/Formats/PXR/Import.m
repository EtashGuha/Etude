(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterImport[
  "PXR",
  System`Convert`PXRDump`ImportPXR,
  {
	"Data" -> ElementsToRasterData,
	"Graphics" -> ElementsToRaster,
	"Image" -> (GraphicToImage[System`ConvertersDump`ElementsToRaster[##]]&),
	"RGBColorArray" -> ElementsToColorData[RGBColor, Heads -> True],
	"GrayLevels" :> ElementsToColorData[GrayLevel, Heads -> False],
    "Channels" :> System`Convert`PXRDump`GetChannels,
    "Summary" :>  System`Convert`PXRDump`CreateSummary
  },
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"Channels", "BitDepth", "ColorSpace", "Data", "DataType", "Graphics", "GrayLevels", "Image", "ImageSize", "RGBColorArray", "Summary"},
  "DefaultElement" -> "Image",
  "Options" -> {"ColorSpace", "ImageSize"},
  "BinaryFormat" -> True
]


End[]
