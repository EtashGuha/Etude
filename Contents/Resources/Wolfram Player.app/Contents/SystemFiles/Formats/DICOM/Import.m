(* ::Package:: *)

Begin["System`Convert`DicomDump`"]


ImportExport`RegisterImport[
  "DICOM",
  {
	"MetaInformation" -> ImportMetaInformation,
	ImportDICOM
  },
  {
	"GraphicsList" -> GetGraphics,
	"ImageList"    -> (GraphicToImage[GetGraphics[##],##]&),
	"Graphics"     -> GetGraphic[1],
	"Image"        -> (GraphicToImage[GetGraphic[1][##],##]&),
	"Data"         -> GetData,
	Automatic      -> GetDefault,
	"Overlays"     :> GetOverlays,
	"ColorMap"     :> GetColormap,
	"ColorSpace"   :> GetColorspace,
	"ImageSize"    :> GetImageSize,
	"BitDepth"     :> GetStoredBits,
    "Summary"      :> CreateSummary,
    "Channels"     :> GetChannels,
	{"MetaInformation", e_String /; StringMatchQ[e, "(*,*)"]} :> GetNumber[e]
  },
  "AvailableElements" ->  {"Channels", "BitDepth", "ColorMap", "ColorSpace", "Data", "Graphics", "GraphicsList", "Image", "ImageList", "ImageSize", "MetaInformation", "Overlays", "Summary"},
  "Sources"           -> {"Convert`DicomDataDictionary`", "Convert`Dicom`"},
  "BinaryFormat"      -> True,
  "AlphaChannel"      -> True
]


End[]
