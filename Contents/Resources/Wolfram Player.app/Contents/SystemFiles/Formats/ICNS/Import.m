(* ::Package:: *)

Begin["System`Convert`ICNSDump`"]


ImportExport`RegisterImport[
  "ICNS",
  {
   "Elements" :> ImportICNSElements,
   ImportICNS
  },
  {
    "BitDepth"         :> ImportICNSBitDepth,
    "ColorSpace"       :> ImportICNSColorSpace,
    "Graphics"         :> ImportICNSGraphics,
    "GraphicsList"     :> ImportICNSGraphicsList,
    "Data"             :> ImportICNSData,
    "Image"            :> ImportICNSImage,
    "ImageSize"        :> ImportICNSImageSize,
    "RGBColorArray"    :> ImportICNSRGBColorArray,
    "GrayLevels"       :> ImportICNSGrayLevels,
    "Summary"          :> CreateSummary,
    "SummarySlideView" :> CreateSummarySlideView,
    "ImageCount"       :> GetImageFramesCount,
    "Channels"         :> GetChannels
  },
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"ImageCount", "Channels", "Summary", "SummarySlideView", "BitDepth","ColorSpace","Data","Graphics","GraphicsList","GrayLevels","Image","ImageList","ImageSize","RGBColorArray"},
  "DefaultElement"-> "ImageList",
  "Options" -> {"BitDepth", "ColorSpace", "ImageSize"},
  "BinaryFormat"->True
]


End[]
