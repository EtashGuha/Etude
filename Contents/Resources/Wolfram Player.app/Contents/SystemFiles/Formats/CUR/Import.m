(* ::Package:: *)

Begin["System`Convert`ICODump`"]


ImportExport`RegisterImport["CUR",
  ImportCUR,
  {
	"GrayLevels" 	:> ICONElementsToGrayLevelsArray,
	"RGBColorArray"  :> ICONElementsToRGBColorArray,
	"GraphicsList"   :> ICONElementsToGraphics,
	"ImageList" 	 :> ICOElementsToImages,
	"Graphics" 	  :> (ICONElementsToGraphics[##][[1]]&),
	"Image" 		 :> (ICOElementsToImages[##][[1]]&),
	"HotSpot"		:> ICOElementsToHotSpot,
    "Channels"       :> GetChannels,
    "Summary"        :> CreateSummary["CUR"],
    "SummarySlideView":>CreateSummarySlideView["CUR"],
    "ImageCount"       :> GetImageFramesCount["CUR"]
  },
  "FunctionChannels"  -> {"Streams"},
  "AvailableElements" -> {"Channels", "BitDepth", "ColorSpace", "Data", "Graphics", "GraphicsList", "SummarySlideView",
						  "GrayLevels", "Image", "ImageList", "ImageSize", "RGBColorArray", "HotSpot", "Summary", "ImageCount"},
  "DefaultElement"    -> "ImageList",
  "Sources"           -> ImportExport`DefaultSources["ICO"],
  "Options"           -> {"BitDepth", "ColorSpace", "ImageSize"},
  "BinaryFormat"      -> True
]


End[]
