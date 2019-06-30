(* ::Package:: *)

Begin["System`Convert`ICODump`"]


ImportExport`RegisterImport["ICO",
  ImportICO,
  {
    "GrayLevels" 	  :> ICONElementsToGrayLevelsArray,
	"RGBColorArray"    :> ICONElementsToRGBColorArray,
	"GraphicsList"     :> ICONElementsToGraphics,
	"ImageList" 	   :> ICOElementsToImages,
	"Graphics" 	    :> (ICONElementsToGraphics[##][[1]]&),
	"Image" 		   :> (ICOElementsToImages[##][[1]]&),
    "Channels"         :> GetChannels,
    "ImageCount"       :> GetImageFramesCount["ICO"],
    "SummarySlideView" :> CreateSummarySlideView["ICO"],
    "Summary"          :> CreateSummary["ICO"]
  },
  "FunctionChannels"  -> {"Streams"},
  "AvailableElements" -> {"ImageCount", "Channels", "BitDepth", "ColorSpace", "Data", "Graphics", "GraphicsList",
						  "GrayLevels", "Image", "ImageList", "ImageSize", "RGBColorArray", "Summary", "SummarySlideView"},
  "DefaultElement"    -> "ImageList",
  "Sources"           -> ImportExport`DefaultSources["ICO"],  
  "Options"           -> {"BitDepth", "ColorSpace", "ImageSize"},
  "BinaryFormat"      -> True
]


End[]
