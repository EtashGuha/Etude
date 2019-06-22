(* ::Package:: *)

Begin["System`Convert`CommonGraphicsDump`"]


ImportExport`RegisterImport["WebP",
	{
		"Graphics" 			 		:> GetGraphicsElement["WebP"],
		"Image" 						:> GetImageElement["WebP"],
        "Thumbnail" | {"Thumbnail", s:(_Integer|_Symbol)} :> GetThumbnailElement["WebP", s],
		"Data" 				 		:> GetDataElement["WebP", All],
		"BitDepth" 			 		:> GetImageMetaData["WebP", "BitDepth", All],
		"ColorProfileData" 	 		:> GetImageMetaData["WebP", "ColorProfileData", All],
		"ColorSpace" 		   		:> GetImageMetaData["WebP", "ColorSpace", All],
		"ImageSize" 					:> GetImageMetaData["WebP", "ImageSize", All],
		"RGBColorArray" 				:> GetRGBColorArrayElement["WebP", All],
		"GrayLevels" 	   			:> GetGrayLevelsElement["WebP", All],
		"Elements"					  :> GetListOfElements["WebP"],
        "Channels"                       :> GetChannelsElement["WebP"],
        "Summary"                        :> CreateSummary["WebP"],
		GetListOfElements["WebP"]
	},
	"Sources" -> {"JLink`", "Convert`Exif`", "Convert`CommonGraphics`"},
	"AvailableElements" ->
		{
			"BitDepth", "ColorProfileData", "ColorSpace", "Data", "Summary", "Channels",
			"Graphics", "GrayLevels", "Image", "ImageSize", "RGBColorArray", "Thumbnail"
		},
	"DefaultElement" -> "Image",
	"Options" -> {"BitDepth", "ColorSpace", "ImageSize"},
	"BinaryFormat" -> True
]

End[]
