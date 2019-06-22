(* ::Package:: *)

Begin["System`Convert`CommonGraphicsDump`"]

(*Documented elements, that also appear in Exif specification*)
$PngDocumentedElementsMetadataMembers = {"ColorMap", "ColorSpace", "DateTime"};

(*PNG hidden elements*)
$PngHiddenElements = {"GrayLevels", "ImageOffset", "Image3D", "RGBColorArray"};

(*PNG documented elements*)
$PngDocumentedElements = {"Animation", "AnimationRepetitions", "BitDepth", "BlendOperation", "Channels", "ColorMap", "ColorProfileData",
							"ColorSpace", "Comments", "Data", "DateTime", "DisplayDurations", "DisposalOperation", "Exif", "Graphics",
							"GraphicsList", "Image", "ImageCount", "ImageList", "ImageSize", "IPTC", "RawData", "RawExif", "RawIPTC",
							"RawXMP", "Summary", "SummarySlideView", "Thumbnail", "ThumbnailList", "XMP"};

(*All supported metadata elements*)
$PngMetadataElements =
	Sort[
		Complement[
			Join[System`ConvertersDump`$metadataElements, (*This list is the union of Exif and IPTC namespaces, which are typically hidden*)
				System`ConvertersDump`$derivedMetadataElements (*This list contains tags, that are not in Exif specification, but are constructed from Exif tags and are documented*)
			]
			,
			$PngDocumentedElementsMetadataMembers
		]
	];

(*All supported elements*)
$PngAvailableElements =
	Sort[
		Join[
			Complement[
				Join[System`ConvertersDump`$metadataElements, $PngHiddenElements]
				,
				$PngDocumentedElementsMetadataMembers
			]
			,
			System`ConvertersDump`$derivedMetadataElements
			,
			$PngDocumentedElements
		]
	];

(*Returns the list of documented elements*)
GetPngElements[___] :=
	"Elements" ->
		Sort[
			DeleteDuplicates[
				Complement[
					"ImportElements" /. System`ConvertersDump`FileFormatDataFull["PNG"]
					,
					Complement[
						Join[System`ConvertersDump`$metadataElements, $PngHiddenElements]
						,
						$PngDocumentedElementsMetadataMembers
					]
				]
			]
		]

ImportExport`RegisterImport["PNG",
	{
    (*Documented elements*)
		"Animation"													       	  :> GetAnimationElement["PNG"],
		"AnimationRepetitions"	                                    	   	  :> GetImageMetaData["PNG", "AnimationRepetitions", All],
		"BitDepth" 				                                    	   	  :> GetImageMetaData["PNG", "BitDepth", All],
		"BlendOperation"|{"BlendOperation", All|"All"}						  :> GetImageMetaData["PNG","BlendOperation", All],
		{"BlendOperation", f:(_Integer|_List)}								  :> GetImageMetaData["PNG","BlendOperation", f],
		"Channels"                                                         	  :> GetChannelsElement["PNG"],
		"ColorMap"				                                     	   	  :> GetRawDataAndColorMapElements["PNG", All, "Element" -> "ColorMap"],
		"ColorProfileData" 		                                    	   	  :> GetImageMetaData["PNG", "ColorProfileData", All],
		"ColorSpace" 			  	                                       	  :> GetImageMetaData["PNG", "ColorSpace", All],
		"Comments" 				                                    	   	  :> GetImageMetaData["PNG", "Comments", All],
		"Data" 					                                    	      :> GetDataElement["PNG", All],
		"DateTime"                                                            :> GetExifIndividualElement["PNG", "DateTime"],
		"DisplayDurations"|{"DisplayDurations", All|"All"}  			      :> GetImageMetaData["PNG","DisplayDurations", All],
		"DisposalOperation"|{"DisposalOperation", All|"All"}				  :> GetImageMetaData["PNG","DisposalOperation", All],
		{"DisplayDurations", f:(_Integer|_List)}						   	  :> GetImageMetaData["PNG","DisplayDurations", f],
		{"DisposalOperation", f:(_Integer|_List)}							  :> GetImageMetaData["PNG","DisposalOperation", f],
		"Exif"                  					                          :> GetExifInformation["PNG"],
		{"Exif", "Elements"}                                                  :> GetExifImportElements,
		"Graphics" 				                                  	       	  :> GetGraphicsElement["PNG"],
		"GraphicsList"|{"GraphicsList", All|"All"}				     	   	  :> GetGraphicsListElement["PNG", All],
		{"GraphicsList", f:(_Integer|_List)}					           	  :> GetGraphicsListElement["PNG", f],
		"Image" 					                                  	   	  :> GetImageElement["PNG"],
		"ImageCount"											              :> GetImageCountElement["PNG"],
		"ImageList"|{"ImageList", All|"All"}					           	  :> GetImageListElement["PNG", All],
		"ImageSize" 				                                   	      :> GetImageMetaData["PNG", "ImageSize", All],
		{"ImageList", f:(_Integer|_List)}							          :> GetImageListElement["PNG", f],
		"IPTC"                                                                :> GetIPTCInformation["PNG"],
		{"IPTC", "Elements"}                                                  :> GetIPTCImportElements,
		"RawData"|{"RawData", All|"All"}						          	  :> GetRawDataAndColorMapElements["PNG", All, "Element" -> "RawData"],
		{"RawData", f:(_Integer|_List)}										  :> GetRawDataAndColorMapElements["PNG", f, "Element" -> "RawData"],
		"RawExif"   				                                          :> GetExifInformationRaw["PNG"],
		"RawIPTC"                                                             :> GetIPTCInformationRaw["PNG"],
		"RawXMP"				                                              :> GetXMPInformationRaw["PNG"],
		"Summary"                                                          	  :> CreateSummary["PNG"],
		"SummarySlideView"                           					   	  :> CreateSummary["PNG", True],
		"Thumbnail" | {"Thumbnail", s:(_Integer|_Symbol)}                  	  :> GetThumbnailElement["PNG", s],
		"ThumbnailList"	   													  :> GetThumbnailListElement["PNG", All],
		{"ThumbnailList", All|"All"}	   									  :> GetThumbnailListElement["PNG", All],
		{"ThumbnailList", f:(_Integer|_List)}					         	  :> GetThumbnailListElement["PNG", f],
		{"ThumbnailList", f:(_Integer|_List|All|"All"), s:(_Integer|_Symbol)} :> GetThumbnailListElement["PNG", f, s],
		"XMP"		                                                          :> GetXMPInformation["PNG"],

	(*Hidden elements*)
		elem_String /; StringMatchQ[elem, "GrayLevels"]	               	      :> GetGrayLevelsElement["PNG", All],
		{"GrayLevels", All|"All"}	      									  :> GetGrayLevelsElement["PNG", All],
		{"GrayLevels", f:(_Integer|_List)} 			         				  :> GetGrayLevelsElement["PNG", f],
		elem_String /; StringMatchQ[elem, "ImageOffset"]					  :> GetImageMetaData["PNG","ImageOffset", All],
		{"ImageOffset", All|"All"}											  :> GetImageMetaData["PNG","ImageOffset", All],
		{"ImageOffset", f:(_Integer|_List)}									  :> GetImageMetaData["PNG","ImageOffset", f],
		elem_String /; StringMatchQ[elem, "Image3D"]					   	  :> GetImage3DElement["PNG"],
		elem_String /; StringMatchQ[elem, "RGBColorArray"]                 	  :> GetRGBColorArrayElement["PNG", All],
		{"RGBColorArray", All|"All"}				         				  :> GetRGBColorArrayElement["PNG", All],
		{"RGBColorArray", f:(_Integer|_List)}							      :> GetRGBColorArrayElement["PNG", f],
		elem_String /; MemberQ[$PngMetadataElements, elem]                    :> GetExifIndividualElement["PNG", elem],

	 (*"DefaultElement" intentionally left out, converter decides whether to return Image or ImageList*)
		Automatic 													       	  :> GetDefaultImageElement["PNG"],

    (*All elements*)
		"Elements"					                                          :> GetPngElements
	},

	"AlphaChannel"      -> True,
	"BinaryFormat"      -> True,
	"Options"           -> {"BitDepth", "ColorSpace", "ImageSize"},
	"Sources"           -> {"Convert`CommonGraphics`", "Convert`Exif`", "Convert`IPTC`", "Convert`XMP`"},
	"AvailableElements" -> $PngAvailableElements
]

End[]