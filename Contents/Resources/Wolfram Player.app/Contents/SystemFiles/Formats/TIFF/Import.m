(* ::Package:: *)

Begin["System`Convert`CommonGraphicsDump`"]

(*Documented elements, that also appear in Exif specification*)
$TiffDocumentedElementsMetadataMembers = {"Author", "ColorMap", "ColorSpace", "CopyrightNotice", "DateTime", "Device", "DeviceManufacturer",
											"ImageCreationDate", "RowsPerStrip", "TileSize"};

(*TIFF hidden elements*)
$TiffHiddenElements = {"GrayLevels", "RGBColorArray"};

(*TIFF documented elements*)
$TiffDocumentedElements = {"Animation", "Author", "BitDepth", "Channels", "ColorMap", "ColorProfileData", "ColorSpace", "Comments",
							"CopyrightNotice", "Data", "DateTime", "Device", "DeviceManufacturer", "Exif", "Graphics", "GraphicsList",
							"Image", "Image3D", "ImageCount", "ImageCreationDate", "ImageEncoding", "ImageList", "ImageResolution",
							"ImageSize", "IPTC", "RawData", "RawExif", "RawIPTC", "RawXMP", "RowsPerStrip", "Summary", "SummarySlideView",
							"Thumbnail", "ThumbnailList", "TileSize", "XMP"};

(*All supported metadata elements*)
$TiffMetadataElements =
	Sort[
		Complement[
			Join[System`ConvertersDump`$metadataElements, (*This list is the union of Exif and IPTC namespaces, which are typically hidden*)
				System`ConvertersDump`$derivedMetadataElements (*This list contains tags, that are not in Exif specification, but are constructed from Exif tags and are documented*)
			]
			,
			$TiffDocumentedElementsMetadataMembers
		]
	];

(*All supported elements*)
$TiffAvailableElements =
	Sort[
		Join[
			Complement[
				Join[System`ConvertersDump`$metadataElements, $TiffHiddenElements]
				,
				$TiffDocumentedElementsMetadataMembers
			]
			,
			System`ConvertersDump`$derivedMetadataElements
			,
			$TiffDocumentedElements
		]
	];

(*Returns the list of documented elements*)
GetTiffElements[___] :=
	"Elements" ->
		Sort[
			DeleteDuplicates[
				Complement[
					"ImportElements" /. System`ConvertersDump`FileFormatDataFull["TIFF"]
					,
					Complement[
						Join[System`ConvertersDump`$metadataElements, $TiffHiddenElements]
						,
						$TiffDocumentedElementsMetadataMembers
					]
				]
			]
		]

ImportExport`RegisterImport["TIFF",
	{	
    (*Documented elements*)
		"Animation"							   						          :> GetAnimationElement["TIFF"],
		"Author"|{"Author", All|"All"}		   						          :> GetImageMetaData["TIFF","Author", All],
		{"Author", f:(_Integer|_List)}   								      :> GetImageMetaData["TIFF","Author", f],
		"BitDepth"|{"BitDepth", All|"All"}   							      :> GetImageMetaData["TIFF","BitDepth", All],
		{"BitDepth", f:(_Integer|_List)}		      					      :> GetImageMetaData["TIFF","BitDepth", f],
		{"CameraTopOrientation", f:(_Integer|_List)}      				      :> GetImageMetaData["TIFF","CameraTopOrientation", f],
		"Channels"                                                            :> GetChannelsElement["TIFF"],
		"ColorMap"|{"ColorMap", All|"All"}					      		      :> GetRawDataAndColorMapElements["TIFF", All],
		{"ColorMap", f:(_Integer|_List)}						      	      :> GetRawDataAndColorMapElements["TIFF", f],
		"ColorProfileData"|{"ColorProfile", All|"All"}				          :> GetImageMetaData["TIFF","ColorProfileData", All],
		{"ColorProfile", f:(_Integer|_List)}						          :> GetImageMetaData["TIFF","ColorProfile", f],
		"ColorSpace"|{"ColorSpace", All|"All"}						          :> GetImageMetaData["TIFF","ColorSpace", All],
		{"ColorSpace", f:(_Integer|_List)}							          :> GetImageMetaData["TIFF","ColorSpace", f],
		"Comments"|{"Comments", All|"All"}							          :> GetImageMetaData["TIFF","Comments", All],
		"Comments"|{"Comments", All|"All"}							          :> GetImageMetaData["TIFF","Comments", All],
		{"Comments", f:(_Integer|_List)}							          :> GetImageMetaData["TIFF","Comments", f],
		"CopyrightNotice"|{"CopyrightNotice", All|"All"}			          :> GetImageMetaData["TIFF","CopyrightNotice", All],
		{"CopyrightNotice", f:(_Integer|_List)}						          :> GetImageMetaData["TIFF","CopyrightNotice", f],
		"Data"|{"Data", All|"All"}								         	  :> GetDataElement["TIFF", All],
		{"Data", f:(_Integer|_List)}								          :> GetDataElement["TIFF", f],
		"DateTime"                                                            :> GetExifIndividualElement["TIFF", "DateTime"],
		"Device"|{"Device", All|"All"}							         	  :> GetImageMetaData["TIFF","Device", All],
		{"Device", f:(_Integer|_List)}								          :> GetImageMetaData["TIFF","Device", f],
		"DeviceManufacturer"|{"DeviceManufacturer", All|"All"}	              :> GetImageMetaData["TIFF","DeviceManufacturer", All],
		{"DeviceManufacturer", f:(_Integer|_List)}					          :> GetImageMetaData["TIFF","DeviceManufacturer", f],
		"Exif"                  					                          :> GetExifInformation["TIFF"],
		{"Exif", "Elements"}                                                  :> GetExifImportElements,
		"Graphics"													          :> GetGraphicsElement["TIFF"],
		"GraphicsList"|{"GraphicsList", All|"All"}				         	  :> GetGraphicsListElement["TIFF", All],
		{"GraphicsList", f:(_Integer|_List)}						          :> GetGraphicsListElement["TIFF", f],
		"Image"													              :> GetImageElement["TIFF"],
		"ImageCount"												          :> GetImageCountElement["TIFF"],
		"ImageCreationDate"|{"ImageCreationDate", All|"All"}	              :> GetImageMetaData["TIFF","ImageCreationDate", All],
		{"ImageCreationDate", f:(_Integer|_List)}					          :> GetImageMetaData["TIFF","ImageCreationDate", f],
		"ImageEncoding"|{"ImageEncoding", All|"All"}			           	  :> GetImageMetaData["TIFF","ImageEncoding", All],
		{"ImageEncoding", f:(_Integer|_List)}						          :> GetImageMetaData["TIFF","ImageEncoding", f],
		"ImageList"|{"ImageList", All|"All"}					           	  :> GetImageListElement["TIFF", All],
		{"ImageList", f:(_Integer|_List)}							          :> GetImageListElement["TIFF", f],
		"ImageResolution"|{"ImageResolution", All|"All"}		              :> GetImageMetaData["TIFF","ImageResolution", All],
		{"ImageResolution", f:(_Integer|_List)}						          :> GetImageMetaData["TIFF","ImageResolution", f],
		"ImageSize"|{"ImageSize", All|"All"}						          :> GetImageMetaData["TIFF","ImageSize", All],
		{"ImageSize", f:(_Integer|_List)}							          :> GetImageMetaData["TIFF","ImageSize", f],
		"Image3D"												              :> GetImage3DElement["TIFF"],
		"IPTC"                                                                :> GetIPTCInformation["TIFF"],
		{"IPTC", "Elements"}                                                  :> GetIPTCImportElements,
		"RawData"|{"RawData", All|"All"}						              :> GetRawDataAndColorMapElements["TIFF", All],
		{"RawData", f:(_Integer|_List)}								          :> GetRawDataAndColorMapElements["TIFF", f],
		"RawExif"   				                                          :> GetExifInformationRaw["TIFF"],
		"RawIPTC"                                                             :> GetIPTCInformationRaw["TIFF"],
		"RawXMP"				                                              :> GetXMPInformationRaw["TIFF"],
		"RowsPerStrip"|{"RowsPerStrip", All|"All"}				         	  :> GetImageMetaData["TIFF", "RowsPerStrip", All],
		{"RowsPerStrip", f:(_Integer|_List)}						          :> GetImageMetaData["TIFF", "RowsPerStrip", f],
		"Summary"                                                             :> CreateSummary["TIFF", False],
		"SummarySlideView"                                                    :> CreateSummary["TIFF", True],
		"TileSize"|{"TileSize", All|"All"}							          :> GetImageMetaData["TIFF", "TileSize", All],
		{"TileSize", f:(_Integer|_List)}							          :> GetImageMetaData["TIFF", "TileSize", f],
        "Thumbnail" 				       							    	  :> GetThumbnailElement["TIFF"],
		{"Thumbnail", f:(_Integer|_Symbol)}           				          :> GetThumbnailElement["TIFF", f],
		"ThumbnailList"												          :> GetThumbnailListElement["TIFF"],
		{"ThumbnailList", f:(_Integer|_List|All|"All")} 			          :> GetThumbnailListElement["TIFF", f],
		{"ThumbnailList", f:(_Integer|_List|All|"All"), s:(_Integer|_Symbol)} :> GetThumbnailListElement["TIFF", f, s],
		"XMP"		                                                          :> GetXMPInformation["TIFF"],

	(*Hidden elements*)
		elem_String /; StringMatchQ[elem, "GrayLevels"]                       :> GetGrayLevelsElement["TIFF", All],
		{"GrayLevels", All|"All"}			            		 	          :> GetGrayLevelsElement["TIFF", All],
		{"GrayLevels", f:(_Integer|_List)}							          :> GetGrayLevelsElement["TIFF", f],
		elem_String /; StringMatchQ[elem, "RGBColorArray"]                    :> GetRGBColorArrayElement["TIFF", All],
		{"RGBColorArray", All|"All"}			                         	  :> GetRGBColorArrayElement["TIFF", All],
		{"RGBColorArray", f:(_Integer|_List)}						          :> GetRGBColorArrayElement["TIFF", f],
		elem_String /; MemberQ[$TiffMetadataElements, elem]                   :> GetExifIndividualElement["TIFF", elem],

    (*"DefaultElement" intentionally left out, converter decides whether to return Image or ImageList*)
		Automatic 													          :> GetDefaultImageElement["TIFF"],

    (*All elements*)
		"Elements"					                                          :> GetTiffElements,
		getTIFFElements
	},

    "AlphaChannel"      -> True,
    "BinaryFormat"      -> True,
    "Options"           -> {"BitDepth", "ColorSpace", "Comments", "ImageSize"},
    "Sources"           -> {"Convert`CommonGraphics`", "Convert`Exif`", "Convert`IPTC`", "Convert`XMP`"},
    "AvailableElements" -> $TiffAvailableElements
]

End[]
