(* ::Package:: *)

Begin["System`Convert`CommonGraphicsDump`"]

(*Documented elements, that also appear in Exif specification*)
$JpegDocumentedElementsMetadataMembers = {"ColorMap", "ColorSpace", "DateTime"};

(*JPEG hidden elements*)
$JpegHiddenElements = {(*Legacy hidden elements*) "ExifLegacy", "GrayLevels", "ImageNoExif", "ImageWithExif", "RGBColorArray"};

(*JPEG documented elements*)
$JpegDocumentedElements = {"BitDepth", "Channels", "ColorMap", "ColorProfileData", "ColorSpace", "DateTime", "Data", "Exif",
							"Graphics", "Image", "ImageSize", "IPTC", "RawData", "RawExif", "RawIPTC", "RawXMP", "Summary",
							"Thumbnail", "XMP"};

(*All supported metadata elements*)
$JpegMetadataElements =
	Sort[
		Complement[
			Join[System`ConvertersDump`$metadataElements, (*This list is the union of Exif and IPTC namespaces, which are typically hidden*)
				System`ConvertersDump`$derivedMetadataElements (*This list contains tags, that are not in Exif specification, but are constructed from Exif tags and are documented*)
			]
			,
			$JpegDocumentedElementsMetadataMembers
		]
	];

(*All supported elements*)
$JpegAvailableElements =
	Sort[
		DeleteDuplicates[
			Join[
				Complement[
					Join[System`ConvertersDump`$metadataElements, $JpegHiddenElements]
					,
					$JpegDocumentedElementsMetadataMembers
				]
				,
				System`ConvertersDump`$derivedMetadataElements
				,
				$JpegDocumentedElements
			]
		]
	];

(*Returns the list of documented elements*)
GetJpegElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["JPEG"]
				,
				Complement[
					Join[System`ConvertersDump`$metadataElements, $JpegHiddenElements]
					,
					$JpegDocumentedElementsMetadataMembers
				]
			]
		]

ImportExport`RegisterImport["JPEG",
	{
    (*Documented elements*)
        "BitDepth" 				                            :> GetImageMetaData["JPEG", "BitDepth", All],
		"Channels"                                          :> GetChannelsElement["JPEG"],
		"ColorMap"|"RawData" 		                        :> GetRawDataAndColorMapElements["JPEG", All],
		"ColorProfileData"		                            :> GetImageMetaData["JPEG", "ColorProfileData", All],
		"ColorSpace" 			                            :> GetImageMetaData["JPEG", "ColorSpace", All],
		"Data" 						                        :> GetDataElement["JPEG", All],
		"DateTime"                                          :> GetExifIndividualElement["JPEG", "DateTime"],
		"Exif"					                            :> GetExifInformation["JPEG"],
		{"Exif", "Elements"}                                :> GetExifImportElements,
		"Graphics" 					                        :> GetGraphicsElement["JPEG"],
		"Image" 					                        :> GetImageElement["JPEG"],
		"ImageSize" 					                    :> GetImageMetaData["JPEG", "ImageSize", All],
		"IPTC"                                              :> GetIPTCInformation["JPEG"],
		{"IPTC", "Elements"}                                :> GetIPTCImportElements,
		"RawExif"   				                        :> GetExifInformationRaw["JPEG"],
		"RawIPTC"                                           :> GetIPTCInformationRaw["JPEG"],
		"RawXMP"				                            :> GetXMPInformationRaw["JPEG"],
		"Summary"                                           :> CreateSummary["JPEG"],
		"Thumbnail" | {"Thumbnail", s:(_Integer|_Symbol)}   :> GetThumbnailElement["JPEG", s],
		"XMP"		                                        :> GetXMPInformation["JPEG"],

	(*Hidden elements*)
		elem_String /; StringMatchQ[elem, "GrayLevels"]	    :> GetGrayLevelsElement["JPEG", All],
		elem_String /; StringMatchQ[elem, "ImageNoExif"]    :> GetImageElementNoExif["JPEG"],
		elem_String /; StringMatchQ[elem, "ImageWithExif"]  :> GetImageElementWithExif["JPEG"],
		elem_String /; StringMatchQ[elem, "RGBColorArray"]  :> GetRGBColorArrayElement["JPEG", All],
		elem_String /; MemberQ[$JpegMetadataElements, elem] :> GetExifIndividualElement["JPEG", elem],

    (*All elements*)
		"Elements"				                            :> GetJpegElements
	},

	"DefaultElement"    -> "Image",
    "BinaryFormat"      -> True,
    "Options"           -> {"BitDepth", "ColorSpace", "ImageSize"},
    "Sources"           -> {"Convert`CommonGraphics`", "Convert`Exif`", "Convert`IPTC`", "Convert`XMP`"},
	"AvailableElements" -> $JpegAvailableElements
]

End[]