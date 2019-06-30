(* ::Package:: *)

Begin["System`Convert`CommonGraphicsDump`"]

(*Raw hidden elements*)
$RawHiddenElements = {"Aperture", "Author", "Date", "EmbeddedThumbnailSize", "ExposureTime", "FocalLength", "ImageDescription", "ISOSpeedRatings", "Make", "Model"};

(*Documented elements, that also appear in Exif specification*)
$RawDocumentedElementsMetadataMembers = {"ColorMap", "ColorSpace", "DateTime"};

(*Raw documented elements*)
$RawDocumentedElements = {"BitDepth", "CameraTopOrientation", "Channels", "ColorProfileData", "ColorSpace", "Data",
							"DateTime", "EmbeddedThumbnail", "Exif", "FilterPattern", "FlashUsed", "Graphics", "Image", 
							"ImageSize", "RawData", "RawExif", "RawImage", "Summary", "Thumbnail"};


(*All supported metadata elements*)
$RawMetadataElements =
	Sort[
		Complement[
			Join[System`ConvertersDump`$metadataElements, (*This list is the union of Exif and IPTC namespaces, which are typically hidden*)
				System`ConvertersDump`$derivedMetadataElements (*This list contains tags, that are not in Exif specification, but are constructed from Exif tags and are documented*)
			]
			,
			$RawDocumentedElementsMetadataMembers
		]
	];

(*All supported elements*)
$RawAvailableElements =
	Sort[
		DeleteDuplicates[
			Join[
				Complement[
					Join[System`ConvertersDump`$metadataElements, $RawHiddenElements]
					,
					$RawDocumentedElementsMetadataMembers
				]
				,
				System`ConvertersDump`$derivedMetadataElements
				,
				$RawDocumentedElements
			]
		]
	];

(*Returns the list of documented elements*)
GetRawElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["Raw"]
				,
				Complement[
					Join[System`ConvertersDump`$metadataElements, $RawHiddenElements]
					,
					$RawDocumentedElementsMetadataMembers
				]
			]
		]
		

ImportExport`RegisterImport["Raw",
	{
	(*Documented elements*)
		"BitDepth"											:> GetImageMetaData["Raw", "BitDepth", All],
		"CameraTopOrientation"								:> GetImageMetaData["Raw", "CameraTopOrientation", All],
		"Channels"											:> GetChannelsElement["Raw"],
		"ColorProfileData"									:> GetImageMetaData["Raw", "ColorProfileData", All],
		"ColorSpace"										:> GetImageMetaData["Raw", "ColorSpace", All],
		"Data"												:> GetDataElement["Raw", All],
		"DateTime"											:> GetImageMetaData["Raw", "DateTime", All],
		"EmbeddedThumbnail"									:> GetEmbeddedThumbnailElement["Raw"],
		"Exif"												:> GetExifInformation["Raw"],
		{"Exif", "Elements"}								:> GetExifImportElements,
		"FilterPattern"										:> GetImageMetaData["Raw", "FilterPattern", All],
		"FlashUsed"											:> GetImageMetaData["Raw", "FlashUsed", All],
		"Graphics"											:> GetGraphicsElement["Raw"],
		"Image"												:> GetImageElement["Raw"],
		"ImageSize"											:> GetImageMetaData["Raw", "ImageSize", All],
		"RawData"											:> GetRawDataAndColorMapElements["Raw", All],
		"RawExif"											:> GetExifInformationRaw["Raw"],
		"RawImage"											:> GetRawImageElement["Raw"],
		"Summary"											:> CreateSummary["Raw"],
        "Thumbnail" | {"Thumbnail", s:(_Integer|_Symbol)}	:> GetThumbnailElement["Raw", s],
	(*Hidden elements*)
		elem : Alternatives @@ $RawHiddenElements			:> GetImageMetaData["Raw", elem, All],
		elem_String /; MemberQ[$RawMetadataElements, elem]	:> GetExifIndividualElement["Raw", elem],
	(*All elements*)
		"Elements"											:> GetRawElements
	},

	"DefaultElement"    -> "Image",
	"BinaryFormat"      -> True,
	"Options"           -> {"BitDepth", "ColorSpace", "ImageSize"},
	"Sources"           -> {"Convert`CommonGraphics`", "Convert`Exif`"},
	"AvailableElements" -> $RawAvailableElements
]

End[]