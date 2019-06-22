(* ::Package:: *)

Begin["System`Convert`FITSDump`"]

$FitsElements = {"Data", "Graphics", "Image", "MetaInformation", "Plaintext", "Range", "RawData", "RawMetaInformation", "Summary", "SummarySlideView", "TableData"};

$FitsHiddenElements = {"ExtendedData", "IndeterminateMask", "Extensions", "Metadata"};

$FitsMetadataElements = {"Airmass", "Author", "BitDepth", "Channels", "ColorSpace", "Comments", "DataType", "Declination", "Device", "Dimensions", "Equinox", "ExposureTime", "Extension", "HDUCount",
	"History", "HourAngle", "ImageSize", "Object", "ObservationDate", "Observer", "Organization", "Reference", "RightAscension", "SiderealTime", "TableHeaders", "TableUnits", "Telescope"
};

$FitsDataPartialAccess = {"Data", "Graphics", "IndeterminateMask", "Image", "Plaintext", "RawData", "ExtendedData", "TableData"};
$FitsHeaderPartialAccess = Join[DeleteCases[$FitsMetadataElements, "HDUCount"], {"DataType", "Metadata", "MetaInformation", "RawMetaInformation", "Range", "SummarySlideView", "TableHeaders", "TableUnits"}];

$FitsDocumentedElements =
$FitsDocumentedElements =
	SortBy[
		DeleteDuplicates[
			Join[$FitsElements,
				$FitsMetadataElements
			]
		],
		ToString
	]

$FitsAvailableElements =
	SortBy[
		DeleteDuplicates[
			Join[$FitsDocumentedElements,
				$FitsHiddenElements
			]
		],
		ToString
	]

$PartialAccessForms = (All | "All" | _Integer | {_Integer.. } | _Span)

(*Returns the list of documented elements*)
GetFitsElements[___] := "Elements" -> $FitsDocumentedElements

ImportExport`RegisterImport[
	"FITS",
	{
	(*Data representation elements*)
		{"Data", args___}               :> getFitsData[args],
		{"DataType", args___}           :> getFitsDataType[args],
		{"Graphics", args___}           :> getFitsGraphics[args],
		{"Image", args___}              :> getFitsImage[args],
		{"MetaInformation", args___}    :> getFitsMeta[args],
		{"RawMetaInformation", args___} :> getFitsMetaRaw[args],
		{"Plaintext", args___}          :> getFitsTextualData[args],
		{"Range", args___}              :> getFitsRange[args],
		{"RawData", args___}            :> getFitsRawData[args],
		"Summary"                       :> getFitsSummary,
		{"SummarySlideView", args___}   :> getFitsSummarySlideView[args],
		{"TableData", args___}          :> getFitsTable[args],
		{"TableHeaders", args___}       :> getFitsTableHeaders[args],
		{"TableUnits", args___}         :> getFitsTableUnits[args],

	(*Metadata elements*)
		{elem_String, args___} /;
			MemberQ[$FitsMetadataElements, elem] :> getFitsMetaElement[elem, args],

	(*Hidden elements*)
		{elem_String, args___} /;
			StringMatchQ[elem, "ExtendedData"]      :> getFitsExtendedData[args],
		{elem_String, args___} /;
			StringMatchQ[elem, "Metadata"]      	:> getFitsMetaLegacy[args],
		{elem_String, args___} /;
			StringMatchQ[elem, "IndeterminateMask"] :> getFitsMask[args],
		{elem_String, args___} /;
			StringMatchQ[elem, "Extensions"]        :> getFitsMetaElement["Extensions", args],

	(*"DefaultElement" intentionally left out.*)
		Automatic :> getFitsDefaultElement,

	(*All elements*)
		"Elements" :> GetFitsElements
	},

(*Converter settings*)
	"BinaryFormat"      -> True,
	"Sources"           -> {"Convert`FITS`"},
	"AvailableElements" -> $FitsAvailableElements,
	"SkipPostImport" -> Join[
		AssociationMap[
			ConstantArray[$PartialAccessForms, 1000] &,
			$FitsDataPartialAccess
		]
		,
		AssociationMap[
			$PartialAccessForms &,
			$FitsHeaderPartialAccess
		]
	]
]

End[]