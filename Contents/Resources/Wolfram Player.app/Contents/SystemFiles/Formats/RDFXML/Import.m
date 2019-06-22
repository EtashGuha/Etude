Begin["System`Convert`RDFXMLDump`"];

$HiddenElements = {};

$DocumentedElements = {"Base", "Data", "Prefixes"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["RDFXML"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["RDFXML",
	{
		"Base" :> ImportRDFXMLBase
		,
		"Data" :> ImportRDFXML
		,
		"Prefixes" :> ImportRDFXMLPrefixes
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
