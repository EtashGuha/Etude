Begin["System`Convert`SPARQLResultsXMLDump`"];

$HiddenElements = {};

$DocumentedElements = {"Data"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["SPARQLResultsXML"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["SPARQLResultsXML",
	{
		"Data" :> ImportSPARQLResultsXML
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
