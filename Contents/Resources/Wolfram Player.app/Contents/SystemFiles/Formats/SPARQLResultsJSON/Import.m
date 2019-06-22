Begin["System`Convert`SPARQLResultsJSONDump`"];

$HiddenElements = {};

$DocumentedElements = {"Data"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["SPARQLResultsJSON"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["SPARQLResultsJSON",
	{
		"Data" :> ImportSPARQLResultsJSON
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
