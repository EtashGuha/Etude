Begin["System`Convert`SPARQLUpdateDump`"];

$HiddenElements = {};

$DocumentedElements = {"Base", "Data", "Prefixes"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["SPARQLUpdate"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["SPARQLUpdate",
	{
		"Base" :> ImportSPARQLUpdateBase
		,
		"Data" :> ImportSPARQLUpdate
		,
		"Prefixes" :> ImportSPARQLUpdatePrefixes
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
