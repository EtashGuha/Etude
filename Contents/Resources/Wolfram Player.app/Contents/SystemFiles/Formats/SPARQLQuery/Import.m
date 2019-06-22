Begin["System`Convert`SPARQLQueryDump`"];

$HiddenElements = {};

$DocumentedElements = {"Base", "Data", "Prefixes"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["SPARQLQuery"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["SPARQLQuery",
	{
		"Base" :> ImportSPARQLQueryBase
		,
		"Data" :> ImportSPARQLQuery
		,
		"Prefixes" :> ImportSPARQLQueryPrefixes
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
