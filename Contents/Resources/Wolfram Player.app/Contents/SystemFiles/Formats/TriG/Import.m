Begin["System`Convert`TriGDump`"];

$HiddenElements = {};

$DocumentedElements = {"Base", "Data", "Prefixes"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["TriG"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["TriG",
	{
		"Base" :> ImportTriGBase
		,
		"Data" :> ImportTriG
		,
		"Prefixes" :> ImportTriGPrefixes
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
