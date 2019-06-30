Begin["System`Convert`NTriplesDump`"];

$HiddenElements = {};

$DocumentedElements = {"Data"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["NTriples"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["NTriples",
	{
		"Data" :> ImportNTriples
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
