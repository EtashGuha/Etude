Begin["System`Convert`NQuadsDump`"];

$HiddenElements = {};

$DocumentedElements = {"Data"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["NQuads"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["NQuads",
	{
		"Data" :> ImportNQuads
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
