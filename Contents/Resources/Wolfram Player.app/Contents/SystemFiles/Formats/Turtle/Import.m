Begin["System`Convert`TurtleDump`"];

$HiddenElements = {};

$DocumentedElements = {"Base", "Data", "Prefixes"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["Turtle"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["Turtle",
	{
		"Base" :> ImportTurtleBase
		,
		"Data" :> ImportTurtle
		,
		"Prefixes" :> ImportTurtlePrefixes
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
