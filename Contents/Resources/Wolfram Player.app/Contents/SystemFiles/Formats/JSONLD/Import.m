Begin["System`Convert`JSONLDDump`"];

$HiddenElements = {};

$DocumentedElements = {"Data", "RawData"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["JSONLD"]
				,
				$HiddenElements
			]
		];

ImportExport`RegisterImport["JSONLD",
	{
		"Data" :> ImportJSONLD
		,
		"RawData" :> ImportJSONLDRawData
		,
		"Elements" :> GetElements
	}
	,
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data"
];

End[];
