(* ::Package:: *)

Begin["System`Convert`TableDump`"];

$HiddenElements = {};

$DocumentedElements = {"Dimensions", "Data", "Dataset", "Grid", "MaxColumnCount", "RawData", "RowCount", "Summary"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["TSV"]
				,
				$HiddenElements
			]
		];

$PartialAccessForms = System`ConvertersDump`Utilities`$PartialAccessForms;

ImportExport`RegisterImport["TSV",
	{
		{elem : ("RawData" | "Data" | "Grid" | "Dataset"), args___}		:> (ImportSV["TSV", elem, args])
		,
		elem : ("ColumnCounts" | "Dimensions" | "RowCount" 
					| "MaxColumnCount" | "Summary")						:> (ImportSVMetadata["TSV", elem])
		,
		"Elements"														:> GetElements
		,
		(ImportSV["TSV", elem, All, All, ##] &)
	}
	,
	"Sources" -> ImportExport`DefaultSources["Table"],
	"FunctionChannels" -> {"FileNames", "Streams"},
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data",
	"BinaryFormat" -> True,
	"SkipPostImport" -> <|
		"RawData" -> {$PartialAccessForms, $PartialAccessForms},
		"Data" -> {$PartialAccessForms, $PartialAccessForms},
		"Grid" -> {$PartialAccessForms, $PartialAccessForms},
		"Dataset" -> {$PartialAccessForms, $PartialAccessForms}
	|>
];

End[];
