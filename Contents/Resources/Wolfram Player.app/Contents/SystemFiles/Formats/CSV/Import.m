(* ::Package:: *)

Begin["System`Convert`TableDump`"];

$HiddenElements = {};

$DocumentedElements = {"Dimensions", "Data", "Dataset", "Grid", "MaxColumnCount", "RawData", "RowCount", "Summary"};

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]];

GetElements[___] :=
	"Elements" ->
		Sort[
			Complement[
				"ImportElements" /. System`ConvertersDump`FileFormatDataFull["CSV"]
				,
				$HiddenElements
			]
		];

$PartialAccessForms = System`ConvertersDump`Utilities`$PartialAccessForms;

ImportExport`RegisterImport["CSV",
	{
		{elem : ("RawData" | "Data" | "Grid" | "Dataset"), args___}		:> (ImportSV["CSV", elem, args])
		,
		elem : ("ColumnCounts" | "Dimensions" | "RowCount" 
					| "MaxColumnCount" | "Summary")						:> (ImportSVMetadata["CSV", elem])
		,
		"Elements"														:> GetElements
		,
		(ImportSV["CSV", elem, All, All, ##] &)
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
