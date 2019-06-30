Begin["System`Convert`ExcelDump`"]

$DocumentedElements = Sort[{"Data", "Dataset", "Dimensions", "Formulas", "FormattedData", "Images", "SheetCount", "Sheets"}]

$HiddenElements = # <> "Legacy" & /@ $DocumentedElements

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]]

GetElements[___] := "Elements" -> $DocumentedElements

$PartialAccessForms = (All | "All" | _Integer | _Span | {_Integer.. })
$SheetForms = ($PartialAccessForms | _String | {(_Integer | _String) ..})
$SubElems = {$SheetForms, $PartialAccessForms, $PartialAccessForms}

ImportExport`RegisterImport["XLS",
	{
		"Elements" 											:> GetElements,

		elem : ("Images" | "Sheets" | "SheetCount" | "Dimensions")							:> GetBookData["XLS"][elem],
		elem : ("Data" | "Formulas" | "FormattedData" | "Dataset")							:> ImportExcel["XLS"][elem],
		{elem : ("Data" | "Formulas" | "FormattedData" | "Dataset" | "Sheets"), args__}		:> ImportExcel["XLS"][elem, args],

		(* Legacy *)

		elem : "DataLegacy"									:> ImportExcelLegacy["XLS"][False],
		elem : {"DataLegacy", "Elements"}					:> getDataElementsLegacy["XLS"],
		elem : {"DataLegacy", name_String|name_Integer}		:> GetDataSheetLegacy["XLS"][name],
		elem : "FormulasLegacy"								:> ImportExcelLegacy["XLS"][True],
		elem : "FormattedDataLegacy"						:> ImportFormattedDataLegacy["XLS"],
		elem : "ImagesLegacy"								:> GetImagesLegacy["XLS"],
		elem : "SheetsLegacy"								:> JustSheetNamesLegacy["XLS"],
		elem : {"SheetsLegacy", "Elements"}					:> SheetNamesLegacy["XLS"]["SheetsLegacy"],
		elem : {"SheetsLegacy", name_String|name_Integer}	:> GetSheetLegacy["XLS"][name]
	}
	,
	"Sources" -> Join[{"JLink`"}, ImportExport`DefaultSources["Excel"]],
	"FunctionChannels" -> {"FileNames"},
	"AvailableElements" -> $AvailableElements,
	"DefaultElement" -> "Data",
	"BinaryFormat" -> True,
	"SkipPostImport" -> <|
		"Data" -> $SubElems,
		"Formulas" -> $SubElems,
		"FormattedData" -> $SubElems,
		"Sheets" -> $SubElems,
		"Dataset" -> $SubElems
	|>
]

End[]