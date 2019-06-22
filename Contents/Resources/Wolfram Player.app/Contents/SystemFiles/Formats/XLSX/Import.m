Begin["System`Convert`ExcelDump`"]

$DocumentedElements = Sort[{"Data", "Dataset", "Dimensions", "Formulas", "FormattedData", "Images", "SheetCount", "Sheets"}]

$HiddenElements = # <> "Legacy" & /@ $DocumentedElements

$AvailableElements = Sort[Join[$HiddenElements, $DocumentedElements]]

GetElements[___] := "Elements" -> $DocumentedElements

$PartialAccessForms = (All | "All" | _Integer | _Span | {_Integer.. })
$SheetForms = ($PartialAccessForms | _String | {(_Integer | _String) ..})
$SubElems = {$SheetForms, $PartialAccessForms, $PartialAccessForms}

ImportExport`RegisterImport["XLSX",
	{
		"Elements" 											:> GetElements,

		elem : ("Images" | "Sheets" | "SheetCount" | "Dimensions")							:> GetBookData["XLSX"][elem],
		elem : ("Data" | "Formulas" | "FormattedData" | "Dataset")							:> ImportExcel["XLSX"][elem],
		{elem : ("Data" | "Formulas" | "FormattedData" | "Dataset" | "Sheets"), args__}		:> ImportExcel["XLSX"][elem, args],

		(* Legacy *)

		elem : "DataLegacy"									:> ImportExcelLegacy["XLSX"][False],
		elem : {"DataLegacy", "Elements"}					:> getDataElementsLegacy["XLSX"],
		elem : {"DataLegacy", name_String|name_Integer}		:> GetDataSheetLegacy["XLSX"][name],
		elem : "FormulasLegacy"								:> ImportExcelLegacy["XLSX"][True],
		elem : "FormattedDataLegacy"						:> ImportFormattedDataLegacy["XLSX"],
		elem : "ImagesLegacy"								:> GetImagesLegacy["XLSX"],
		elem : "SheetsLegacy"								:> JustSheetNamesLegacy["XLSX"],
		elem : {"SheetsLegacy", "Elements"}					:> SheetNamesLegacy["XLSX"]["SheetsLegacy"],
		elem : {"SheetsLegacy", name_String|name_Integer}	:> GetSheetLegacy["XLSX"][name]
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