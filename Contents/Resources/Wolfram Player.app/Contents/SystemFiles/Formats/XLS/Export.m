(* ::Package:: *)

ImportExport`RegisterExport[
 "XLS",
 (System`Convert`ExcelDump`ExportExcel["XLS"][##]&),
 "Options"->{"Formulas", "Images"},
 "Sources"-> Join[{"JLink`"}, ImportExport`DefaultSources["Excel"]],
 "DefaultElement" -> "Data",
 "BinaryFormat" -> True
]
