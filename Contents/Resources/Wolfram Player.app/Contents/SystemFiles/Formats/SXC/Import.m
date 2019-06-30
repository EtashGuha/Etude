(* ::Package:: *)

Begin["System`Convert`SXCDump`"]


ImportExport`RegisterImport[
 "SXC",
 {
	"Data" :> ImportSXC["Data"],
	"Elements" :> getElements,
	"Sheets" :> ImportSXC["SheetNames"],
	"Formulas" :> ImportSXC["Formulas"],
	{"Sheets",number_Integer} :> ImportSXC["number"],
	{"Sheets", name_String} :> ImportSXC["name"],
	{"Sheets", "Elements"} :> ImportSXC["Elements"]
 },
 "FunctionChannels" -> {"FileNames"},
 "AvailableElements" -> {"Data", "Formulas", "Sheets"},
 "DefaultElement" -> "Data",
 "BinaryFormat" -> True
]


End[]
