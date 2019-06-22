(* ::Package:: *)

Begin["System`Convert`ODSDump`"]


ImportExport`RegisterImport[
 "ODS",
 {
   "Data" :> `ImportODS["Data"],
   "Elements" :> getElements,
   "Sheets" :> ImportODS["SheetNames"],
   "Formulas" :> ImportODS["Formulas"],
   {"Sheets",number_Integer} :> ImportODS["number"],
   {"Sheets", name_String} :> ImportODS["name"],
   {"Sheets", "Elements"} :>  ImportODS["Elements"]
 },
 {},
 "FunctionChannels" -> {"FileNames"},
 "AvailableElements" -> {"Data", "Formulas", "Sheets"},
 "DefaultElement" -> "Data",
 "BinaryFormat" -> True
]


End[]
