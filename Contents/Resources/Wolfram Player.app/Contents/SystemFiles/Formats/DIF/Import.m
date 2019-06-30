(* ::Package:: *)

Begin["System`Convert`DIFDump`"]


ImportExport`RegisterImport[
 "DIF",
 ImportDIF,
 {
   "Grid"->ImportDIFGrid
 },
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"Data", "Grid"},
 "DefaultElement" -> "Data",
 "BinaryFormat" -> True
]


End[]
