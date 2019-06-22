(* ::Package:: *)

Begin["System`Convert`DBFDump`"]


ImportExport`RegisterImport[
  "DBF",
  {
 	"Elements" :> getElements,
 	"LabeledData" :> ImportDBF,
 	{"LabeledData", "Elements"} :> extractMetadata,
 	"Labels" :> getDatasets,
 	ImportDBF
  },
  {
 	"Data" :> getData
  },
  "Sources" -> {"JLink`", "Convert`DBF`"},
  "AvailableElements" -> {"Data", "LabeledData", "Labels"}, 
  "DefaultElement" -> "Data",
  "BinaryFormat" -> True
]


End[]
