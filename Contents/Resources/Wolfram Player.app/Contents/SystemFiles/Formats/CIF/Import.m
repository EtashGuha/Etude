(* ::Package:: *)

Begin["System`Convert`CIFDump`"]


ImportExport`RegisterImport[
  "CIF",
  {
	"Elements" :> getCIFElements,
	ImportCIF
  },
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"Comments", "Data"},
  "DefaultElement" -> "Data",
  "BinaryFormat" -> False
]


End[]
