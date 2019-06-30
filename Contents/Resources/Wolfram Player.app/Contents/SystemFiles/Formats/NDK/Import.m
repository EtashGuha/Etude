(* ::Package:: *)

Begin["System`Convert`NDKDump`"]


ImportExport`RegisterImport[
    "NDK",
	ImportNDK,
  {
	"Data":> getNDKData,
	"Labels" :> getNDKLabels
  },
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> {"Data", "LabeledData", "Labels"},
	"DefaultElement" -> "Data"
]




End[]
