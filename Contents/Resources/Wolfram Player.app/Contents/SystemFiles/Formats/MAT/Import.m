(* ::Package:: *)

Begin["System`Convert`MATDump`"]


ImportExport`RegisterImport[
 "MAT",
 ImportMAT,
 {
   "Data" -> ToDataElement,
   "LabeledData"->ToLabeledData,
   "Labels"->ToLabels
 },
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"Comments", "Data", "LabeledData", "Labels"},
 "DefaultElement" -> "Data",
 "BinaryFormat" -> True
]


End[]
