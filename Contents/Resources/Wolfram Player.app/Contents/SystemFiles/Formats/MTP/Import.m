(* ::Package:: *)

Begin["System`Convert`MTPDump`"]


ImportExport`RegisterImport[
 "MTP",
 {
  "TextData" :> ImportMTPText,
  "LabeledData" :> ImportMTP,
  "Data" :> ImportMTPData,
  ImportMTP
 },
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"Data", "LabeledData", "TextData"},
 "DefaultElement" -> {"LabeledData"}
]


End[]
