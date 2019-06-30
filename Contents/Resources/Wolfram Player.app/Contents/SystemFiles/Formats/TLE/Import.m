(* ::Package:: *)

Begin["System`Convert`TLEDump`"]


ImportExport`RegisterImport[
  "TLE",
  ImportTLE,
  {n_?((IntegerQ[#]||(StringQ[#]&&StringMatchQ[#, RegularExpression["\\d+"]]))&):> getSatellite[n]},
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"Data", "LabeledData", "Labels", _Integer, _String},
  "DefaultElement" -> "LabeledData"
]


End[]
