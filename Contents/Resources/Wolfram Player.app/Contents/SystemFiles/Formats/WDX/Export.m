(* ::Package:: *)

ImportExport`RegisterExport[
  "WDX",
  System`Convert`WDXDump`ExportWDX,
  "DefaultElement" -> "Expression",
  "FunctionChannels" -> {"FileNames"},
  "BinaryFormat" -> True
]
