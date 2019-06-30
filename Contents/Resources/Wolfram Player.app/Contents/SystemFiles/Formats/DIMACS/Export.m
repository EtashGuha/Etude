(* ::Package:: *)

ImportExport`RegisterExport[
  "DIMACS", 
  System`Convert`DIMACSDump`ExportDIMACS,
  "FunctionChannels" -> {"Streams"},
  "DefaultElement" -> "Automatic",
  "Options" -> {"BinaryFormat"},
  "BinaryFormat" -> True
]
