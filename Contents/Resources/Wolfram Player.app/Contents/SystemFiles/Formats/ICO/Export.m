(* ::Package:: *)

ImportExport`RegisterExport[
  "ICO",
  System`Convert`ICODump`ExportItem["ICO"][##]&,
  "FunctionChannels" ->  {"Streams"},
  "BinaryFormat" -> True
]