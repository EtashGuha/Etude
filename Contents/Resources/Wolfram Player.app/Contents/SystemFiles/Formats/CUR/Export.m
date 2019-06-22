(* ::Package:: *)

ImportExport`RegisterExport[
  "CUR",
  System`Convert`ICODump`ExportItem["CUR"][##] &,
  "FunctionChannels" ->  {"Streams"},
  "BinaryFormat" -> True,
  "Sources" -> ImportExport`DefaultSources["ICO"],
  "Options" -> {"HotSpot"}
]