(* ::Package:: *)

ImportExport`RegisterExport[
  "Sparse6",
  System`Convert`GraphDump`ExportSparse6,
  "Sources" ->ImportExport`DefaultSources["Graph"],
  "FunctionChannels" -> {"Streams"}
  (* "DefaultElement" -> explicitly not included *)
]
