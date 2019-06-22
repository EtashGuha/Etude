(* ::Package:: *)

ImportExport`RegisterExport[
  "JVX", 
  System`Convert`JVXDump`ExportJVX,
  "Options" -> {"Summary", "ShortSummary", "Keywords", "Title", "Version"},
  "Sources" -> {"Convert`Common3D`", "Convert`JVX`"},
  "FunctionChannels" -> {"Streams"},
  "Options" -> {"InvertNormals", "VerticalAxis"},
  "BinaryFormat" -> True
]
