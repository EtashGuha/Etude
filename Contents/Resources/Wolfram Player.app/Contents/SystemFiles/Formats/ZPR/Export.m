(* ::Package:: *)

ImportExport`RegisterExport[
  "ZPR",
  System`Convert`ZPRDump`ExportZPR,
  "Sources" -> {"Convert`Common3D`", "Convert`ZPR`"},
  "Options" -> {"VerticalAxis"},
  "BinaryFormat" -> True
]
