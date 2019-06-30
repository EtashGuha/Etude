(* ::Package:: *)

Begin["System`Convert`HarwellBoeingDump`"]


ImportExport`RegisterImport[
 "HarwellBoeing",
 ImportHB,
 {"Graphics" :> ToHBGraphics},
 "AvailableElements" -> {"Data", "Graphics", "Key", "MatrixStructure", "Title"},
 "DefaultElement" -> "Data",
 "Options" -> {"MatrixStructure", "Title", "Key"}
]


End[]
