(* ::Package:: *)

Begin["System`Convert`MatrixMarketDump`"]


ImportExport`RegisterImport[
 "MTX",
 ImportMatrixMarket,
 {"Graphics" :> ToMTXGraphics},
 "Sources" -> ImportExport`DefaultSources["MatrixMarket"],
 "AvailableElements" -> {"Comments", "Data", "Graphics", "MatrixStructure"},
 "DefaultElement" -> "Data",
 "Options" -> {"Comments", "MatrixStructure"}
]


End[]
