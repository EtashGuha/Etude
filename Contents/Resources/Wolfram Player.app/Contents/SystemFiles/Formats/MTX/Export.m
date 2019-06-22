(* ::Package:: *)

ImportExport`RegisterExport[
 "MTX",
 System`Convert`MatrixMarketDump`ExportMatrixMarket,
 "Sources" -> ImportExport`DefaultSources["MatrixMarket"],
 "Options" -> {"Comments", "MatrixStructure"},
 "DefaultElement" -> "Data"
]
