(* ::Package:: *)

Begin["System`Convert`DIMACSDump`"]


ImportExport`RegisterImport[
 "DIMACS",
 ImportDIMACS,
 {
    "AdjacencyMatrix" :> ImportDIMACSedgeAdjacencyMatrix,
    "Graphics" :> ImportDIMACSedgeGraphics
 },
 "FunctionChannels" -> {"Streams"},
 "DefaultElement" -> "Graph",
 "AvailableElements" -> { "AdjacencyMatrix", "EdgeRules", "Graph", "Graphics", "VertexCount"},
 "Options" -> {},
 "BinaryFormat" -> True
]


End[]
