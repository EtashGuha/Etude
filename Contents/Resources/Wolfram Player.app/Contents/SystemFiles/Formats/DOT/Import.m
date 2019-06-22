(* ::Package:: *)

Begin["System`Convert`DOTDump`"]


ImportExport`RegisterImport[
 "DOT",
 DOTRawImporter,
 {
 	"AdjacencyMatrix" :> ImportDOTAdjacencyMatrix,
 	"Graph" :> ImportDOTGraph,
 	"Graphics":>ImportDOTGraphics
 },
 "Sources" -> {"Convert`DOT`", "Convert`MLStringData`"},
 "FunctionChannels" -> {"FileNames"},
 "DefaultElement" -> "Graph",
 "AvailableElements" -> {
    "AdjacencyMatrix", "EdgeAttributes", "EdgeRules", "Graph", "GraphAttributes",
    "Graphics", "VertexAttributes", "VertexCount", "VertexList"
 },
 "Options" -> {}
]


End[]
