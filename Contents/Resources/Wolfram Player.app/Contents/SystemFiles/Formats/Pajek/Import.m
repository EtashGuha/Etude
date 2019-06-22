(* ::Package:: *)

Begin["System`Convert`PajekDump`"]


ImportExport`RegisterImport[
 "Pajek",
 ImportPajek,
 {
 	"AdjacencyMatrix" :> ImportPajekAdjacencyMatrix,
 	"Graphics" :> ImportPajekGraphics,
	"Graph" :> ImportPajekGraph
 },
 "FunctionChannels" -> {"Streams"},
 "DefaultElement" -> "Graph",
 "AvailableElements" -> {
    "AdjacencyMatrix", "EdgeAttributes", "EdgeRules", "EdgeRulesDirected",
    "EdgeRulesUndirected", "Graph", "Graphics", "VertexAttributes", "VertexCount"
 },
 "Options" -> {}
]


End[]
