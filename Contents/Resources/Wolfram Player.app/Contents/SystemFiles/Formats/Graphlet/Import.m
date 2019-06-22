(* ::Package:: *)

Begin["System`Convert`GraphletDump`"]


ImportExport`RegisterImport[
 "Graphlet",
 {
  "Elements"   :> ImportGMLElements,
  "Graph"      :> ImportGMLGraph,
  "EdgeRules"  :> ImportGMLEdgeRules,
  "Graphics"   :> ImportGMLGraphics,
  "DirectedEdges" :> ImportGMLDirectedEdges,
  ImportGMLGraphData
 },
 {},
 "FunctionChannels" -> {"Streams"},
 "DefaultElement" -> "Graph",
 "AvailableElements" -> {
    "AdjacencyMatrix", "DirectedEdges", "EdgeAttributes", "EdgeRules", "Graph",
    "GraphAttributes", "Graphics", "VertexAttributes", "VertexCount", "VertexList"
 },
 "Options" -> {}
]


End[]
