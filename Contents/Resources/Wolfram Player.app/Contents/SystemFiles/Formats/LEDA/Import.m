(* ::Package:: *)

Begin["System`Convert`LEDADump`"]


ImportExport`RegisterImport[
  "LEDA",
  {
		ImportLEDA
  },
  {
		"AdjacencyMatrix" :> ImportLEDAAdjacencyMatrix,
		"Graph" :> ImportLEDAGraph,
		"Graphics" :> ImportLEDAGraphics
  },
  "FunctionChannels" -> {"Streams"},
  "DefaultElement" -> "Graph",
  "AvailableElements" -> {
     "AdjacencyMatrix", "EdgeData", "EdgeRules", "EdgeType", "Graph",
     "Graphics", "VertexCount", "VertexData", "VertexType"
  },
  "Options" -> {}
]


End[]
