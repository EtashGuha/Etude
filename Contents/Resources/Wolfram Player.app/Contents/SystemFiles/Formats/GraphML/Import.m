(* ::Package:: *)

Begin["System`Convert`GraphMLDump`"]


ImportExport`RegisterImport[
 "GraphML", 
 {
    "Graph"    :> ImportGraphMLGraph,
    "Graphics" :> ImportGraphMLGraphics,
    "Elements" :> ImportGraphElements,
    Automatic  :> ImportGraphMLDefault,
    s_String   :> (CommonReturn[All, s, ##]&),
    {s_String, n_?((StringQ[#]&&(!StringMatchQ[#,"Elements"]))&) }:> (CommonReturn[n, s, ##]&),
    ImportGraphElements
 },
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {
   "AdjacencyMatrix", "DirectedEdges", "EdgeAttributes", "EdgeLabels", "EdgeRules", "Graph", "GraphList", "Graphics", 
   "GraphicsList", "GraphNames", "VertexAttributes", "VertexCount", "VertexList"
  }
]


End[]
