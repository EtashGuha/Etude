(* ::Package:: *)

Begin["System`Convert`GXLDump`"]


ImportExport`RegisterImport[
 "GXL",
 {
   "Graph"    :> ImportGXLGraph,
   "Graphics" :> ImportGXLGraphics,
   "Elements" :> ImportGraphElements,
   Automatic  :> ImportGXLDefault,
   s_String  :> (CommonReturn[All, s, ##]&),
   {s_String, n_?((StringQ[#]&&(!StringMatchQ[#,"Elements"]))&) }:> (CommonReturn[n, s, ##]&),
   ImportGraphElements
 },
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"AdjacencyMatrix", "DirectedEdges", "EdgeAttributes", "EdgeLabels",
      "EdgeRules", "EdgeTypes", "Graph", "GraphList", "Graphics", "GraphicsList", "GraphNames",
      "VertexAttributes", "VertexCount", "VertexList", "VertexTypes"}
]


End[]
