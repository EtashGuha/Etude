(* ::Package:: *)

ImportExport`RegisterExport[
  "DOT", 
  System`Convert`DOTDump`ExportDOT,
  "FunctionChannels" -> {"Streams"},
  "Options" -> {"DirectedEdges", DirectedEdges, "VertexCoordinates", VertexCoordinates},
  "DefaultElement" -> "Automatic"
]
