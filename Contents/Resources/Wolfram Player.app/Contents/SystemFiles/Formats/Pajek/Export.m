(* ::Package:: *)

ImportExport`RegisterExport[
    "Pajek", 
	System`Convert`PajekDump`ExportPajek,
	"FunctionChannels" -> {"Streams"},
    "Options" -> {"DirectedEdges", DirectedEdges, "VertexCoordinates", VertexCoordinates},
    "DefaultElement" -> "Automatic"
]
