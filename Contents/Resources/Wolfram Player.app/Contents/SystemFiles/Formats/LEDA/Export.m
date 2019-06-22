(* ::Package:: *)

ImportExport`RegisterExport[
    "LEDA",
	System`Convert`LEDADump`ExportLEDA,
	"FunctionChannels" -> {"Streams"},
	"Options" -> {DirectedEdges, "DirectedEdges"},
	"DefaultElement" -> "Automatic"
]
