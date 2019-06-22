(* ::Package:: *)

ImportExport`RegisterExport["Graphlet",
	System`Convert`GraphletDump`ExportGML,
	"FunctionChannels" -> {"Streams"},
	"Options" -> {"DirectedEdges", DirectedEdges},
	"DefaultElement" -> Automatic
]
