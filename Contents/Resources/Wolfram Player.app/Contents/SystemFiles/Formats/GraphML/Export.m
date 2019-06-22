(* ::Package:: *)

ImportExport`RegisterExport["GraphML",
	System`Convert`GraphMLDump`ExportGraphML,
	"FunctionChannels" -> {"Streams"},
    "Options" -> {DirectedEdges, "DirectedEdges"},
    "DefaultElement" -> Automatic
]
