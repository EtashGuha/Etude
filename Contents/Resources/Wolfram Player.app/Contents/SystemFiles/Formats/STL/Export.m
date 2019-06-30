(* ::Package:: *)

ImportExport`RegisterExport["STL",
	System`Convert`STLDump`ExportSTL,
	"Sources" -> {"Convert`Common3D`", "Convert`STL`"},
	"FunctionChannels" -> {"Streams"},
	"Options" -> {"BinaryFormat", "SurfaceOrientation", "VerticalAxis"},
	"DefaultElement"->"Graphics3D",
	"BinaryFormat" -> True
]
