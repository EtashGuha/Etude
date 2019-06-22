(* ::Package:: *)

ImportExport`RegisterExport[
    "NOFF",
	System`Convert`OFFDump`ExportOFF,
	"Sources" 			-> ImportExport`DefaultSources[{"Common3D", "OFF"}],
	"FunctionChannels"	-> {"Streams"},
	"Options"			-> {"BinaryFormat", "ColorSupport", "InvertNormals", "VerticalAxis"},
	"Options"			-> {"BinaryFormat", "InvertNormals", "VerticalAxis"},
	"DefaultElement"	-> "Graphics3D",
	"BinaryFormat" 	    -> True
]
