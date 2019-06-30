(* ::Package:: *)

ImportExport`RegisterExport[
    "LWO",
    System`Convert`LWODump`ExportLightwave,
	"Sources" -> {"Convert`Common3D`","Convert`LWO`"},
	"FunctionChannels" -> {"Streams"},
	"DefaultElement"->"Graphics3D",
	"Options" -> {"VerticalAxis"},
	"BinaryFormat" -> True
]
