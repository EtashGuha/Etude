(* ::Package:: *)

ImportExport`RegisterExport[
 "RIB",
 System`Convert`RIBDump`ExportRIB,
 "Sources"				-> {"Convert`Common3D`", "Convert`RIB`"},
 "FunctionChannels"	-> {"Streams"},
 "Options" 			-> {"Comments", "Scene", "Creator", "CreationDate", "VerticalAxis", "InvertNormals"},
 "DefaultElement"		->"Graphics3D"
]
