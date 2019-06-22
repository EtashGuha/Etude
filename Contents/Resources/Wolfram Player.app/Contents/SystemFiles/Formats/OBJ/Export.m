(* ::Package:: *)

ImportExport`RegisterExport[
 "OBJ",
 System`Convert`OBJDump`ExportOBJ,
 "Sources" -> ImportExport`DefaultSources[{"Common3D", "OBJ"}],
 "FunctionChannels" -> {"Streams"},
 "DefaultElement"->"Graphics3D",
 "Options" -> {"InvertNormals", "VerticalAxis", "Comments"},
 "BinaryFormat" -> True
]
