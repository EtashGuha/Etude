(* ::Package:: *)

ImportExport`RegisterExport[
 "DAE",
 System`Convert`DAEDump`ExportDAE,
 "Sources" -> ImportExport`DefaultSources[{"Common3D", "DAE"}],
 "FunctionChannels" -> {"Streams"},
 "DefaultElement"->"Graphics3D",
 "Options" -> {"InvertNormals", "VerticalAxis", "Comments"},
 "BinaryFormat" -> True
]
