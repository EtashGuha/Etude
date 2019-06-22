(* ::Package:: *)

ImportExport`RegisterExport[
 "Maya",
 System`Convert`MayaDump`ExportMA,
 "Sources" -> {"Convert`Common3D`","Convert`Maya`"},
 "FunctionChannels" -> {"Streams"},
 "Options"->{"Comments", "VerticalAxis"},
 "DefaultElement" -> "Graphics3D",
 "BinaryFormat" -> True
]
