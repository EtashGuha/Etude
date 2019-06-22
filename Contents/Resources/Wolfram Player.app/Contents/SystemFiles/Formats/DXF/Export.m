(* ::Package:: *)

ImportExport`RegisterExport[
 "DXF",
 System`Convert`DXFDump`ExportDXF,
 "Sources" -> {"Convert`Common3D`", "Convert`DXF`"},
 "DefaultElement" -> "Graphics3D",
 "FunctionChannels" -> {"Streams"},
 "Options"->{"ColorSupport", "Comments"}
]
