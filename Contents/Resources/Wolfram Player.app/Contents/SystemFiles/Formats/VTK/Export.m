(* ::Package:: *)

ImportExport`RegisterExport[
  "VTK",
  System`Convert`VTKDump`ExportVTK,
  "Sources" 			-> {"Convert`Common3D`", "Convert`VTK`"},
  "FunctionChannels" 	-> {"Streams"},
  "Options" 			-> {"BinaryFormat", "VerticalAxis"},
  "BinaryFormat"		-> True
]
