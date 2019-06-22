(* ::Package:: *)

ImportExport`RegisterExport[
  "PLY",
  System`Convert`PLYDump`ExportPLY,
  "Sources" 			-> {"Convert`Common3D`", "Convert`PLY`"},
  "FunctionChannels" 	-> {"Streams"},
  "Options" 			-> {"BinaryFormat", "DataFormat", "Comments", "InvertNormals", "VerticalAxis"},
  "DefaultElement"		-> "Graphics3D",
  "BinaryFormat"		-> True
]
