(* ::Package:: *)

ImportExport`RegisterExport[
 "3DS", 
 System`Convert`ThreeDSDump`Export3DS,
 "Sources" -> {"Convert`Common3D`", "Convert`ThreeDS`"},
 "FunctionChannels" -> {"Streams"},
 "DefaultElement"->"Graphics3D",
 "Options" -> {"VerticalAxis"},
 "BinaryFormat" -> True
]

