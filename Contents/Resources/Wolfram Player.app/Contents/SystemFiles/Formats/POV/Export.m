(* ::Package:: *)

ImportExport`RegisterExport[
 "POV",
 System`Convert`POVDump`ExportPOV,
 "Sources" -> {"Convert`Common3D`", "Convert`POV`"},
 "FunctionChannels" -> {"Streams"},
 "Unevaluated" -> False,
 "Options"->{"Header", "VerticalAxis", "Comments"},
 "DefaultElement"->"Graphics3D"
]
