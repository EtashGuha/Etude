(* ::Package:: *)

Begin["System`Convert`BinaryDump`"]


ImportExport`RegisterExport[
 "Binary",
 ExportBinary,
 "Sources" -> ImportExport`DefaultSources["Binary"],
 "FunctionChannels"-> {"Streams"},
 "Options" -> {"DataFormat"},
 "DefaultElement" -> Automatic,
 "BinaryFormat" -> True
]


End[]
