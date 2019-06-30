(* ::Package:: *)

Begin["System`Convert`BinaryDump`"]


ImportExport`RegisterExport[
 "Bit",
 ExportBits,
 "FunctionChannels"-> {"Streams"},
 "Sources" -> ImportExport`DefaultSources["Binary"],
 "BinaryFormat" -> True
]


End[]
