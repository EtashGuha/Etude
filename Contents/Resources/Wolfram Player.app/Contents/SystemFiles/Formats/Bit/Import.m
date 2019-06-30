(* ::Package:: *)

Begin["System`Convert`BinaryDump`"]


ImportExport`RegisterImport[
 "Bit",
 ImportBits,
 "FunctionChannels"-> {"Streams"},
 "AvailableElements" -> {_Integer},
 "Sources" -> ImportExport`DefaultSources["Binary"],
 "BinaryFormat" -> True
]


End[]
