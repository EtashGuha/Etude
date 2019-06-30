(* ::Package:: *)

Begin["System`Convert`BinaryDump`"]


ImportExport`RegisterImport[
 "Binary",
 {
   {"Elements"} -> BinaryElements,
   {Automatic} -> ImportBinaryAutomatic,
   {type_, ___} :> ImportBinary[type]
 },
 {},
 "Sources" -> ImportExport`DefaultSources["Binary"],
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"Bit", "Byte", "Character16", "Character32", "Character8", "Complex128",
			"Complex256", "Complex64", "Integer128", "Integer16", "Integer24",
			"Integer32", "Integer64", "Integer8", "Real128", "Real32", "Real64",
			"TerminatedString", "UnsignedInteger128", "UnsignedInteger16",
			"UnsignedInteger24", "UnsignedInteger32", "UnsignedInteger64",
			"UnsignedInteger8"},
 "DefaultElement" -> Automatic,
 "BinaryFormat" -> True
]


End[]
