(* ::Package:: *)

ImportExport`RegisterImport[
	"String",
	System`Convert`TextDump`StringImport,
	"FunctionChannels" -> {"Streams"},
	"Sources"	-> ImportExport`DefaultSources["Text"],
	"AvailableElements" -> {"String"},
	"DefaultElement" -> "String",
	"BinaryFormat" -> True
]
