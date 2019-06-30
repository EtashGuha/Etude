(* ::Package:: *)

ImportExport`RegisterExport[
    "String",
	System`Convert`TextDump`exportStringFormat,
	"FunctionChannels" -> {"Streams"},
	"Sources" -> ImportExport`DefaultSources["Text"],
	"BinaryFormat" -> True
]
