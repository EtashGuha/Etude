(* ::Package:: *)

ImportExport`RegisterExport[
    "Text",
	System`Convert`TextDump`TextExport,
	"FunctionChannels" -> {"Streams"},
	"Options" -> {"ByteOrderMark"},
	"DefaultElement" -> "Plaintext",
	"BinaryFormat" -> True
]
