(* ::Package:: *)

Begin["System`Convert`TableDump`"]

ImportExport`RegisterExport["TSV",
	System`Convert`TableDump`ExportSV["TSV", ##] &,
	"Sources" -> ImportExport`DefaultSources["Table"],
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> "Data",
	"BinaryFormat" -> True
]

End[]