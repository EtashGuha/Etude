(* ::Package:: *)

Begin["System`Convert`TableDump`"]

ImportExport`RegisterExport["CSV",
	System`Convert`TableDump`ExportSV["CSV", ##] &,
	"Sources" -> ImportExport`DefaultSources["Table"],
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> "Data",
	"BinaryFormat" -> True
]

End[]