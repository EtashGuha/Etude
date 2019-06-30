(* ::Package:: *)

ImportExport`RegisterExport[
	"List",
	System`Convert`TableDump`ExportList,
	"Sources" -> ImportExport`DefaultSources["Table"],
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> "Data"
]
