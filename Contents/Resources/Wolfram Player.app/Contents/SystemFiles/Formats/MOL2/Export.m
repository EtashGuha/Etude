(* ::Package:: *)

Begin["System`Convert`MOL2Dump`"]

ImportExport`RegisterExport[
	"MOL2",
	ExportMol2,
	"FunctionChannels" -> {"Streams"},
	"Sources" -> ImportExport`DefaultSources["MOL2"],
	"DefaultElement" -> "Molecule"
]

End[]
