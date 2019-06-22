(* ::Package:: *)

Begin["System`Convert`MolDump`"]

ImportExport`RegisterExport[
	"SDF",
	ExportSDF,
	"FunctionChannels" -> {"FileNames"},
	"Sources"->ImportExport`DefaultSources["Mol"],
	"DefaultElement" -> "Molecule"
]

End[]
