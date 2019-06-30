(* ::Package:: *)

Begin["System`Convert`MolDump`"]

ImportExport`RegisterExport[
	"MOL",
	ExportMOL,
	"FunctionChannels" -> {"FileNames"},
	"Sources" -> ImportExport`DefaultSources["Mol"],
	"DefaultElement" -> "Molecule"
];

End[]
