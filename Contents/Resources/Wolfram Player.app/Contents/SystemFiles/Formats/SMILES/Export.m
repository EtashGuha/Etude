
Begin["System`Convert`SMILESDump`"]

ImportExport`RegisterExport[
	"SMILES",
	ExportSMILES,
	"FunctionChannels" -> {"Streams"},
	"Sources"->ImportExport`DefaultSources["SMILES"],
	"DefaultElement" -> "Molecule"
]

End[]
