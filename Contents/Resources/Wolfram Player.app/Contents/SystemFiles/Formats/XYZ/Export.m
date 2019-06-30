
Begin["System`Convert`XYZDump`"]

ImportExport`RegisterExport[
	"XYZ",
	ExportXYZ,
	"FunctionChannels" -> {"Streams"},
	"Sources" -> ImportExport`DefaultSources["XYZ"],
	"DefaultElement" -> "Molecule"
];

End[]
