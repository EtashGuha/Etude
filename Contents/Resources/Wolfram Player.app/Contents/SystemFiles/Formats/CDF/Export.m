(* Wolfram Computable Document Format *)
ImportExport`RegisterExport["CDF", 
	System`Convert`MathematicaCDFDump`ExportCDF,
	"BinaryFormat" -> False,
	"FunctionChannels" -> {"FileNames"},
	"Sources" -> ImportExport`DefaultSources["CDF"]
]
