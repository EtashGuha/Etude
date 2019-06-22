(* Wolfram Computable Document Format *)
ImportExport`RegisterImport["CDF",
	{
		"Plaintext" 		-> System`Convert`MathematicaCDFDump`ImportCDFText,
		"NotebookObject"	-> System`Convert`MathematicaCDFDump`ImportCDFNBObj,
  		System`Convert`MathematicaCDFDump`ImportCDF
  	},
	"Sources" -> ImportExport`DefaultSources[{"CDF", "NBImport"}],
	"AvailableElements" -> {"Notebook", "NotebookObject", "Plaintext"},
	"FunctionChannels" -> {"FileNames"},
	"BinaryFormat" -> False,
	"DefaultElement" -> "Notebook"
]