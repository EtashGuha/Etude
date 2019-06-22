(* ::Package:: *)

Begin["System`Convert`NotebookDump`"]

ImportExport`RegisterImport["NB",
	{
		"Notebook"					-> ImportNB,
		"Plaintext"					-> NotebookImportText,

		(* these two return the same *)
		"Cells"						-> (ImportCellsElement[All][##]&),
		{"Cells","Elements"} 		:> (ImportCellsElement["Elements"][##]&),

		{"Cells",str_String} 		:> (ImportCellsElement[{str}][##]&),
		{"Cells",lst:{__String}}	:> (ImportCellsElement[lst][##]&),	
		ImportNB
	},
	{
		"NotebookObject" -> ToNotebookObjectElement,
		"Initialization" -> ToInitializationElement
	},
	"Sources" -> ImportExport`DefaultSources[{"Notebook", "NBImport"}],
	"DefaultElement" -> "Notebook"
]


End[]
