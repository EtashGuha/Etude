(* ::Package:: *)

Begin["System`Convert`RSSDump`"]


ImportExport`RegisterImport[
    "RSS",
	RSSToSymbolicXML,
	{
		"Notebook" -> ToNotebookElement,
		"NotebookObject" -> ToNotebookObjectElement
	},
	"OriginalChannel" -> True,
	"AvailableElements" -> {"Notebook", "NotebookObject", "SymbolicXML"},
	"DefaultElement" -> "Notebook"
]


End[]
