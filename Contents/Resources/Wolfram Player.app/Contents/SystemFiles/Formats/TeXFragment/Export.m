(* ::Package:: *)


ImportExport`RegisterExport[
	"TeXFragment",
	System`Convert`TeXDump`ExportTeXFragment,
	"AvailableElements" -> {"Notebook", "NotebookObject"},
	"DefaultElement" -> "Notebook", 
	"Sources" -> {"Convert`ConvertCommon`", "Convert`TeXForm`", "Convert`TeXConvert`"}
]
