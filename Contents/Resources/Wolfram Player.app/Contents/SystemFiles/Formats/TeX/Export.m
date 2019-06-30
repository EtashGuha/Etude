(* ::Package:: *)


ImportExport`RegisterExport[
	"TeX",
	System`Convert`TeXDump`ExportTeXElements,
	"AvailableElements" -> {"Notebook", "NotebookObject"},
	"DefaultElement" -> "Notebook",
	"Sources" -> {"Convert`ConvertCommon`", "Convert`TeXForm`", "Convert`TeXConvert`"}
]
