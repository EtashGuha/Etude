(* ::Package:: *)

ImportExport`RegisterExport["XHTML",
	System`Convert`HTMLDump`exportUsingHTMLSave,
	"Sources" -> {"Convert`ConvertCommon`", "Convert`MLStringData`", "Convert`HTMLConvert`"},
	"Unevaluated" -> False
]
