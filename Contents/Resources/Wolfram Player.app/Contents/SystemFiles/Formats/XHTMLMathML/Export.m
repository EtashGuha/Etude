(* ::Package:: *)

ImportExport`RegisterExport[
  "XHTMLMathML",
  System`Convert`HTMLDump`exportUsingHTMLSaveWithMathML,
  "Sources" -> {"Convert`ConvertCommon`", "Convert`MLStringData`", "Convert`HTMLConvert`"},
  "Unevaluated" -> False
]
