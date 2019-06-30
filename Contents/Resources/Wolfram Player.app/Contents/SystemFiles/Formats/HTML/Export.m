(* ::Package:: *)

ImportExport`RegisterExport[
  "HTML",
  System`Convert`HTMLDump`exportUsingHTMLSave,
  "Sources" -> {"Convert`ConvertCommon`", "Convert`MLStringData`", "Convert`HTMLConvert`"},
  "Unevaluated" -> False
]
