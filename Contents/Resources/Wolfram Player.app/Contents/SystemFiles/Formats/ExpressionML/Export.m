(* ::Package:: *)

ImportExport`RegisterExport[
  "ExpressionML",
  System`Convert`NotebookMLDump`exprmlSave,
  "Sources" -> {"Convert`ConvertCommon`", "Convert`MLStringData`", "Convert`ConvertInit`"},
  "FunctionChannels" -> {"FileNames", "Streams"},
  "DefaultElement" -> "Expression"
]
