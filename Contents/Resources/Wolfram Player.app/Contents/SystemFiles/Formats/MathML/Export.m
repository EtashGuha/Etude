(* ::Package:: *)

ImportExport`RegisterExport[
  "MathML",
  System`Convert`MathMLDump`mathMLExport,
  "Sources" -> {"Convert`ConvertCommon`", "Convert`MLStringData`", "Convert`ConvertInit`"},
  "FunctionChannels" -> {"FileNames", "Streams"},
  "DefaultElement"->Automatic
]
