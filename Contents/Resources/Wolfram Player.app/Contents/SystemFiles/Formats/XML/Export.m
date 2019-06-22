(* ::Package:: *)

ImportExport`RegisterExport[
  "XML",
  System`Convert`XMLDump`ExportXML,
  "Sources" -> {"Convert`ConvertCommon`", "Convert`MLStringData`", "Convert`ConvertInit`"},
  "FunctionChannels" -> {"FileNames", "Streams"}
  (* No "DefaultElement" on purpose.
  "DefaultElement" ->  *)
]
