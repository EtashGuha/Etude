(* ::Package:: *)

Begin["System`Convert`NotebookDump`"]


ImportExport`RegisterImport[
 "RTF",
 ImportRTF,
 {
  "NotebookObject" -> ToNotebookObjectElement
 },
 "Sources" -> ImportExport`DefaultSources["Notebook"],
 "FunctionChannels" -> {"FileNames"},
 "AvailableElements" -> {"Notebook", "NotebookObject"},
 "DefaultElement" -> "Notebook",
 "BinaryFormat" -> True
]


End[]
