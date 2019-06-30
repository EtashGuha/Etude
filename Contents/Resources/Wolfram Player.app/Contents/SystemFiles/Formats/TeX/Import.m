(* ::Package:: *)

Begin["System`Convert`TeXImportDump`"]

ImportExport`RegisterImport[
  "TeX",
  ImportTeXElements,
  {
      "Notebook" :> ElementsToNotebook,
      "NotebookObject" :> ElementsToNotebookObject,
      "Cell" :> ElementsToCell
  },
  "Sources" -> ImportExport`DefaultSources["TeXImport"],
  "AvailableElements" -> {"Notebook", "NotebookObject", "Cell"},
  "DefaultElement" -> "Notebook"
]


End[]
