(* ::Package:: *)

Begin["System`Convert`JCAMPDXDump`"]


ImportExport`RegisterImport[
  "JCAMP-DX",
  {
    "Data"    :> ImportData,
    ImportMetadata
  },
  "AvailableElements" -> {"Data", "Metadata"},
  "DefaultElement" -> "Data",
  "Sources" -> ImportExport`DefaultSources[{"JCAMPDX"}]
]


End[]
