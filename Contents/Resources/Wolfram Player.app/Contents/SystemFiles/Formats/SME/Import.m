(* ::Package:: *)

ImportExport`RegisterImport[
  "SME",
  {
    "SystemModelSimulationData" :> WSM`PackageScope`loadDataFromSme
  },
  "AvailableElements" -> {"SystemModelSimulationData"},
  "DefaultElement" -> "SystemModelSimulationData",
  "Sources" -> {"WSM`"}
]
