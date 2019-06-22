(* ::Package:: *)

ImportExport`RegisterImport[
  "MO",
  {
    "SystemModel" :> WSM`PackageScope`loadModelFromMo
  },
  "AvailableElements" -> {"SystemModel"},
  "DefaultElement" -> "SystemModel",
  "Sources" -> {"WSM`"}
]
