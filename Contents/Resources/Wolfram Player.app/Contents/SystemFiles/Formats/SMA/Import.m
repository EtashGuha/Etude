(* ::Package:: *)

ImportExport`RegisterImport[
  "SMA",
  {
    "SystemModel" :> WSM`PackageScope`loadModelFromSma
  },
  "AvailableElements" -> {"SystemModel"},
  "DefaultElement" -> "SystemModel",
  "Sources" -> {"WSM`"}
]
