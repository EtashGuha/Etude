(* ::Package:: *)

ImportExport`RegisterExport[
  "FMU",
  WSMLink`Utilities`exportFMU,
  "Options" -> Sort@{"FMIVersion", "FMIKind", "IncludeProtectedVariables"},
  "DefaultElement" -> Automatic,
  "Sources" -> {"WSM`"}
]
