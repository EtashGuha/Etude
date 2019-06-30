(* ::Package:: *)

Begin["System`Convert`MPSDump`"]


ImportExport`RegisterImport[
 "MPS",
 {
  "Equations" -> ImportMPSEquations,
  ImportMPSElements
 },
 {
  "ConstraintMatrix" -> ToConstraintMatrix
 },
 "AvailableElements" -> {"ConstraintMatrix", "Equations", "LinearProgrammingData"},
 "DefaultElement" -> "Equations"
]


End[]
