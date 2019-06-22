(* ::Package:: *)

Begin["System`Convert`FASTQDump`"]


ImportExport`RegisterImport[
 "FASTQ",
 {
  (*Fast header-only import*)
  "Header":> HeaderImport,
  (***************************************) 
  (*Slow all-element import*)
  FASTQImport},
 (***************************************) 
 {
  (*Post-processors*)
  "LabeledData" :> LabeledDataFormat,
  "Data" :> DataFormat
  },
 "AvailableElements" -> {"Data", "Header", "LabeledData", "Length", "Sequence", "Qualities"},
 "DefaultElement" -> "Sequence"
 ]

End[]