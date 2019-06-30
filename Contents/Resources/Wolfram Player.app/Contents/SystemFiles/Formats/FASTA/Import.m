(* ::Package:: *)

Begin["System`Convert`FASTADump`"]


ImportExport`RegisterImport[
  "FASTA",
  getAllFASTAData,
  "AvailableElements" -> {"Accession", "Data", "Description", "GenBankID", "Header", "LabeledData", "Length", "Plaintext", "Sequence"},
  "DefaultElement" -> "Sequence",
  "FunctionChannels" -> {"FileNames"},
  "Options" -> {"HeaderFormat","ToUpperCase"->True}
]


End[]
