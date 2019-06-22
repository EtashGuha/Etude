(* ::Package:: *)

Begin["System`Convert`HDFDump`"]


ImportExport`RegisterImport[
  "HDF",
  {
	{"Datasets"|"Dimensions"|"DataFormat"} :> HDFGetMetadata,
	{"Datasets", name_String} :> HDFGetDataset[name],
	{"Datasets", name_Integer} :> HDFGetDataset[name],
	"Data" :> HDFGetData,
	HDFGetDatasetElements
  },
  "AvailableElements" -> {"Data", "DataFormat", "Datasets", "Dimensions"},
  "DefaultElement"->"Datasets",
  "BinaryFormat" -> True
]


End[]
