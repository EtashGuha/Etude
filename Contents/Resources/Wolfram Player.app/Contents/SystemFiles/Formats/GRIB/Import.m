(* ::Package:: *)

Begin["System`Convert`NetCDFDump`"]


ImportExport`RegisterImport[
  "GRIB",
  {
 	"Elements" :> getNetCDFElements["GRIB"],
 	"Metadata":> GetNetCDFGlobalMetadata["GRIB"],
 	{"Data"} :> getNetCDFData["GRIB"],
 	"Datasets" -> getDatasetsNames["GRIB"],
 	{"Datasets", name:(_String|_Integer)} :> ({"Datasets"-> {name->getVarData["GRIB"][name][##]}}&),
 	{"Annotations" | "DataFormat" | "Dimensions"} :> NetCDFGetInfo["GRIB"],
  	GetNetCDFMetadata["GRIB"]["String"]
  },
  "Sources" -> Join[{"JLink`"}, ImportExport`DefaultSources["NetCDF"]],
  "AvailableElements" -> {"Annotations", "Data", "DataFormat", "Datasets", "Dimensions", "Metadata"},
  "DefaultElement" -> "Datasets",
  "BinaryFormat" -> True
]


End[]
