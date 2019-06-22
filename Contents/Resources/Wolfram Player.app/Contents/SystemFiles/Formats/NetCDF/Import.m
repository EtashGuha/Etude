(* ::Package:: *)

Begin["System`Convert`NetCDFDump`"]


ImportExport`RegisterImport[
 "NetCDF",
 {
 	"Elements" :> getNetCDFElements["NetCDF"],
 	"Metadata":> GetNetCDFGlobalMetadata["NetCDF"],
 	{"Data"} :> getNetCDFData["NetCDF"],
 	"Datasets" -> getDatasetsNames["NetCDF"],
 	{"Datasets", name:(_String|_Integer)} :> ({"Datasets"-> {name->getVarData["NetCDF"][name][##]}}&),
 	{"Annotations" | "DataFormat" | "Dimensions"} :> NetCDFGetInfo["NetCDF"],
  	GetNetCDFMetadata["NetCDF"]["String"]
 },
 "Sources" -> Join[{"JLink`"}, ImportExport`DefaultSources["NetCDF"]],
 "AvailableElements" -> {"Annotations", "Data", "DataFormat", "Datasets", "Dimensions", "Metadata"},
 "DefaultElement" -> "Datasets",
 "BinaryFormat" -> True
]


End[]
