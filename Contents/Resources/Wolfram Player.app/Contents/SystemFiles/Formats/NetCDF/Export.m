(* ::Package:: *)

ImportExport`RegisterExport[
 "NetCDF",
 System`Convert`NetCDFDump`ExportNetCDF,
 "Sources" -> Join[{"JLink`"}, ImportExport`DefaultSources[{"NetCDF","DataCommon"}]],
 "DefaultElement" -> "Datasets",
 "BinaryFormat" -> True
]
