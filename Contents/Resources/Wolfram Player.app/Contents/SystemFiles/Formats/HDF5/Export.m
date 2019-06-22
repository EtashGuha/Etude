(* ::Package:: *)

ImportExport`RegisterExport[
  "HDF5",
  System`Convert`HDF5Dump`ExportHDF5,
  "DefaultElement"->"Datasets",
  "Sources" -> ImportExport`DefaultSources[{"HDF5","DataCommon"}],
  "BinaryFormat" -> True
]
