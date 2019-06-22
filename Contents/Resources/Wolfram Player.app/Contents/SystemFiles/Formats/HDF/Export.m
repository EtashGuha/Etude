(* ::Package:: *)

ImportExport`RegisterExport[
  "HDF",
  System`Convert`HDFDump`ExportHDFElements,
  "DefaultElement"->"Datasets",
  "Sources" -> ImportExport`DefaultSources[{"HDF","DataCommon"}],
  "BinaryFormat" -> True
]
