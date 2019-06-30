(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
  "PXR",
  ExportRasterElements["PXR", System`Convert`PXRDump`ExportPXR, ##] &,
  "FunctionChannels" -> {"Streams"},
  "DefaultElement" -> "Graphics",
  "Options" -> {},
  "BinaryFormat" -> True
]


End[]
