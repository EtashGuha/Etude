(* ::Package:: *)

ImportExport`RegisterExport[
  "SurferGrid",
  System`Convert`SurferDump`ExportSurfer,
  "FunctionChannels" -> {"FileNames"},
  "DefaultElement" -> "Data",
  "Options" -> {"BinaryFormat", "DefaultElevation", "DownsamplingFactor", "LegacyFormat", "SpatialRange"},
  "BinaryFormat" -> True
]
