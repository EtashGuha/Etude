(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
 "SCT",
 ExportRasterElements["SCT", System`Convert`SCTDump`ExportSCT, ##] &,
 "FunctionChannels" -> {"Streams"},
 "DefaultElement" -> Automatic,
 "Options" -> {"ImageResolution"},
 "BinaryFormat" -> True
 ]


End[]
