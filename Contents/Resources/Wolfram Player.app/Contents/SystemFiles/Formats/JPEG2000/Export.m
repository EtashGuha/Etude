(* ::Package:: *)

Begin["System`Convert`JPEG2000Dump`"]


ImportExport`RegisterExport[
  "JPEG2000",
  System`ConvertersDump`ExportRasterElements["JPEG2000", ExportJPEG2000, ##] &,
  "DefaultElement" -> Automatic,
  "Options" -> {"BitDepth", "CompressionLevel", "ImageEncoding"},
  "Sources" -> {"JLink`", "Convert`JPEG2000`"},
  "BinaryFormat" -> True,
  "AlphaChannel" -> True

]


End[]
