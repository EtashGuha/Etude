(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
  "JPEG",
  System`Convert`CommonGraphicsDump`ExportElementsToRasterFormat["JPEG", ##]&,
  "Sources" -> {"Convert`CommonGraphics`","Convert`Exif`"},
  "DefaultElement" -> Automatic,
  "Options" -> {"BitDepth", "ColorSpace"},
  "BinaryFormat" -> True
]


End[]
