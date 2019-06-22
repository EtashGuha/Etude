(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
    "WebP",
    System`Convert`CommonGraphicsDump`ExportElementsToRasterFormat["WebP", ##]&,
    "Sources" -> {"Convert`CommonGraphics`"},
	"DefaultElement" -> Automatic,
	"Options" -> {"BitDepth", "ColorSpace"},
	"BinaryFormat" -> True
]


End[]
