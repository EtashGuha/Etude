(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
    "TGA",
	ExportRasterElements["TGA", System`Convert`TGADump`ExportTGA, ##]&,
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> Automatic,
	"Options" -> {"DataType", "BitDepth", "ColorSpace"},
	"BinaryFormat" -> True,
	"AlphaChannel"->True
]


End[]
