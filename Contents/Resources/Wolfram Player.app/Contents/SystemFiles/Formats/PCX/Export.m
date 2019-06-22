(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
    "PCX",
	ExportRasterElements["PCX", System`Convert`PCXDump`ExportPCX, ##]&,
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> Automatic,
	"Options" -> {"DataType", "BitDepth", "ColorSpace"},
	"BinaryFormat" -> True
]


End[]
