(* ::Package:: *)

Begin["System`Convert`PNMDump`"]


ImportExport`RegisterExport[
	"PPM",
	ExportPNM["PPM", ##]&,
	"Sources" -> ImportExport`DefaultSources["PNM"],
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> Automatic,
	"Options" -> {"DataType", "BitDepth", "ColorSpace"},
	"BinaryFormat" -> True
]


End[]
