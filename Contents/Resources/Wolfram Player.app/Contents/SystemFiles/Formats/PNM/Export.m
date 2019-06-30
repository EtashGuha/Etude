(* ::Package:: *)

Begin["System`Convert`PNMDump`"]


ImportExport`RegisterExport[
	"PNM",
	ExportPNM["PNM", ##]&,
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> Automatic,
	"Options" -> {"BitDepth", "ColorSpace"},
	"BinaryFormat" -> True
]


End[]
