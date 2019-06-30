(* ::Package:: *)

Begin["System`Convert`PNMDump`"]


ImportExport`RegisterExport[
	"PBM",
	ExportPNM["PBM", ##]&,
	"Sources" -> ImportExport`DefaultSources["PNM"],
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> Automatic,
	"Options" -> {"DataType", "BitDepth", "ColorSpace"},
	"BinaryFormat" -> True
]


End[]
