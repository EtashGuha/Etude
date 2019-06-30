(* ::Package:: *)

Begin["System`Convert`PNMDump`"]


ImportExport`RegisterExport[
    "PGM",
	ExportPNM["PGM", ##]&,
	"Sources" -> ImportExport`DefaultSources["PNM"],
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> Automatic,
	"Options" -> {"DataType", "BitDepth", "ColorSpace"},
	"BinaryFormat" -> True
]


End[]
