(* ::Package:: *)

Begin["System`Convert`ICCDump`"]

ImportExport`RegisterImport[
	"ICC",
	ImportICCRawData,
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> "ColorProfileData",
	"AvailableElements" -> {"ColorProfileData"},
	"BinaryFormat" -> True
]

End[]
