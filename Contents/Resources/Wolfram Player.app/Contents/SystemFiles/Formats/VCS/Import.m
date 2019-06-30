(* ::Package:: *)

Begin["System`Convert`VCSDump`"]

ImportExport`RegisterImport["VCS",
	ImportVCS,
	"Sources" -> {"Convert`VCS`"},
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> "Events",
	"AvailableElements" -> {"Creator", "Events", "GeoPosition", "Tasks", "TimeZone"},
	"BinaryFormat" -> True
]

End[]
