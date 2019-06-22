(* ::Package:: *)

Begin["System`Convert`ICSDump`"]

ImportExport`RegisterImport["ICS",
	ImportICS,
	"Sources" -> {"Convert`ICS`"},
	"FunctionChannels" -> {"Streams"},
	"DefaultElement" -> "Events",
	"AvailableElements" -> {"CalendarSystem", "Creator", "Events", "Availability", "JournalEntries", "TimeZones", "Tasks"},
	"BinaryFormat" -> True
]

End[]
