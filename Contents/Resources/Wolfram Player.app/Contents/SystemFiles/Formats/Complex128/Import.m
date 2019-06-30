ImportExport`RegisterImport[
	"Complex128",
	Import[#1,{"Binary","Complex128"},##2]&,
	"AvailableElements" -> {_Integer},
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]