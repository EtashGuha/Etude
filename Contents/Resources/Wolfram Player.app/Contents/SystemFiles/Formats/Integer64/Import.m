ImportExport`RegisterImport[
	"Integer64",
	Import[#1,{"Binary","Integer64"},##2]&,
	"AvailableElements" -> {_Integer},
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]