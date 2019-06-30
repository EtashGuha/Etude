ImportExport`RegisterImport[
	"Complex64",
	Import[#1,{"Binary","Complex64"},##2]&,
	"AvailableElements" -> {_Integer},
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]