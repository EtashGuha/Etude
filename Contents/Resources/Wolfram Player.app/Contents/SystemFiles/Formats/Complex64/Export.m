ImportExport`RegisterExport[
	"Complex64",
	Export[#1,#2,{"Binary","Complex64"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]