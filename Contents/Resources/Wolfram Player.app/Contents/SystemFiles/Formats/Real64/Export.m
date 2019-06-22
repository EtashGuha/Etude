ImportExport`RegisterExport[
	"Real64",
	Export[#1,#2,{"Binary","Real64"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]