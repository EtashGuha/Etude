ImportExport`RegisterExport[
	"Integer64",
	Export[#1,#2,{"Binary","Integer64"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]