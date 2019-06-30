ImportExport`RegisterExport[
	"Integer16",
	Export[#1,#2,{"Binary","Integer16"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]