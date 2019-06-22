ImportExport`RegisterExport[
	"Integer128",
	Export[#1,#2,{"Binary","Integer128"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]