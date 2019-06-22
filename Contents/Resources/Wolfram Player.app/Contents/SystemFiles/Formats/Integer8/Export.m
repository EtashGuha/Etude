ImportExport`RegisterExport[
	"Integer8",
	Export[#1,#2,{"Binary","Integer8"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]