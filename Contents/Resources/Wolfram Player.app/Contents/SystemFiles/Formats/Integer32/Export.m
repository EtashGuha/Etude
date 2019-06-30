ImportExport`RegisterExport[
	"Integer32",
	Export[#1,#2,{"Binary","Integer32"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]