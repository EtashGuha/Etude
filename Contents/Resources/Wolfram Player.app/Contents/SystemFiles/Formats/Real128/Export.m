ImportExport`RegisterExport[
	"Real128",
	Export[#1,#2,{"Binary","Real128"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]