ImportExport`RegisterExport[
	"Real32",
	Export[#1,#2,{"Binary","Real32"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]