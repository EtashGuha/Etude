ImportExport`RegisterExport[
	"Byte",
	Export[#1,#2,{"Binary","Byte"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]