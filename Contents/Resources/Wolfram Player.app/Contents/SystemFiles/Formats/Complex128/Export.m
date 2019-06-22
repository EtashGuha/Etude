ImportExport`RegisterExport[
	"Complex128",
	Export[#1,#2,{"Binary","Complex128"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]