ImportExport`RegisterExport[
	"Complex256",
	Export[#1,#2,{"Binary","Complex256"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]