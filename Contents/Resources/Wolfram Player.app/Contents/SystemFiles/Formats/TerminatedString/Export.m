ImportExport`RegisterExport[
	"TerminatedString",
	Export[#1,#2,{"Binary","TerminatedString"},##3]&,
	"Sources" -> ImportExport`DefaultSources["Binary"],
	"FunctionChannels" -> {"Streams"},
	"BinaryFormat" -> True
]