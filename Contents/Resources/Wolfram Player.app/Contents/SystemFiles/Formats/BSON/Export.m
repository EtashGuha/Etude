(* ::Package:: *)

Begin["System`Convert`BSONDump`"]

ImportExport`RegisterExport["BSON",
	exportBSON,
	"FunctionChannels" -> {"FileNames"},
	"BinaryFormat" -> True
]

End[]
