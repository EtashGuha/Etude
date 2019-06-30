(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

ImportExport`RegisterExport["JSON",
	exportJSON,
	"FunctionChannels" -> {"FileNames", "Streams"},
    "Sources" -> {"Convert`JSON`"}
]

End[]
