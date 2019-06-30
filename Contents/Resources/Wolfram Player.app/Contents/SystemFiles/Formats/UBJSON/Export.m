(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

writeUBJSON[filename_String, expr_, opts___] :=
	Developer`WriteUBJSONFile[filename, expr, "IssueMessagesAs" -> Export, opts];

writeUBJSON[stream_OutputStream, expr_, opts___] :=
	Developer`WriteUBJSONStream[stream, expr, "IssueMessagesAs" -> Export, opts];

ImportExport`RegisterExport["UBJSON",
	writeUBJSON,
	"FunctionChannels" -> {"FileNames", "Streams"},
	"BinaryFormat" -> True
]

End[]
