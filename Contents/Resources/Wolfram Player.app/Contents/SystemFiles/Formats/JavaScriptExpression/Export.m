(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

writeJSON[filename_String, expr_, opts___] :=
	Developer`WriteRawJSONFile[filename, expr, "IssueMessagesAs" -> Export, opts];

writeJSON[stream_OutputStream, expr_, opts___] :=
	Developer`WriteRawJSONStream[stream, expr, "IssueMessagesAs" -> Export, opts];

ImportExport`RegisterExport["JavaScriptExpression",
	writeJSON,
	"FunctionChannels" -> {"FileNames", "Streams"}
]

End[]
