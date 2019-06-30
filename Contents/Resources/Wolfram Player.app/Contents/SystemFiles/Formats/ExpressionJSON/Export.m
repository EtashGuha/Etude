(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

writeExpressionJSON[filename_String, expr_, opts___] :=
	Developer`WriteExpressionJSONFile[filename, expr, "IssueMessagesAs" -> Export, opts];
writeExpressionJSON[stream_OutputStream, expr_, opts___] :=
	Developer`WriteExpressionJSONStream[stream, expr, "IssueMessagesAs" -> Export, opts];

ImportExport`RegisterExport["ExpressionJSON",
	writeExpressionJSON,
	"FunctionChannels" -> {"FileNames", "Streams"}
]

End[]
