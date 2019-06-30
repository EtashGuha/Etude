(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

parseJSON[filename_String, opts___] := Developer`ReadRawJSONFile[filename, "IssueMessagesAs" -> Import];
parseJSON[stream_InputStream, opts___] := Developer`ReadRawJSONStream[stream, "IssueMessagesAs" -> Import];

ImportExport`RegisterImport["JavaScriptExpression",
	parseJSON,
	"FunctionChannels" -> {"FileNames", "Streams"}
]

End[]
