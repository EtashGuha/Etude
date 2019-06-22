(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

$ValidRawJSONStreamOptions = Keys[
	FilterRules[
		Options[Developer`WriteRawJSONStream], 
		Except["IssueMessagesAs"]
	]
];

$ValidRawJSONFileOptions = Keys[
	FilterRules[
		Options[Developer`WriteRawJSONFile], 
		Except["IssueMessagesAs"]
	]
];

writeJSON[filename_String, expr_, opts___] := Developer`WriteRawJSONFile[
	filename, 
	expr, 
	"IssueMessagesAs" -> Export, 
	Sequence @@ FilterRules[{opts}, $ValidRawJSONFileOptions]
];

writeJSON[stream_OutputStream, expr_, opts___] := Developer`WriteRawJSONStream[
	stream, 
	expr, 
	"IssueMessagesAs" -> Export, 
	Sequence @@ FilterRules[{opts}, $ValidRawJSONStreamOptions]
];

ImportExport`RegisterExport["RawJSON",
	writeJSON,
	"FunctionChannels" -> {"FileNames", "Streams"}
]

End[]
