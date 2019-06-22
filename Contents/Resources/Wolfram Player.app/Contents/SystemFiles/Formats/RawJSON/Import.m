(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

parseJSON[filename_String, opts___] := "Data"->Developer`ReadRawJSONFile[filename, "IssueMessagesAs" -> Import];
parseJSON[stream_InputStream, opts___] := "Data"->Developer`ReadRawJSONStream[stream, "IssueMessagesAs" -> Import];

(* For consistency with JSON we prefer Data over Expression *)
$AvailableElements = {"Data"};

GetElements[___] := "Elements" -> $AvailableElements;

ImportExport`RegisterImport["RawJSON",
	{
		"Data" :> parseJSON,
		"Elements" :> GetElements
	},
	"FunctionChannels" -> {"FileNames", "Streams"},
	"DefaultElement" -> "Data",
	"AvailableElements" -> $AvailableElements
]

End[]
