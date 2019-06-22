(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

readExpressionJSON[filename_String, opts___] := "Expression" -> Developer`ReadExpressionJSONFile[filename, "IssueMessagesAs" -> Import];
readExpressionJSON[stream_InputStream, opts___] := "Expression" -> Developer`ReadExpressionJSONStream[stream, "IssueMessagesAs" -> Import];

$AvailableElements = {"Expression"};

GetElements[___] := "Elements" -> $AvailableElements;

ImportExport`RegisterImport["ExpressionJSON",
	{
		"Expression" :> readExpressionJSON,
		"Elements" :> GetElements
	},
	"FunctionChannels" -> {"FileNames", "Streams"},
	"DefaultElement" -> "Expression",
	"AvailableElements" -> $AvailableElements
]

End[]
