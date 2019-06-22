(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

parseUBJSON[filename_String, opts___] := "Data" -> Developer`ReadUBJSONFile[filename, "IssueMessagesAs" -> Import];
parseUBJSON[stream_InputStream, opts___] := "Data" -> Developer`ReadUBJSONStream[stream, "IssueMessagesAs" -> Import];

$AvailableElements = {"Data"};

GetElements[___] := "Elements" -> $AvailableElements;

ImportExport`RegisterImport["UBJSON",
	{
		"Data" :> parseUBJSON,
		"Elements" :> GetElements
	},
	"FunctionChannels" -> {"FileNames", "Streams"},
	"BinaryFormat" -> True,
	"DefaultElement" -> "Data",
	"AvailableElements" -> $AvailableElements
]

End[]
