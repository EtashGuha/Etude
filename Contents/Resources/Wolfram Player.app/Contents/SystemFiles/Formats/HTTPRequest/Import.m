(* ::Package:: *)

Begin["HTTPLink`ImportExport`Symbols`"];

$HTTPRequestAvailableElements = {
	"HTTPRequest"
};

GetHTTPRequestElements[___] := "Elements" -> $HTTPRequestAvailableElements


ImportExport`RegisterImport[
	"HTTPRequest",
	{
		"HTTPRequest"->importHTTPRequestStream,
		"Elements"->GetHTTPRequestElements
	},
	"Sources"->{"HTTPLink`ImportExport`"},
	"AvailableElements"->$HTTPRequestAvailableElements,
	"DefaultElement"->"HTTPRequest",
	"FunctionChannels"->{"Streams"},
	"BinaryFormat"->True
];

End[];