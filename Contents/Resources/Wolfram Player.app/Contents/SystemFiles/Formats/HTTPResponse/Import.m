(* ::Package:: *)

Begin["HTTPLink`ImportExport`Symbols`"];

$HTTPResponseAvailableElements = {
	"HTTPRequest"
};

GetHTTPResponseElements[___] := "Elements" -> $HTTPResponseAvailableElements


ImportExport`RegisterImport[
	"HTTPResponse",
	{
		"HTTPResponse"->importHTTPResponseStream,
		"Elements"->GetHTTPResponseElements
	},
	"Sources"->{"HTTPLink`ImportExport`"},
	"AvailableElements"->$HTTPResponseAvailableElements,
	"DefaultElement"->"HTTPResponse",
	"FunctionChannels"->{"Streams"},
	"BinaryFormat"->True
];

End[];