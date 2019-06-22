(* ::Package:: *)

Begin["HTTPLink`ImportExport`Symbols`"];

ImportExport`RegisterExport[
	"HTTPRequest",
	exportHTTPRequestStream,
	"Sources"->{"HTTPLink`ImportExport`"},
	"FunctionChannels"->{"Streams"},
	"BinaryFormat"->True
];

End[];
