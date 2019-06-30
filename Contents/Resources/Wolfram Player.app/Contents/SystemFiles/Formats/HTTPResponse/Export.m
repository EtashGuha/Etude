(* ::Package:: *)

Begin["HTTPLink`ImportExport`Symbols`"];

ImportExport`RegisterExport[
	"HTTPResponse",
	exportHTTPResponseStream,
	"Sources"->{"HTTPLink`ImportExport`"},
	"FunctionChannels"->{"Streams"},
	"BinaryFormat"->True
];

End[];
