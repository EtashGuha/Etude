(* ::Package:: *)

Begin["TINSLink`ImportExport`Symbols`"];

$PCAPAvailableElements = {"Data"};

GetPCAPElements[___] := "Elements" -> $PCAPAvailableElements


ImportExport`RegisterImport[
	"PCAP",
	{
		"Data"->ImportPacketCapture,
		"Elements" -> GetPCAPElements
	},
	"Sources"->{"TINSLink`ImportExport`"},
	"AvailableElements"->{"Data"},
	"DefaultElement"->"Data",
	"FunctionChannels"->{"FileNames"},
	"BinaryFormat"->True
];

End[];