(* ::Package:: *)

Begin["HTTPLink`ImportExport`Symbols`"];

$WARCAvailableElements = {
	"Dataset",
	"RawData",
	"RawDataset",
	"RawStringDataset"
};

GetWARCElements[___] := "Elements" -> $WARCAvailableElements

ImportExport`RegisterImport[
	"WARC",
	{
		"Dataset"->importWARCStreamDataset,
		"RawData"->importWARCStreamRawData,
		"RawDataset"->importWARCStreamRawDataset,
		"RawStringDataset"->importWARCStreamRawStringDataset,
		"Elements"->GetWARCElements
	},
	"Sources"->{"HTTPLink`ImportExport`"},
	"AvailableElements"->$WARCAvailableElements,
	"DefaultElement"->"Dataset",
	"FunctionChannels"->{"Streams"},
	"BinaryFormat"->True
];

End[];
