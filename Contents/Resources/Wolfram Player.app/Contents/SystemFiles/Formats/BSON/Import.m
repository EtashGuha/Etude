(* ::Package:: *)

Begin["System`Convert`BSONDump`"]

$AvailableElements = {"Data"};

GetElements[___] := "Elements"->$AvailableElements;

ImportExport`RegisterImport["BSON",
	{
		"Data" :> importBSON,
		"Elements" :> GetElements
	},
	"DefaultElement" -> "Data",
	"AvailableElements" -> $AvailableElements,
	"FunctionChannels" -> {"FileNames"},
	"BinaryFormat" -> True
]

End[]
