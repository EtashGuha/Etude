(* ::Package:: *)

Begin["System`Convert`JSONDump`"]

(* JSON legacy element is Data even if Expression would be better. *)
$AvailableElements = {"Data"};

GetElements[___] := "Elements" -> $AvailableElements;

ImportExport`RegisterImport[
	"JSON",
	{
		"Data" :> importJSON,
		"Elements" :> GetElements
	},
	"FunctionChannels" -> {"Streams"},
	"Sources" -> {"Convert`JSON`"},
	"DefaultElement" -> "Data", 
	"AvailableElements" -> $AvailableElements
]


End[]
