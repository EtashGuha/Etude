(* ::Package:: *)

Begin["System`Convert`TableDump`"]


ImportExport`RegisterImport[
	"Table",
	ImportTable,
	{
		"Data" :> GetData,
		"Grid" :> GetGrid
	},
	"FunctionChannels" -> {"FileNames"},
	"AvailableElements" -> {"Data", "Grid"},
	"DefaultElement" -> "Data"
]


End[]
