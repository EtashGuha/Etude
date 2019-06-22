(* ::Package:: *)

Begin["System`Convert`ACODump`"]


ImportExport`RegisterImport[
	"ACO",
	ImportACO,
	{
		"ColorList" :> CreateColorList,
		"ColorSetters" :> CreateColorSetters
	},
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> {"ColorList", "ColorRules", "ColorSetters"},
	"DefaultElement" -> "ColorList",
	"BinaryFormat" -> True
]


End[]
