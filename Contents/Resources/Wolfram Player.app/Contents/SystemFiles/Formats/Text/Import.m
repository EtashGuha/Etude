(* ::Package:: *)

Begin["System`Convert`TextDump`"]


ImportExport`RegisterImport[
    "Text",
	{
		"String" :> StringImport,
		"Plaintext" :> PlaintextImport,
		"Data" :> DataImport,
		"Lines" :> LinesImport,
		{"Lines", n_Integer} :> SingleLinesTextImport[{n}],
	    {"Lines", L_List} :> SingleLinesTextImport[L],
		"Words" :> WordsImport,
		PlaintextImport
	},
	{},
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> {"Data", "Lines", "Plaintext", "String", "Words"},
	"DefaultElement" -> "Plaintext",
	"BinaryFormat" -> True
]


End[]
