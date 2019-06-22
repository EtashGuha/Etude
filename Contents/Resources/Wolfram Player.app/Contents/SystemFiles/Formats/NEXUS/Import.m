(* ::Package:: *)

Begin["System`Convert`NEXUSDump`"]


ImportExport`RegisterImport[
    "NEXUS",
	{
		"DistanceMatrix"	:> ImportDistances,
		ImportCharacters
	},
	{
		"Sequence"			:> getSequenceElem,
		"Taxa"				:> getTaxaElem,
		"Data"				:> getDataElem
	},
	"FunctionChannels" -> {"Streams"},
	"Sources" -> {"Convert`NEXUS`"},
	"AvailableElements" -> {"DistanceMatrix", "LabeledData", "Sequence", "Taxa", "Data"},
	"DefaultElement" -> "Sequence"
]


End[]
