(* ::Package:: *)

Begin["System`Convert`VCFDump`"]


ImportExport`RegisterImport[
	"VCF",
	{
		"Records"	-> ImportVCF,
		"Data"		-> ImportData,
		"Labels"	-> ImportLabels,
		ImportVCF
	},
	{
		{"Elements"}-> GetVCFElements,
		{"Data"}	-> SkipPostProcs["Data"],
		{"Records"}	-> SkipPostProcs["Records"],
		{"Labels"}	-> SkipPostProcs["Labels"],
		str_String	:> GetAnElement[str]
	},
	"FunctionChannels" -> {"Streams"},
	"AvailableElements" -> {"Data", "Labels", "Records", _String},
	"DefaultElement" -> "Records"
]


End[]

