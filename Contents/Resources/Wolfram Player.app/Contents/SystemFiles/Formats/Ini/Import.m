Begin["System`Convert`IniDump`"]

ImportExport`RegisterImport["Ini",
	{
		"Data" :> ImportIniData,
		"DataRules" :> ImportIniDataRules,
		"AnnotatedData" :> ImportIniCommentedData,
		ImportIniData
	},
	"FunctionChannels" -> {"FileNames", "Streams"},
	"AvailableElements" -> {"Data", "DataRules", "AnnotatedData"},
	"DefaultElement" -> "Data"
];

End[]
