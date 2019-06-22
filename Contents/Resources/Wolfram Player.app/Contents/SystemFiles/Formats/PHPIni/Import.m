Begin["System`Convert`PHPIniDump`"]

ImportExport`RegisterImport["PHPIni",
	{
		"Data" :> ImportPHPIniData,
		"DataRules" :> ImportPHPIniDataRules,
		"CommentedData" :> ImportPHPIniCommentedData,
		ImportPHPIniData
	},
	"FunctionChannels" -> {"FileNames", "Streams"},
	"AvailableElements" -> {"Data", "DataRules", "CommentedData"},
	"DefaultElement" -> "Data"
];

End[]
