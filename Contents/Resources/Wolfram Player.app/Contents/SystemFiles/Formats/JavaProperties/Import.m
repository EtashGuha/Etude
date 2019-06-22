Begin["System`Convert`JavaPropertiesDump`"]

ImportExport`RegisterImport["JavaProperties",
	{
		"Data" :> ImportJavaProperties,
		"DataRules" :> ImportJavaPropertiesRules,
		"AnnotatedData" :> ImportAnnotatedJavaProperties,
		ImportJavaProperties
	},
	"FunctionChannels" -> {"FileNames", "Streams"},
	"DefaultElement" -> "Data",
	"AvailableElements" -> {"Data", "DataRules", "AnnotatedData"}
];

End[]
