Begin["GeneralUtilities`ImportExport`"]

wmlfImport[args___] := "Data" -> GeneralUtilities`MLImport[args];
getWMLFElements[___] := "Elements" -> Lookup[System`ConvertersDump`FileFormatDataFull["WMLF"], "ImportElements"];

ImportExport`RegisterImport["WMLF",
	{
		"Data" :> wmlfImport,
		"Elements" :> getWMLFElements
	},
	"DefaultElement" -> "Data",
	"AvailableElements" -> {"Data"},
	"BinaryFormat" -> True
]

End[]