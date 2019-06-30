(* ::Package:: *)

Begin["System`Convert`NASACDFDump`"]


ImportExport`RegisterExport["NASACDF",
	ExportCDF,
	"DefaultElement" -> "Datasets",
	"BinaryFormat" -> True,
	"Sources" -> ImportExport`DefaultSources[{"NASACDF", "CDF.exe", "DataCommon"}]
]

End[]
