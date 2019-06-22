(* ::Package:: *)

Begin["System`Convert`DirectoryDump`"]


ImportExport`RegisterImport[
    "Directory",
	{
		{"Elements"} -> ({"FileNames"->{}}&),
		{"FileNames"} -> GetFileNames[{"FileNames"}],
		{"FileNames", "Elements"} -> GetFileNames[{"FileNames", "Elements"}],
		{"FileNames", (files:(_String | {__String})), elems___} :> ImportFile[files, {elems}],
		{file:Except[({"FileNames"}|"FileNames"|All|Automatic)], rest___}:> ImportFileDirectly[file, {rest}],
		GetFileNames[{"FileNames"}]
	}, {},
	"FunctionChannels" -> {"Directories"},
	"Sources" -> ImportExport`DefaultSources["Directory"],
	"Options" -> {},
	"AvailableElements" -> {"FileNames", _String},
	"DefaultElement" ->"FileNames",
	"BinaryFormat" -> True
]


End[]
