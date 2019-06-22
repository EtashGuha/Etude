(* ::Package:: *)

Begin["System`Convert`WDXDump`"]


ImportExport`RegisterImport[
  "WDX",
  {
	{"DataTable", key_, prop__ } :> ImportWDXObject[key, prop],
	ImportWDX
  },
  "AvailableElements" -> {"Attributes", "DataGroups", "DataIndex", "DataTable", "Expression", "Subtype", "Unknown", "Version"},
  "DefaultElement" -> "Expression",
  "FunctionChannels" -> {"FileNames", "Streams"},
  "BinaryFormat" -> True
]


End[]
