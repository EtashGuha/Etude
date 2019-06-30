(* ::Package:: *)

Begin["System`Convert`GeoJSONDump`"]

ImportExport`RegisterExport["GeoJSON",
	ExportGeoJSON,
    "BinaryFormat" -> True,
    "FunctionChannels" -> {"Streams"}
]

End[]