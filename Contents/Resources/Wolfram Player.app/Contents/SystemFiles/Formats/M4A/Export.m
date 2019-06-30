(* ::Package:: *)

ImportExport`RegisterExport[
 "M4A",
 System`Convert`AudioDump`ExportAudio["M4A", ##]&,
 "Options" -> {"AudioChannels", "BitRate", "SampleRate", "MetaInformation"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]
