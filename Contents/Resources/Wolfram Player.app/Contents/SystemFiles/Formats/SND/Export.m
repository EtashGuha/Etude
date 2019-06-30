(* ::Package:: *)

Begin["System`Convert`AudioDump`"]


ImportExport`RegisterExport[
 "SND",
 System`Convert`AudioDump`ExportAudio["SND", ##]&,
 "Options" -> {"AudioChannels", "AudioEncoding", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]


End[]
