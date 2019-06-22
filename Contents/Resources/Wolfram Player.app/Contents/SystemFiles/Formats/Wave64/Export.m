(* ::Package:: *)

ImportExport`RegisterExport[
 "Wave64",
 System`Convert`AudioDump`ExportAudio["Wave64", ##]&,
 "Options" -> {"AudioChannels", "AudioEncoding", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]
