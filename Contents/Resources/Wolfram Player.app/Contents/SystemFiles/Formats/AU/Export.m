(* ::Package:: *)

ImportExport`RegisterExport[
 "AU",
 System`Convert`AudioDump`ExportAudio["AU", ##]&,
 "Options" -> {"AudioChannels", "AudioEncoding", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]
