(* ::Package:: *)

ImportExport`RegisterExport[
 "AIFF",
 System`Convert`AudioDump`ExportAudio["AIFF", ##]&,
 "Options" -> {"AudioChannels", "AudioEncoding", "MetaInformation", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]
