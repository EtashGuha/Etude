(* ::Package:: *)

ImportExport`RegisterExport[
 "FLAC",
 System`Convert`AudioDump`ExportAudio["FLAC", ##]&,
 "Options" -> {"AudioChannels", "AudioEncoding", "MetaInformation", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]
