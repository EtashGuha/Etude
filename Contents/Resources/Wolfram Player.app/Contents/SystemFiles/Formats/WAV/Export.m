(* ::Package:: *)

ImportExport`RegisterExport[
 "WAV",
 System`Convert`AudioDump`ExportAudio["WAV", ##]&,
 "Options" -> {"AudioChannels", "AudioEncoding", "MetaInformation", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]
