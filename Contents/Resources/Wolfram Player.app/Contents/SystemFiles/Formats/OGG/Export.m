(* ::Package:: *)

ImportExport`RegisterExport[
 "OGG",
 System`Convert`AudioDump`ExportAudio["OGG", ##]&,
 "Options" -> {"AudioChannels", "AudioEncoding", "CompressionLevel", "MetaInformation", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]
