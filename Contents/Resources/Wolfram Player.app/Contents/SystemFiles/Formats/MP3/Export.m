(* ::Package:: *)

ImportExport`RegisterExport[
 "MP3",
 System`Convert`AudioDump`ExportAudio["MP3", ##]&,
 "Options" -> {"AudioChannels", "CompressionLevel", "SampleRate", "MetaInformation"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "DefaultElement" -> "Audio",
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True
]
