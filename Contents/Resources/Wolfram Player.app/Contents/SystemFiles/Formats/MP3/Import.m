(* ::Package:: *)

Begin["System`Convert`AudioDump`"]

ImportExport`RegisterImport[
 "MP3",
 {"Audio" 	  			  :> ImportAudio["MP3"],
  "AudioChannels" 		  :> ImportAudioMetaData["MP3", "AudioChannels"],
  "AudioFile" 			  :> ImportAudioFile["MP3"],
  "Data" 	  			  :> ImportAudioData["MP3"],
  "Duration" 		  	  :> ImportAudioMetaData["MP3", "Duration"],
  "Length" 		  	  	  :> ImportAudioMetaData["MP3", "Length"],
  "SampledSoundList" 	  :> ImportSampledSoundList["MP3"],
  "SampleRate" 		  	  :> ImportAudioMetaData["MP3", "SampleRate"],
  "Sound" 	  			  :> ImportSound["MP3"],
  "MetaInformation" 	  :> ImportAudioMetaData["MP3", "MetaInformation"],
  "RawMetaInformation" 	  :> ImportAudioMetaData["MP3", "RawMetaInformation"],

  "Elements"  			  :> GetListOfAudioElements["MP3"],
  GetDefaultAudioElement["MP3"]}
 ,
 "Options" -> {"AudioChannels", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "AvailableElements" -> {"Audio", "AudioChannels", "AudioFile", "Data", "Duration", "Length", "SampledSoundList", "SampleRate", "Sound", "MetaInformation", "RawMetaInformation"},
 "BinaryFormat" -> True
]


End[]
