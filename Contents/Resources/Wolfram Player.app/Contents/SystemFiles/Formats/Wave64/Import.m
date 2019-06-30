(* ::Package:: *)

Begin["System`Convert`AudioDump`"]

ImportExport`RegisterImport[
"Wave64",
 {"Audio" 	  			  :> ImportAudio["Wave64"],
  "AudioChannels" 		  :> ImportAudioMetaData["Wave64", "AudioChannels"],
  "AudioEncoding" 		  :> ImportAudioMetaData["Wave64", "AudioEncoding"],
  "Data" 	  			  :> ImportAudioData["Wave64"],
  "Duration" 		  	  :> ImportAudioMetaData["Wave64", "Duration"],
  "Length" 		  	  	  :> ImportAudioMetaData["Wave64", "Length"],
  "SampleDepth" 		  :> ImportAudioMetaData["Wave64", "SampleDepth"],
  "SampledSoundList" 	  :> ImportSampledSoundList["Wave64"],
  "SampleRate" 		  	  :> ImportAudioMetaData["Wave64", "SampleRate"],
  "Sound" 	  			  :> ImportSound["Wave64"],
  
  "Elements"  			  :> GetListOfAudioElements["Wave64"],
  GetListOfAudioElements["Wave64"]}
 ,
 "DefaultElement" -> "Audio",
 "Options" -> {"AudioChannels", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "AvailableElements" -> {"Audio", "AudioChannels", "AudioEncoding", "Data", "Duration", "Length", "SampleDepth", "SampledSoundList", "SampleRate", "Sound"},
 "BinaryFormat" -> True
]


End[]
