(* ::Package:: *)

Begin["System`Convert`AudioDump`"]

ImportExport`RegisterImport[
"AU",
 {"Audio" 	  			  :> ImportAudio["AU"],
  "AudioChannels" 		  :> ImportAudioMetaData["AU", "AudioChannels"],
  "AudioEncoding" 		  :> ImportAudioMetaData["AU", "AudioEncoding"],
  "Data" 	  			  :> ImportAudioData["AU"],
  "Duration" 		  	  :> ImportAudioMetaData["AU", "Duration"],
  "Length" 		  	  	  :> ImportAudioMetaData["AU", "Length"],
  "SampleDepth" 		  :> ImportAudioMetaData["AU", "SampleDepth"],
  "SampledSoundList" 	  :> ImportSampledSoundList["AU"],
  "SampleRate" 		  	  :> ImportAudioMetaData["AU", "SampleRate"],
  "Sound" 	  			  :> ImportSound["AU"],
  
  "Elements"  			  :> GetListOfAudioElements["AU"],
  GetListOfAudioElements["AU"]}
 ,
 "DefaultElement" -> "Audio",
 "Options" -> {"AudioChannels", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "AvailableElements" -> {"Audio", "AudioChannels", "AudioEncoding", "Data", "Duration", "Length", "SampleDepth", "SampledSoundList", "SampleRate", "Sound"},
 "BinaryFormat" -> True
]


End[]
