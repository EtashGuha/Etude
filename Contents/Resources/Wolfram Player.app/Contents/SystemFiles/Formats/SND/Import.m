(* ::Package:: *)

Begin["System`Convert`AudioDump`"]

ImportExport`RegisterImport[
"SND",
 {"Data" 	  			  :> ImportAudioData["SND"],
  "Audio" 	  			  :> ImportAudio["SND"],
  "Sound" 	  			  :> ImportSound["SND"],
  "SampledSoundList" 	  :> ImportSampledSoundList["SND"],
  "Audio" 	  			  :> ImportAudio["SND"],
  "AudioChannels" 		  :> ImportAudioMetaData["SND", "AudioChannels"],
  "SampleRate" 		  	  :> ImportAudioMetaData["SND", "SampleRate"],
  "AudioEncoding" 		  :> ImportAudioMetaData["SND", "AudioEncoding"],
  "Elements"  			  :> GetListOfAudioElements["SND"],
  GetListOfAudioElements["SND"]}
 ,
 "DefaultElement" -> "Audio",
 "Options" -> {"AudioChannels", "AudioEncoding", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "AvailableElements" -> {"Audio", "AudioChannels", "AudioEncoding", "Data", "SampledSoundList", "SampleRate", "Sound"},
 "BinaryFormat" -> True
]

End[]
