(* ::Package:: *)

Begin["System`Convert`AudioDump`"]

ImportExport`RegisterImport[
"AIFF",
 {"Audio" 	  			  :> ImportAudio["AIFF"],
  "AudioChannels" 		  :> ImportAudioMetaData["AIFF", "AudioChannels"],
  "AudioEncoding" 		  :> ImportAudioMetaData["AIFF", "AudioEncoding"],
  "AudioFile" 			  :> ImportAudioFile["AIFF"],
  "Data" 	  			  :> ImportAudioData["AIFF"],
  "Duration" 		  	  :> ImportAudioMetaData["AIFF", "Duration"],
  "Length" 		  	  	  :> ImportAudioMetaData["AIFF", "Length"],
  "MetaInformation"   	  :> ImportAudioMetaData["AIFF", "MetaInformation"],
  "RawMetaInformation" 	  :> ImportAudioMetaData["AIFF", "RawMetaInformation"],
  "SampleDepth" 		  :> ImportAudioMetaData["AIFF", "SampleDepth"],
  "SampledSoundList" 	  :> ImportSampledSoundList["AIFF"],
  "SampleRate" 		  	  :> ImportAudioMetaData["AIFF", "SampleRate"],
  "Sound" 	  			  :> ImportSound["AIFF"],
  
  "Elements"  			  :> GetListOfAudioElements["AIFF"],
  GetDefaultAudioElement["AIFF"]}
 ,
 "Options" -> {"AudioChannels", "SampleRate"},
 "Sources" -> ImportExport`DefaultSources["Audio"],
 "AvailableElements" -> {"Audio", "AudioChannels", "AudioEncoding", "AudioFile", "Data", "Duration", "Length", "MetaInformation", "RawMetaInformation", "SampleDepth", "SampledSoundList", "SampleRate", "Sound"},
 "BinaryFormat" -> True
]


End[]
