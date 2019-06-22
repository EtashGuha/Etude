(* ::Package:: *)

Begin["System`Convert`AudioDump`"]

(****************** M4A for Windows & OSX ******************)
ImportExport`RegisterImport[
	"M4A",
 	{"Audio" 	  			  :> ImportAudio["M4A"],
  	"AudioChannels" 		  :> ImportAudioMetaData["M4A", "AudioChannels"],
  	"AudioFile" 			  :> ImportAudioFile["M4A"],
  	"Data" 		  			  :> ImportAudioData["M4A"],
  	"Duration" 			  	  :> ImportAudioMetaData["M4A", "Duration"],
  	"Length" 		  	  	  :> ImportAudioMetaData["M4A", "Length"],
  	"SampledSoundList" 	  	  :> ImportSampledSoundList["M4A"],
  	"SampleRate" 		  	  :> ImportAudioMetaData["M4A", "SampleRate"],
    "BitRate" 		  	      :> ImportAudioMetaData["M4A", "BitRate"],
    "Sound" 	  			  :> ImportSound["M4A"],
    "MetaInformation" 		  :> ImportAudioMetaData["M4A", "MetaInformation"],
    "RawMetaInformation" 	  :> ImportAudioMetaData["M4A", "RawMetaInformation"],
  	"Elements"  			  :> GetListOfAudioElements["M4A"],
  	GetListOfAudioElements["M4A"]}
 	,
 	"DefaultElement" -> "Audio",
 	"Options" -> {"AudioChannels", "SampleRate"},
 	"Sources" -> ImportExport`DefaultSources["Audio"],
 	"SystemID" -> ("Windows*" | "Mac*"),
 	"AvailableElements" -> {"Audio", "AudioChannels", "AudioFile", "BitRate", "Data", "Duration", "Length", "SampledSoundList", "SampleRate", "Sound", "MetaInformation", "RawMetaInformation"},
 	"BinaryFormat" -> True
]

End[]
