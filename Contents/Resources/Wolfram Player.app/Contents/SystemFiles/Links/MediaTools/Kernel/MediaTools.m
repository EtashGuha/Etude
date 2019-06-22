(* ::Package:: *)

BeginPackage["MediaTools`"]
Begin["`Private`"]

Needs["GeneralUtilities`"];

(* ================= MEDIAFOUNDATION & COREAUDIO AUDIO API DOCUMENTATION ================= *)

SetUsage[ReadAudioProperties,
	"ReadAudioProperties['file$'] returns an association of the audio properties " <>
	"for the default audio stream in 'file$'." <> "\n" <>
	"ReadAudioProperties['file$', streamId$] returns an association of " <>
	"the audio properties for streamId$ in 'file$'." <> "\n\n" <>
	"Audio property keys include:" <> "\n" <>
	"| 'Channels' | 'SampleRate' | 'SampleCount' | 'BitRate' | 'Duration' |" <> "\n\n" <>
	"Example output:" <> "\n" <> "<|" <> "\n" <>
	"'Channels' -> 1, 'SampleRate' -> 44100," <>
	"'SampleCount' -> 88200, 'BitRate' -> 128000, 'Duration' -> 2.0" <> "\n" <> "|>"
];

SetUsage[$ReadAudioProperties, 
	"$ReadAudioProperties['file$'] returns a nested association " <>
	"of the audio properties for each stream index in file$." <> "\n\n" <>
	"Audio property keys include:" <> "\n" <>
	"| 'Channels' | 'SampleRate' | 'SampleCount' | 'BitRate' | 'Duration' |" <> "\n\n" <>
	"Example output:" <> "\n" <> "<|" <> "\n" <> "1 -> " <> "<|" <>
	"'Channels' -> 1, 'SampleRate' -> 44100, 'SampleCount' -> 88200, " <>
	"'BitRate' -> 128000, 'Duration' -> 2.0 |>," <> "\n" <> "2 -> " <> "<|" <>
	"'Channels' -> 2, 'SampleRate' -> 48000, 'SampleCount' -> 432000, " <>
	"'BitRate' -> 196000, 'Duration' -> 4.5 |>" <> "\n" <> "|>"
];

SetUsage[$ReadAudioData,
	"$ReadAudioData['file$', streamId$] returns a 'NumericArray' of type " <>
	"SignedInteger16 containing raw audio data for stream index streamId$ in 'file$'."
];

SetUsage[ReadAudioData,
	"ReadAudioData['file$'] returns a 'NumericArray' of type SignedInteger16 " <>
	"containing raw audio data for the default stream index in 'file$'." <> "\n" <>
	"ReadAudioData['file$', streamId$] returns a 'NumericArray' of type " <>
	"SignedInteger16 containing raw audio data for the streamId$ in 'file$'."
];

SetUsage[$WriteAudio,
	"$WriteAudio[data$, sr$, br$, 'file$'] encodes audio and writes it to file " <>
	"using the following properties:" <> "\n" <>
	"data$ is a rank 1 or 2 'NumericArray' of type SignedInteger16 " <>
	"containing the raw audio data." <> "\n" <> 
	"sr$ is an integer specifying the sample rate of the output audio file. " <>
	"Supported values are 44100 and 48000." <> "\n" <>
	"br$ is a positive integer specifying the bit rate of the output audio file." <> "\n" <>
	"file$ is a string of the output audio file name." <> "\n" <>
	"Supported audio file extensions include:" <> "\n" <>
	"| '.mp3' | '.m4a' | '.mov' | '.aac' | '.mp4' | '.3gp' | '.3gpp' | '.3g2' | '.amr' |"
];

SetUsage[WriteAudio,
	"WriteAudio[data$, sr$, br$, 'file$'] encodes audio and writes it to file " <>
	"using the following properties:" <> "\n" <>
	"data$ is a rank 1 or 2 'NumericArray' of type SignedInteger16 " <>
	"containing the raw audio data." <> "\n" <> 
	"sr$ is an integer specifying the sample rate of the output audio file." <> "\n" <>
	"Supported values are 44100 and 48000." <> "\n" <>
	"br$ is a positive integer specifying the bit rate of the output audio file." <> "\n" <>
	"file$ is a string of the output audio file name." <> "\n" <>
	"Supported audio file extensions include:" <> "\n" <>
	"| '.mp3' | '.m4a' | '.mov' | '.aac' | '.mp4' | '.3gp' | '.3gpp' | '.3g2' | '.amr' |"
];

SetUsage[$DeleteAudioCache,
	"$DeleteAudioCache['file$'] deletes 'file$' from the audio cache." <> "\n" <>
	"If 'file$' is an empty string, then the whole audio cache is deleted."
];

(* ================= MEDIAFOUNDATION VIDEO API DOCUMENTATION ================= *)

SetUsage[$MFInitVideoReader,
	"$MFInitVideoReader['file$'] initializes a MediaFoundation based " <>
	"video reader for file$." <> "\n" <>
	"Returns True$ if the reader is successfully initialized, False$ otherwise."
];

SetUsage[$MFReadNextVideoFrame,
	"$MFReadNextVideoFrame[] reads the next video frame from an initialized " <>
	"MediaFoundation video reader and returns a 2D Image."
];

SetUsage[$MFReadVideoProperties,
	"$MFReadVideoProperties['file$'] returns a list of the most common video properties " <>
	"of 'file$' using MediaFoundation." <> "\n" <>
	"Video property keys include:" <> "\n" <>
	"| 'Duration' | 'FrameRate' | 'FrameWidth' | 'FrameHeight' | 'BitRate' | 'VideoEncoding' |"
];

SetUsage[$MFReadVideoFrameCount,
	"$MFReadVideoFrameCount['file$'] returns the total number of video frames " <>
	"in 'file$' using MediaFoundation."
];

SetUsage[$MFReadVideoFrameDurations,
	"$MFReadVideoFrameDurations['file$'] returns a list of doubles representing " <>
	"frame durations (in seconds) for all video frames in 'file$' using MediaFoundation."
];

SetUsage[$MFReadVideoData,
	"$MFReadVideoData['file$', {}] returns all the video frames from 'file$' " <>
	"as a 3D Image using MediaFoundation." <> "\n" <>
	"$MFReadVideoData['file$', {frame$1, frame$2, $$, frame$n}] returns the video frames " <>
	"at the specified indices of 'file$' as a 3D Image using MediaFoundation."
];

SetUsage[$MFFinalizeVideoReader,
	"$MFFinalizeVideoReader[] finalizes the active MediaFoundation based video reader." <> "\n" <>
	"Returns True$ if the reader is successfully finalized, 'False$' otherwise."
];

SetUsage[$MFInitVideoWriter,
	"$MFInitVideoWriter['file$', w$, h$, br$, fr$, codec$] opens file for writing with a " <>
	"MediaFoundation based video writer using the following properties:" <> "\n" <>
	"file$ is a string of the output video file name." <> "\n" <>
	"w$ is a positive integer specifying the widths of the video frames."  <> "\n" <>
	"h$ is a positive integer specifying the heights of the video frames." <> "\n" <>
	"br$ is a positive integer specifying the bit rate of the resulting video." <> "\n" <>
	"fr$ is a positive integer specifying the frame rate of the resulting video."  <> "\n" <>
	"codec$ is a string specifying the video encoding of the video stream of 'file$'." <> "\n" <>
	"Returns True$ if the reader is successfully initialized, 'False$' otherwise." <> "\n" <>
	"Use $MFSupportedVideoEncodings to get the available codecs." <> "\n" <>
	"Supported video file extensions include:" <> "\n" <>
	"| '.mov' | '.mp4' | '.3gp' | '.3gpp' | '.avi' |"
];

SetUsage[$MFWriteNextVideoFrame,
	"$MFWriteNextVideoFrame[image$] writes the input 2D image$ to a new frame in the " <>
	"initialized MediaFoundation writer." <> "\n" <>
	"Use $MFInitVideoWriter to initialize a MediaFoundation writer."
];

SetUsage[$MFWriteVideo,
	"$MFWriteVideo['file$', image$, br$, fr$, indFrDur$, codec$] uses MediaFoundation to encode " <>
	"raw video data using the following properties:" <> "\n" <>
    "file$ is a string of the output video file name." <> "\n" <>
    "image$ is a 3D Image consisting of the video frames." <> "\n" <>
	"br$ is a positive integer specifying the bit rate of the resulting video." <> "\n" <>
	"fr$ is a positive integer specifying the average frame rate of the resulting video "  <>
	"(indFrDur$ should be set to empty list {} in this case)." <> "\n" <>
	"indFrDur$ is a list containing individual frame duration of each video frame of the output video. " <>
	"Dimension of indFrDur$ should match the number of slices in the input image " <>
	"(fr$ should be set to 0 in this case)." <> "\n" <>
    "codec$ is a string specifying the video encoding of the video stream of 'file$'." <> "\n" <>
	"Use $MFSupportedVideoEncodings to get the available codecs." <> "\n" <>
	"Supported video file extensions include:" <> "\n" <>
	"| '.mov' | '.mp4' | '.3gp' | '.3gpp' | '.avi' |"
];

SetUsage[$MFFinalizeVideoWriter,
	"$MFFinalizeVideoWriter[] finalizes the active MediaFoundation based video writer." <> "\n" <>
	"Returns True$ if the writer is successfully finalized, False$ otherwise."
];

SetUsage[$MFSupportedVideoEncodings,
	"$MFSupportedVideoEncodings[type$] returns a list of all available video codecs " <>
	"for the corresponding file type." <> "\n" <>
	"Supported file types are 'QuickTime' and 'AVI'."
];


(* ================= DIRECTSHOW VIDEO API DOCUMENTATION ================= *)

SetUsage[$DSInitVideoReader,
	"$DSInitVideoReader['file$'] initializes a DirectShow based " <>
	"video reader for file$." <> "\n" <>
	"Returns True$ if the reader is successfully initialized, False$ otherwise."
];

SetUsage[$DSReadNextVideoFrame,
	"$DSReadNextVideoFrame[] reads the next video frame from an initialized " <>
	"DirectShow video reader and returns a 2D Image."
];

SetUsage[$DSReadVideoProperties,
	"$DSReadVideoProperties['file$'] returns a list of the most common video properties " <>
	"of 'file$' using DirectShow." <> "\n" <>
	"Video property keys include:" <> "\n" <>
	"| 'Duration' | 'FrameWidth' | 'FrameHeight' | 'BitRate' | 'VideoEncoding' |"
];

SetUsage[$DSReadVideoFrameCount,
	"$DSReadVideoFrameCount['file$'] returns the total number of video frames " <>
	"in 'file$' using DirectShow."
];

SetUsage[$DSReadVideoFrameDurations,
	"$DSReadVideoFrameDurations['file$'] returns a list of doubles representing " <>
	"frame durations (in seconds) for all video frames in 'file$' using DirectShow."
];

SetUsage[$DSReadVideoFrameRate, "
	$DSReadVideoFrameRate['file$'] returns the average frame rate " <>
	"of file$ using DirectShow."
];

SetUsage[$DSReadVideoData,
	"$DSReadVideoData['file$', {}] returns all the video frames from 'file$' " <>
	"as a 3D Image using DirectShow." <> "\n" <>
	"$DSReadVideoData['file$', {frame$1, frame$2, $$, frame$n}] returns the video frames " <>
	"at the specified indices of 'file$' as a 3D Image using DirectShow."
];

SetUsage[$DSFinalizeVideoReader,
	"$DSFinalizeVideoReader[] finalizes the active DirectShow based video reader." <> "\n" <>
	"Returns True$ if the reader is successfully finalized, False$ otherwise."
];


(* ================= AVFOUNDATION VIDEO API DOCUMENTATION ================= *)

SetUsage[$AVFInitVideoReader,
	"$AVFInitVideoReader['file$'] initializes an AVFoundation based " <>
	"video reader for 'file$'." <> "\n" <>
	"Returns True$ if the reader is successfully initialized, False$ otherwise."
];

SetUsage[$AVFReadNextVideoFrame,
	"$AVFReadNextVideoFrame[] reads the next video frame from an initialized " <>
	"AVFoundation video reader and returns a 2D Image."
];

SetUsage[$AVFReadVideoProperties,
	"$AVFReadVideoProperties['file$'] returns a list of the most common video properties " <>
	"of 'file$' using AVFoundation." <> "\n" <>
	"Video property keys include:" <> "\n" <>
	"| 'Duration' | 'FrameRate' | 'FrameWidth' | 'FrameHeight' | 'BitRate' | 'VideoEncoding' |"
];

SetUsage[$AVFReadVideoFrameCount,
	"$AVFReadVideoFrameCount['file$'] returns the total number of video frames " <>
	"in 'file$' using AVFoundation."
];

SetUsage[$AVFReadVideoFrameDurations,
	"$AVFReadVideoFrameDurations['file$'] returns a list of doubles representing " <>
	"frame durations (in seconds) for all video frames in 'file$' using AVFoundation."
];

SetUsage[$AVFReadVideoData,
	"$AVFReadVideoData['file$', {}] returns all the video frames from 'file$' " <>
	"as a 3D Image using AVFoundation." <> "\n" <>
	"$AVFReadVideoData['file$', {frame$1, frame$2, $$, frame$n}] returns the video frames " <>
	"at the specified indices of 'file$' as a 3D Image using AVFoundation."
];

SetUsage[$AVFFinalizeVideoReader,
	"$AVFFinalizeVideoReader[] finalizes the active AVFoundation based video reader." <> "\n" <>
	"Returns True$ if the reader is successfully finalized, False$ otherwise."
];

SetUsage[$AVFInitVideoWriter,
	"$AVFInitVideoWriter['file$', w$, h$, br$, fr$, codec$] opens a file for writing with an " <>
	"AVFoundation based video writer using the following properties:" <> "\n" <>
	"file$ is a string of the output video file name." <> "\n" <>
	"w$ is a positive integer specifying the widths of the video frames."  <> "\n" <>
	"h$ is a positive integer specifying the heights of the video frames." <> "\n" <>
	"br$ is a positive integer specifying the bit rate of the resulting video." <> "\n" <>
	"fr$ is a positive integer specifying the frame rate of the resulting video."  <> "\n" <>
	"codec$ is a string specifying the video encoding of the video stream of 'file$'." <> "\n" <>
	"Returns True$ if the reader is successfully initialized, False$ otherwise." <> "\n" <>
	"Use $AVFSupportedVideoEncodings to get the available codecs." <> "\n" <>
	"Supported video file extensions include:" <> "\n" <>
	"| '.mov' | '.mp4' | '.3gp' | '.3gpp' | '.avi' |"
];

SetUsage[$AVFWriteNextVideoFrame,
	"$AVFWriteNextVideoFrame[image$] writes the input 2D 'image$' to a new frame in the " <>
	"initialized AVFoundation writer." <> "\n" <>
	"Use $AVFInitVideoWriter to initialize an AVFoundation writer."
];

SetUsage[$AVFWriteVideo,
	"$AVFWriteVideo['file$', image$, br$, fr$, indFrDur$, codec$]  uses AVFoundation to encode " <>
	"raw video data to file$ using the following properties:" <> "\n" <>
    "file$ is a string of the output video file name." <> "\n" <>
    "image$ is a 3D Image consisting of the video frames." <> "\n" <>
	"br$ is a positive integer specifying the bit rate of the resulting video." <> "\n" <>
	"fr$ is a positive integer specifying the average frame rate of the resulting video. "  <>
	"(indFrDur$ should be empty list {} in this case)." <> "\n" <>
	"indFrDur$ is a list containing individual frame duration of each video frame of the output video. " <>
	"Dimension of indFrDur$ should match the number of slices in the input image " <>
	"(fr$ should be set to 0 in this case)." <> "\n" <>
    "codec$ is a string specifying the video encoding of the video stream of 'file$'." <> "\n" <>
	"Use $AVFSupportedVideoEncodings to get the available codecs." <> "\n" <>
	"Supported video file extensions include:" <> "\n" <>
	"| '.mov' | '.mp4' | '.3gp' | '.3gpp' | '.avi' |"
];

SetUsage[$AVFFinalizeVideoWriter,
	"$AVFFinalizeVideoWriter[] finalizes the active AVFoundation based video writer." <> "\n" <>
	"Returns True$ if the writer is successfully finalized, False$ otherwise."
];

SetUsage[$AVFSupportedVideoEncodings,
	"$AVFSupportedVideoEncodings[type$] returns a list of all available video codecs " <>
	"for the corresponding file type." <> "\n" <>
	"Supported file types are 'QuickTime' and 'AVI'."
];

SetUsage[$AVFNeedsModernization, 
	"$AVFNeedsModernization['file$'] checks if the current file needs to be " <>
	"converted before decoding." <> "\n" <>
	"Returns True$ if the conversion is needed, False$ otherwise."
];

SetUsage[$DeleteVideoCache,
	"$DeleteVideoCache['file$'] deletes 'file$' from the video cache." <> "\n" <>
	"If 'file$' is an empty string, then the whole video cache is deleted."
];


(* ================= CROSS-PLATFORM VIDEO API DOCUMENTATION ================= *)


SetUsage[InitializeVideoReader,
	"InitializeVideoReader['file$'] initializes a video reader for file$." <> "\n" <>
	"Returns True$ if the reader is successfully initialized, False$ otherwise."
];

SetUsage[ReadNextVideoFrame,
	"ReadNextVideoFrame[] reads the next video frame from an initialized " <>
	"video reader and returns a 2D Image."
];

SetUsage[ReadVideoProperties,
	"ReadVideoProperties['file$'] returns a list of the most common video properties " <>
	"of 'file$'." <> "\n" <>
	"Video property keys include:" <> "\n" <>
	"| 'Duration' | 'FrameRate' | 'FrameWidth' | 'FrameHeight' | 'BitRate' | 'VideoEncoding' |"
];

SetUsage[ReadVideoFrameCount,
	"ReadVideoFrameCount['file$'] returns the total number of video frames " <>
	"in 'file$.'"
];

SetUsage[ReadVideoFrameDurations,
	"ReadVideoFrameDurations['file$'] returns a list of doubles representing " <>
	"frame durations (in seconds) for all video frames in file$."
];

SetUsage[ReadVideoData,
	"ReadVideoData['file$', {}] returns all the video frames of 'file$' " <>
	"as a 3D Image." <> "\n" <>
	"ReadVideoData['file$', {frame$1, frame$2, $$, frame$n}] returns the video frames " <>
	"at the specified indices of 'file$' as a 3D Image."
];

SetUsage[FinalizeVideoReader,
	"FinalizeVideoReader[] finalizes the active based video reader." <> "\n" <>
	"Returns True$ if the reader is successfully finalized, False$ otherwise."
];

SetUsage[InitializeVideoWriter,
	"InitializeVideoWriter['file$', w$, h$, br$, fr$, codec$] opens file for writing " <>
	"using the following properties:" <> "\n" <>
	"'file$' is a string of the output video file name." <> "\n" <>
	"w$ is a positive integer specifying the widths of the video frames."  <> "\n" <>
	"h$ is a positive integer specifying the heights of the video frames." <> "\n" <>
	"br$ is a positive integer specifying the bit rate of the resulting video." <> "\n" <>
	"fr$ is a positive integer specifying the frame rate of the resulting video."  <> "\n" <>
	"codec$ is a string specifying the video encoding of the video stream of 'file$'." <> "\n" <>
	"Returns True$ if the reader is successfully initialized, False$ otherwise." <> "\n" <>
	"Use SupportedVideoEncodings to get the available codecs." <> "\n" <>
	"Supported video file extensions include:" <> "\n" <>
	"| '.mov' | '.mp4' | '.3gp' | '.3gpp' | '.avi' |"
];

SetUsage[WriteNextVideoFrame,
	"WriteNextVideoFrame[image$] writes in input 2D 'image$' to a new frame in the " <>
	"initialized writer." <> "\n" <>
	"Use InitVideoWriter$ to initialize a writer."
];

SetUsage[WriteVideo,
	"WriteVideo['file$', image$, br$, fr$, indFrDur$, codec$] encodes raw video data " <>
	"to file using the following properties:" <> "\n" <>
    "'file$' is a string of the output video file name." <> "\n" <>
    "image$ is a 3D Image consisting of the video frames." <> "\n" <>
	"br$ is a positive integer specifying the bit rate of the resulting video." <> "\n" <>
	"fr$ is a positive integer specifying the average frame rate of the resulting video "  <>
	"(indFrDur$ should be set to empty list {} in this case)." <> "\n" <>
	"indFrDur$ is a list containing individual frame duration of each video frame of the output video. " <>
	"Dimension of indFrDur$ should match the number of slices in the input image " <>
	"(fr$ should be set to 0 in this case)." <> "\n" <>
    "codec$ is a string specifying the video encoding of the video stream of 'file$'." <> "\n" <>
	"Use SupportedVideoEncodings to get the available codecs." <> "\n" <>
	"Supported video file extensions include:" <> "\n" <>
	"| '.mov' | '.mp4' | '.3gp' | '.3gpp' | '.avi' |"
];

SetUsage[FinalizeVideoWriter,
	"FinalizeVideoWriter[] finalizes the active video writer." <> "\n" <>
	"Returns True$ if the writer is successfully finalized, False$ otherwise."
];

SetUsage[SupportedVideoEncodings,
	"SupportedVideoEncodings[type$] returns a list of all available video codecs " <>
	"for the corresponding file type." <> "\n" <>
	"Supported file types are 'QuickTime' and 'AVI'."
];

SetUsage[InitMediaTools,
	"Initializes the MediaTools library."
];


$InitMediaTools = False;
$MediaToolsLibrary = "MediaTools";
$MediaToolsBaseDirectory = FileNameDrop[$InputFileName, -2];
$BaseLibraryDirectory = FileNameJoin[{$MediaToolsBaseDirectory, "LibraryResources", $SystemID}];

Once[Get[FileNameJoin[{$MediaToolsBaseDirectory, "LibraryResources", "LibraryLinkUtilities.wl"}]]];


InitMediaTools[debug_:False] := If[TrueQ[$InitMediaTools],
	$InitMediaTools,
	$InitMediaTools = Catch[
		Block[{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},
			SafeLibraryLoad[$MediaToolsLibrary];
			
			RegisterPacletErrors[$MediaToolsLibrary, <||>];

			$DeleteAudioCache = SafeLibraryFunction["DeleteAudioCache", {"UTF8String"},  True|False];
			$DeleteVideoCache = SafeLibraryFunction["DeleteVideoCache", {"UTF8String"},  True|False];


		If[StringMatchQ[$OperatingSystem, "Windows" | "MacOSX"],

										(* Media Foundation Audio & Apple Core Audio *)

			$ReadAudioProperties = SafeLibraryFunction["ReadAudioProperties", LinkObject, LinkObject];
			$ReadAudioData = SafeLibraryFunction["ReadAudioData", {"UTF8String", Integer }, {"NumericArray"}];
			$WriteAudio = SafeLibraryFunction["WriteAudio", {{"NumericArray", "Constant"}, Integer, Integer, "UTF8String"}, {"UTF8String"}];
		];

		If[SameQ[$OperatingSystem, "Windows"],

												(* Media Foundation Video *)
																				
			$MFSupportedVideoEncodings = SafeLibraryFunction["MFSupportedVideoEncodings", {"UTF8String"}, {"UTF8String"}];

			$MFReadVideoProperties = SafeLibraryFunction["MFReadVideoProperties", LinkObject, LinkObject];
			$MFReadVideoFrameDurations = SafeLibraryFunction["MFReadVideoFrameDurations", {"UTF8String"}, {Real, 1}];
			$MFReadVideoFrameCount = SafeLibraryFunction["MFReadVideoFrameCount", {"UTF8String"}, Integer];
			$MFReadVideoData = SafeLibraryFunction["MFReadVideoData", {"UTF8String", {Integer, 1}}, {"Image3D"}];
			$MFWriteVideo = SafeLibraryFunction["MFWriteVideo", {"UTF8String", {"Image3D", "Constant"}, Integer, Integer, {Real, 3, "Constant"}, "UTF8String"}, {"UTF8String"}];
			
			$MFInitVideoReader = SafeLibraryFunction["MFInitVideoReader", {"UTF8String"}, True|False];
			$MFReadNextVideoFrame = SafeLibraryFunction["MFReadNextVideoFrame", {}, {"Image"}];
			$MFFinalizeVideoReader = SafeLibraryFunction["MFFinalizeVideoReader", {}, True|False];

			$MFInitVideoWriter = SafeLibraryFunction["MFInitVideoWriter", {"UTF8String", Integer, Integer, Integer, Integer, "UTF8String"}, True|False];
			$MFWriteNextVideoFrame = SafeLibraryFunction["MFWriteNextVideoFrame", {{"Image", "Constant"}}, True|False];
			$MFFinalizeVideoWriter = SafeLibraryFunction["MFFinalizeVideoWriter", {}, True|False];


														(* DirectShow Video *)

			$DSReadVideoProperties = SafeLibraryFunction["DSReadVideoProperties", LinkObject, LinkObject];
			$DSReadVideoFrameRate = SafeLibraryFunction["DSReadVideoFrameRate", {"UTF8String"}, Real];
			$DSReadVideoFrameCount = SafeLibraryFunction["DSReadVideoFrameCount", {"UTF8String"}, Integer];
			$DSReadVideoFrameDurations = SafeLibraryFunction["DSReadVideoFrameDurations", {"UTF8String"}, {Real, 1}];
			$DSReadVideoData = SafeLibraryFunction["DSReadVideoData", {"UTF8String", {Integer, 1}}, {"Image3D"}];

			$DSInitVideoReader = SafeLibraryFunction["DSInitVideoReader", {"UTF8String"}, True|False];
			$DSReadNextVideoFrame = SafeLibraryFunction["DSReadNextVideoFrame", {}, {"Image"}];
			$DSFinalizeVideoReader = SafeLibraryFunction["DSFinalizeVideoReader", {}, True|False];
		];


		If[SameQ[$OperatingSystem, "MacOSX"],

														(* AVFoundation Video *)

			$AVFSupportedVideoEncodings = SafeLibraryFunction["AVFSupportedVideoEncodings", {"UTF8String"}, {"UTF8String"}];
			$AVFNeedsModernization = SafeLibraryFunction["AVFNeedsModernizationQ", {"UTF8String"}, True|False];

			$AVFReadVideoProperties = SafeLibraryFunction["AVFReadVideoProperties", LinkObject, LinkObject];
			$AVFReadVideoFrameCount = SafeLibraryFunction["AVFReadVideoFrameCount", {"UTF8String"}, Integer];
			$AVFReadVideoFrameDurations = SafeLibraryFunction["AVFReadVideoFrameDurations", {"UTF8String"}, {Real, 1}];
			$AVFReadVideoData = SafeLibraryFunction["AVFReadVideoData", {"UTF8String", {Integer, 1}}, {"Image3D"}];
			$AVFWriteVideo = SafeLibraryFunction["AVFWriteVideo", {"UTF8String", {"Image3D", "Constant"}, Integer, Integer, {Real, 3, "Constant"}, "UTF8String"}, {"UTF8String"}];
			$AVFInitVideoReader = SafeLibraryFunction["AVFInitVideoReader", {"UTF8String"}, True|False];
			$AVFReadNextVideoFrame = SafeLibraryFunction["AVFReadNextVideoFrame", {}, {"Image"}];
			$AVFFinalizeVideoReader = SafeLibraryFunction["AVFFinalizeVideoReader", {}, True|False];

			$AVFInitVideoWriter = SafeLibraryFunction["AVFInitVideoWriter", {"UTF8String", Integer, Integer, Integer, Integer, "UTF8String"}, True|False];
			$AVFWriteNextVideoFrame = SafeLibraryFunction["AVFWriteNextVideoFrame", {{"Image", "Constant"}}, True|False];
			$AVFFinalizeVideoWriter = SafeLibraryFunction["AVFFinalizeVideoWriter", {}, True|False];
		];

		];
	True
	]
]

ReadAudioProperties[file_, streamID_ : 1] := Block[{res},
	res = Quiet[$ReadAudioProperties[file]];
	If[!AssociationQ[res], Return[res]];
	res = Lookup[res, streamID, Return[Failure["InvaildInput", <|"MessageTemplate" -> "Invalid audio track id"|>]]];
	res = Map[If[(NumberQ[#] && # === 0) || (StringQ[#] && # === ""), Missing["NotAvailable"], #] &, res];
	Return[res];
];

ReadAudioData[file_, streamID_ : 1] := Catch[Quiet[$ReadAudioData[file, streamID]]];

WriteAudio[data_, sampleRate_, bitRate_, file_] := Catch[Quiet[$WriteAudio[data, sampleRate, bitRate, file]]];

SupportedVideoEncodings[format_] := Block[{res},
	Switch[$OperatingSystem,
		"Windows",
		res = $MFSupportedVideoEncodings[format],
		"MacOSX",
		res = $AVFSupportedVideoEncodings[format],
		_,
		res = ""
	];
	res = StringSplit[res, ","];
	Return[res];
];

ReadVideoProperties[file_] := Block[{res, frameRate},	
	Switch[$OperatingSystem,	
		"Windows",
		(* Try to use MediaFoundation *)
    	res = Quiet[$MFReadVideoProperties[file]];
    	(* Try to use DirectShow *)
        If[!AssociationQ[res],
        	res = Quiet[$DSReadVideoProperties[file]];
        	If[!AssociationQ[res], Return[res]];
        	(* Read FrameRate manually for DirectShow *)
        	frameRate = Quiet[$DSReadVideoFrameRate[file]];
        	If[!Internal`RealValuedNumberQ[frameRate],
        		frameRate = Missing["NotAvailable"];
        	];
        	AssociateTo[res, "FrameRate" -> frameRate];
        ];
        ,
        "MacOSX",
        (* Try to use AVFoundation *)
    	res = Quiet[$AVFReadVideoProperties[file]];
        If[!AssociationQ[res], Return[res]];
        ,
        (* Not supported OS *)
        _,
        Return[Failure["OSError", <|"MessageTemplate" -> "MediaTools is not supported in this OS"|>]];
    ];
    res = Replace[res, (_?PossibleZeroQ | "") :> Missing["NotAvailable"], {1}];
    Return[res];
];

ReadVideoFrameCount[file_] := Block[{res},
    If[SameQ[$OperatingSystem, "Windows"],
        res = Quiet[$MFReadVideoFrameCount[file]];
        If[!Internal`PositiveIntegerQ[res], res = Quiet[$DSReadVideoFrameCount[file]]];
    ];
    If[SameQ[$OperatingSystem, "MacOSX"],
    	res = Quiet[$AVFReadVideoFrameCount[file]];    	
    ];
    Return[res];
];

ReadVideoFrameDurations[file_] := Block[{res},
    If[SameQ[$OperatingSystem, "Windows"],
    	res = Quiet[$MFReadVideoFrameDurations[file]];
        If[!ListQ[res] || Length[res] === 0, res = Quiet[$DSReadVideoFrameDurations[file]]];
    ];
    If[SameQ[$OperatingSystem, "MacOSX"],
    	res = Quiet[$AVFReadVideoFrameDurations[file]];
    ];
    If[!Length[res], Return[Failure["InfoGetError", <|"MessageTemplate" -> "Failed to get video frame durations"|>]]];
    Return[res];
];

ReadVideoData[file_, frames_] := Block[{res},
    If[SameQ[$OperatingSystem, "Windows"],
    	res = Quiet[$MFReadVideoData[file, frames]];
        If[!Image`Image3DQ[res], res = Quiet[$DSReadVideoData[file, frames]]];
    ];
    If[SameQ[$OperatingSystem, "MacOSX"],
    	res = Quiet[$AVFReadVideoData[file, frames]];
    ];
    Return[res];
];

WriteVideo[fileName_, frames_, bitRate_, frameRate_, frameDurations_, codec_] := Block[{res},
	  If[SameQ[$OperatingSystem, "Windows"],
	  	res = Quiet[$MFWriteVideo[fileName, frames, bitRate, frameRate, frameDurations, codec]];
	  ];
	  If[SameQ[$OperatingSystem, "MacOSX"],
	  	res = Quiet[$AVFWriteVideo[fileName, frames, bitRate, frameRate, frameDurations, codec]];
	  ];
	  Return[res];
];

OOCReader = 0;
OOCFileName = "";

InitializeVideoReader[fileName_] := Block[{res},
	If[SameQ[$OperatingSystem, "Windows"],
		OOCReader = 1;
		If[!(res = Quiet[$MFInitVideoReader[fileName]]) || !ListQ[res = Quiet[$MFReadVideoProperties[fileName]]],
			If[!(res = Quiet[$DSInitVideoReader[fileName]]), OOCReader = 0; Return[res]];
			OOCReader = 2;
		];
	];
	If[SameQ[$OperatingSystem, "MacOSX"],
		If[!Quiet[res = $AVFInitVideoReader[fileName]],
			Return[res];
		];
		OOCReader = 3;
	];
	OOCFileName = fileName;
	
	Return[True];
]

ReadNextVideoFrame[] := Block[{res},
	If[OOCReader == 0, Return[$Failed]];
	
	If[SameQ[$OperatingSystem, "Windows"],
		If[OOCReader == 1, res = Quiet[$MFReadNextVideoFrame[]];
			If[FailureQ[res],
				If[!(res = Quiet[$DSInitVideoReader[OOCFileName]]),
					OOCReader = 0; 
					Return[res];
				];
				OOCReader = 2;
			];
		];
		If[OOCReader == 2, res = Quiet[$DSReadNextVideoFrame[]]];
	];
	If[SameQ[$OperatingSystem, "MacOSX"],
		If[OOCReader == 3, res = Quiet[$AVFReadNextVideoFrame[]]];
	];
	Return[res];
];

FinalizeVideoReader[] := Block[{res},
	If[OOCReader == 0, Return[Failure["InvalidReader", <|"MessageTemplate" -> "Reader is not initialized"|>]]];
	If[SameQ[$OperatingSystem, "Windows"],
		If[OOCReader == 1, res = Quiet[$MFFinalizeVideoReader[]]];
		If[OOCReader == 2, res = Quiet[$DSFinalizeVideoReader[]]];
	];
	If[SameQ[$OperatingSystem, "MacOSX"],
		If[OOCReader == 3, res = Quiet[$AVFFinalizeVideoReader[]]];
	];
	Return[res];
];

InitializeVideoWriter[fileName_, width_, height_, bitRate_, frameRate_, codec_] := Block[{res},
	If[SameQ[$OperatingSystem, "Windows"],
		res = Quiet[$MFInitVideoWriter[fileName, width, height, bitRate, frameRate, codec]];
	];
	If[SameQ[$OperatingSystem, "MacOSX"],
		res = Quiet[$AVFInitVideoWriter[fileName, width, height, bitRate, frameRate, codec]];
	];
	Return[res];
];

WriteNextVideoFrame[image_] := Block[{res},
	If[SameQ[$OperatingSystem, "Windows"],
		res = Quiet[$MFWriteNextVideoFrame[image]];
	];
	If[SameQ[$OperatingSystem, "MacOSX"],
		res = Quiet[$AVFWriteNextVideoFrame[image]];
	];
	Return[res];
];

FinalizeVideoWriter[] := Block[{res},
	If[SameQ[$OperatingSystem, "Windows"],
		res = Quiet[$MFFinalizeVideoWriter[]];
	];
	If[SameQ[$OperatingSystem, "MacOSX"],
		res = Quiet[$AVFFinalizeVideoWriter[]];
	];
	Return[res];
];

InitMediaTools[];

End[]
EndPackage[]
