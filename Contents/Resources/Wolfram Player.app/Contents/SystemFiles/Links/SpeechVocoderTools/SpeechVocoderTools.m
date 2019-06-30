BeginPackage["SpeechVocoderTools`"]
Begin["`Private`"]

Get[FileNameJoin[{FileNameDrop[$InputFileName, -1], "LibraryResources", "LibraryLinkUtilities.wl"}]];

$InitSpeechVocoderTools = False;

$ThisDirectory = FileNameDrop[$InputFileName, -1]
$BaseLibraryDirectory = FileNameJoin[{$ThisDirectory, "LibraryResources", $SystemID}];
$SpeechVocoderToolsLibrary = "SpeechVocoderTools";

InitSpeechVocoderTools[debug_:False] := If[TrueQ[$InitSpeechVocoderTools],
	$InitSpeechVocoderTools,
	$InitSpeechVocoderTools = Catch[
		Block[{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},

			SafeLibraryLoad[$SpeechVocoderToolsLibrary];

			RegisterPacletErrors[$SpeechVocoderToolsLibrary, <||>];

			(* Import *)
			$InitWorld = SafeLibraryFunction[
				"InitWorld",
				{
					Integer,
					Integer,
					Integer,
					Real,
					Real,
					Real,
					Integer
				},
				True|False
			];
			$DeInitWorld = SafeLibraryFunction[
				"DeInitWorld",
				{},
				True|False
			];
			$Analysis = SafeLibraryFunction[
				"Analysis",
				{{"RawArray", "Constant"}},
				True|False
			];
			$ReSynthesis = SafeLibraryFunction[
				"ReSynthesis",
				{},
				"RawArray"
			];
			$GetSpectrogramDimensions = SafeLibraryFunction[
				"GetSpectrogramDimensions",
				{},
				{Integer, 1}
			];
			$GetSampleRate = SafeLibraryFunction[
				"GetSampleRate",
				{},
				Integer
			];
			$GetWindowSize = SafeLibraryFunction[
				"GetWindowSize",
				{},
				Integer
			];
			$GetFramePeriod = SafeLibraryFunction[
				"GetFramePeriod",
				{},
				Real
			];
			$GetWindowOffset = SafeLibraryFunction[
				"GetWindowOffset",
				{},
				Integer
			];
			$GetF0Floor = SafeLibraryFunction[
				"GetF0Floor",
				{},
				Real
			];
			$GetF0 = SafeLibraryFunction[
				"GetF0",
				{},
				{Real, 1}
			];
			$GetSpectralEnvelope = SafeLibraryFunction[
				"GetSpectralEnvelope",
				{},
				{Real, 2}
			];
			$GetAperiodicity = SafeLibraryFunction[
				"GetAperiodicity",
				{},
				{Real, 2}
			];
			$SetF0 = SafeLibraryFunction[
				"SetF0",
				{{Real, 1}},
				True|False
			];
			$SetSpectralEnvelope = SafeLibraryFunction[
				"SetSpectralEnvelope",
				{{Real, 2}},
				True|False
			];
			$SetAperiodicity = SafeLibraryFunction[
				"SetAperiodicity",
				{{Real, 2}},
				True|False
			];
			$SetAllParameters = SafeLibraryFunction[
				"SetAllParameters",
				{{Real, 1}, {Real, 2}, {Real, 2}},
				True|False
			];
		];
		True
	]
]

End[]
EndPackage[]
