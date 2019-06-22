BeginPackage["SpeechSynthesisTools`"]
Begin["`Private`"]

$InitSpeechSynthesisTools = False;

$PacletDirectory = FileNameDrop[$InputFileName, -2]
If[$SystemID == "Linux-ARM",
	$BaseLibraryDirectory = FileNameJoin[{"","usr","lib","arm-linux-gnueabihf"}],
	$BaseLibraryDirectory = FileNameJoin[{$PacletDirectory, "LibraryResources", $SystemID}];
];

Get[FileNameJoin[{$PacletDirectory, "LibraryResources", "LibraryLinkUtilities.wl"}]];

$SpeechSynthesisToolsLibrary = "SpeechSynthesisTools";
	
dlls["Linux"|"Linux-x86-64"] = { 
	"libflite.so",
	"libflite_cmu_grapheme_lang.so", "libflite_cmu_grapheme_lex.so", 
	"libflite_cmulex.so",
	"libflite_cmu_indic_lang.so", "libflite_cmu_indic_lex.so", 
	"libflite_cmu_time_awb.so","libflite_cmu_us_awb.so", 
	"libflite_cmu_us_kal.so", "libflite_cmu_us_kal16.so", 
	"libflite_cmu_us_rms.so", "libflite_cmu_us_slt.so", 
	"libflite_usenglish.so"
	};
	
dlls["Linux-ARM"] = { 
	"libflite.so.1",
	"libflite_cmu_grapheme_lang.so.1", "libflite_cmu_grapheme_lex.so.1", 
	"libflite_cmulex.so.1",
	"libflite_cmu_indic_lang.so.1", "libflite_cmu_indic_lex.so.1", 
	"libflite_cmu_time_awb.so.1","libflite_cmu_us_awb.so.1", 
	"libflite_cmu_us_kal.so.1", "libflite_cmu_us_kal16.so.1", 
	"libflite_cmu_us_rms.so.1", "libflite_cmu_us_slt.so.1", 
	"libflite_usenglish.so.1"
	};
	
dlls["MacOSX-x86-64" | "MacOSX-x86"] = {};
dlls["Windows"|"Windows-x86-64"]     = {"Flite"};
dlls[___] := $Failed;

safeLibraryLoad[debug_, lib_] :=
	Quiet[
		Check[
			LibraryLoad[lib],
			If[TrueQ[debug],
				Print["Failed to load ", lib]
			];
			Throw[$InitSpeechSynthesisTools = $Failed]
		]
	]
safeLibraryFunctionLoad[debug_, args___] :=
	Quiet[
		Check[
			LibraryFunctionLoad[$SpeechSynthesisToolsLibrary, args],
			If[TrueQ[debug],
				Print["Failed to load the function ", First[{args}], " from ", $SpeechSynthesisToolsLibrary]
			];
			Throw[$InitSpeechSynthesisTools = $Failed]
		]
	]
    
InitSpeechSynthesisTools[debug_:False] := If[TrueQ[$InitSpeechSynthesisTools],
	$InitSpeechSynthesisTools,
	$InitSpeechSynthesisTools = Catch[
	  Block[{$LibraryPath = Append[$LibraryPath, $BaseLibraryDirectory]},
		  safeLibraryLoad[debug, #]& /@  Flatten[{dlls[$SystemID], $SpeechSynthesisToolsLibrary}];
		  
		  (*FLITE Initialize*)
		  $InitFLITE = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "InitFLITE", {}, "Void"];
		  (*FLITE Uninitialize*)
		  $DeInitFLITE = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "DeInitFLITE", {}, "Void"];
		  (*FLITE Synthesize*)
		  $FLITESynthesize = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "FLITESynthesize", {"UTF8String",{"UTF8String"}}, {"RawArray"}];
		  (*FLITE Save The Wave*)
		  $FLITESynthSave = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "FLITESynthSave", {"UTF8String",{"UTF8String"},{"UTF8String"}}, "Void"];
		  (*FLITE Extract Info*)
		  lf$FLITEExtractInfo = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "FLITEExtractInfo", {"UTF8String"}, {"UTF8String"}];
		  (*FLITE Available Voices*)
		  lf$FLITEAvailableVoices = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "FLITEAvailableVoices", {}, {"UTF8String"}];
		  
		  If[StringContainsQ[$Version, "Mac"],
		  (*MacSynth Initialize*)
			  $InitMacSynth = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "InitMacSynth", {"UTF8String"}, "Void"];
			  (*MacSynth Uninitialize*)
			  $DeInitMacSynth = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "DeInitMacSynth", {}, "Void"];
			  (*MacSynth Save The Wave*)
			  $MacSynthSave = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacSynthSave", {"UTF8String",{"UTF8String"}}, "Void"];
			  (*MacSynth Extract Info*)
			  lf$MacExtractInfo = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacExtractInfo", {"UTF8String"}, {"UTF8String"}];
			  (*MacSynth Available Voices*)
			  lf$MacAvailableVoices = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacAvailableVoices", {}, "RawArray"];
			  (*MacSynth Pitch Shift*)
			  $MacSetPitch = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacSetPitch", {Real}, "Void"];
			  $MacGetPitch = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacGetPitch", {}, Real];
			  (*MacSynth Rate Shift*)
			  $MacSetRate = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacSetRate", {Integer}, "Void"];
			  $MacGetRate = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacGetRate", {}, Integer];
			  (*MacSynth Volume Shift*)
			  $MacSetVolume = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacSetVolume", {Real}, "Void"];
			  $MacGetVolume = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacGetVolume", {}, Real];
			  (*MacSynth digit pronounce mode*)
			  $MacSetDigitByDigit = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacSetDigitByDigit", {Boolean}, "Void"];
			  (*MacSynth character pronounce mode*)
			  $MacSetCharByChar = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "MacSetCharByChar", {Boolean}, "Void"];
			];
		  
		  If[StringContainsQ[$Version, "Win"],
			  (*Create and initialize synth. *)
			  $InitWinSynth = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "InitWinSynth", {}, "Void"];
			  (*Uninitialize and destroy the synth. *)
			  $DeInitWinSynth = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "DeInitWinSynth", {}, "Void"];
			  (*Set the voice of the synth. Input: voice *)
			  $WinSetVoice = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinSetVoice", {"UTF8String"}, "Void"];
			  (*Get the current voice of the synth. *)
			  $WinGetVoice = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinGetVoice", {}, {"UTF8String"}];
			  (*Set the volume. Input: volume *)
			  $WinSetVolume = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinSetVolume", {Integer}, "Void"];
			  (*Get the current volume. Output: volume *)
			  $WinGetVolume = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinGetVolume", {}, {Integer}];
			  (*Set the rate. Input: rate *)
			  $WinSetRate = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinSetRate", {Integer}, "Void"];
			  (*Get the current rate. Output: rate *)
			  $WinGetRate = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinGetRate", {}, {Integer}];
			  (*Speak text and save to file. Input: speakString, filePath *)
			  $WinSynthSave = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinSynthSave", {"UTF8String", "UTF8String"}, "Void"];  
			  (*Get information about a voice. Input: voice *)
			  lf$WinExtractInfo = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinExtractInfo", {"UTF8String"}, {"UTF8String"}]; 	  
			  (*List available voices. Output: voicesListString *)
			  lf$WinAvailableVoices = CatchLibraryFunctionError@*safeLibraryFunctionLoad[debug, "WinAvailableVoices", {}, {"UTF8String"}];
		  	];

		  RegisterPacletErrors[$SpeechSynthesisToolsLibrary, <||>];
	  ];
	  True
	]
]

$FLITEAvailableVoices[] := (If[StringQ[#], StringSplit[#, "/"], #]& @ lf$FLITEAvailableVoices[])
$FLITEExtractInfo[voice_?StringQ] := SpeechSynthesisTools`Private`SpeechDump`stringToAssoc[lf$FLITEExtractInfo[voice], {"SampleRate", "Age", "F0TargetMean"}] 
$WinAvailableVoices[] /; StringContainsQ[$SystemID, "Win"] := (If[StringQ[#], StringSplit[#, " "], #]& @ lf$WinAvailableVoices[])
$WinExtractInfo[voice_?StringQ] /; (StringContainsQ[$SystemID, "Win"]) := SpeechSynthesisTools`Private`SpeechDump`stringToAssoc[lf$WinExtractInfo[voice]]
$MacAvailableVoices[] /; StringContainsQ[$SystemID, "Mac"] := (If[Developer`RawArrayQ[#], FromCharacterCode[SplitBy[Normal[#],(#==0)&][[1 ;; ;; 2]]], #]& @ lf$MacAvailableVoices[]) 
$MacExtractInfo[voice_?StringQ] /; (StringContainsQ[$SystemID, "Mac"]) := SpeechSynthesisTools`Private`SpeechDump`stringToAssoc[lf$MacExtractInfo[voice], {"Age"}] 

Begin["`SpeechDump`"]
stringToAssoc[s_?StringQ, numerics_:{}] := 
Quiet[Check[
	Association[
		Replace[Rule @@@ (StringSplit[#, ":"]& /@ StringSplit[s, "\n"])
			, Rule[x_,y_] /; MemberQ[numerics, x] :> Rule[x, ToExpression[y]]
			, {1}]]
	, s
]]
stringToAssoc[s_?FailureQ, ___] := s;
stringToAssoc[___] := $Failed;
End[]
End[]
EndPackage[]
