BeginPackage["MIDITools`"]
Begin["`Private`"]

$InitMIDITools = False;

InitMIDITools[] := 
 Module[{dir, dlls, MIDIToolsDll, DLLTable}, 
  DLLTable = {"MacOSX-x86-64" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "MIDITools", "LibraryResources", "MacOSX-x86-64"}],
  	 {"libglib.dylib", "libintl.dylib", "libreadline.dylib", "libfluidsynth.dylib"}}, 
    "Linux" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "MIDITools", "LibraryResources", "Linux"}],
    	 {"libfluidsynth.so"}}, 
    "Linux-x86-64" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "MIDITools", "LibraryResources", "Linux-x86-64"}],
    	 {"libfluidsynth.so"}}, 
    "Windows" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "MIDITools", "LibraryResources", "Windows"}],
    	 {"libfluidsynth.dll"}}, 
    "Windows-x86-64" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "MIDITools", "LibraryResources", "Windows-x86-64"}],
    	 {"libfluidsynth.dll"}}};
  dlls = Cases[DLLTable, ($SystemID -> l_) :> l];
  If[dlls === {},
  	Message[MIDITools::sys, "Incompatible SystemID"];
  	$InitMIDITools = $Failed;
  	Return[$Failed]];
  {dir, dlls} = First@dlls;
  Needs["CCompilerDriver`"];
  LibraryLoad[FileNameJoin[{dir, #}]] & /@ dlls;
  MIDIToolsDll = FileNameJoin[{dir, "MIDITools"}];
  $ErrorDescription = LibraryFunctionLoad[MIDIToolsDll, "ErrorDescription", {}, "UTF8String"];
  $SynthesizeMIDI = LibraryFunctionLoad[MIDIToolsDll,"SynthesizeMIDI", {"UTF8String"}, {_Real, _}];
  $InitializeFluidSynth = LibraryFunctionLoad[MIDIToolsDll,"InitializeFluidSynth", {"UTF8String", _Integer}, "Boolean"];
  $UninitializeFluidSynth = LibraryFunctionLoad[MIDIToolsDll,"UninitializeFluidSynth", {}, "Boolean"];
  $InitializeAudioDriver = LibraryFunctionLoad[MIDIToolsDll,"InitializeAudioDriver", {"UTF8String"}, "Boolean"];
  $GetCurrentAudioDriver = LibraryFunctionLoad[MIDIToolsDll,"GetCurrentAudioDriver", {}, "UTF8String"];
  $GetAvailableAudioDrivers = LibraryFunctionLoad[MIDIToolsDll,"GetAvailableAudioDrivers", {}, "UTF8String"];
  $Noteon = LibraryFunctionLoad[MIDIToolsDll,"Noteon", {_Integer, _Integer, _Integer}, "Boolean"];
  $Noteoff = LibraryFunctionLoad[MIDIToolsDll,"Noteoff", {_Integer, _Integer}, "Boolean"];
  $CC = LibraryFunctionLoad[MIDIToolsDll,"CC", {_Integer, _Integer, _Integer}, "Boolean"];
  $PitchBend = LibraryFunctionLoad[MIDIToolsDll,"PitchBend", {_Integer, _Integer}, "Boolean"];
  $PitchWheelSens = LibraryFunctionLoad[MIDIToolsDll,"PitchWheelSens", {_Integer, _Integer}, "Boolean"];
  $ProgramChange = LibraryFunctionLoad[MIDIToolsDll,"ProgramChange", {_Integer, _Integer}, "Boolean"];
  $BankSelect = LibraryFunctionLoad[MIDIToolsDll,"BankSelect", {_Integer, _Integer}, "Boolean"];
  $PlayerSetTempo = LibraryFunctionLoad[MIDIToolsDll,"PlayerSetTempo", {_Integer}, "Boolean"];
  $PlayerSetBPM = LibraryFunctionLoad[MIDIToolsDll,"PlayerSetBPM", {_Integer}, "Boolean"];
  $PlayerSetLoop = LibraryFunctionLoad[MIDIToolsDll,"PlayerSetLoop", {_Integer}, "Boolean"];
  $InitMIDITools = If[$ErrorDescription === $Failed ||
    					$SynthesizeMIDI === $Failed ||
    					$InitializeFluidSynth === $Failed ||
    					$UninitializeFluidSynth === $Failed ||
    					$InitializeAudioDriver === $Failed ||
    					$GetCurrentAudioDriver === $Failed ||
    					$GetAvailableAudioDrivers === $Failed ||
    					$Noteon === $Failed ||
    					$Noteoff === $Failed ||
    					$CC === $Failed ||
    					$PitchBend === $Failed ||
    					$PitchWheelSens === $Failed ||
    					$ProgramChange === $Failed ||
    					$BankSelect === $Failed ||
    					$PlayerSetTempo === $Failed ||
    					$PlayerSetBPM === $Failed ||
    					$PlayerSetLoop === $Failed, $Failed, True];
  $InitMIDITools
]

End[]
EndPackage[]
