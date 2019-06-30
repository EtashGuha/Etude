BeginPackage["MP3Tools`"]
Begin["`Private`"]

$InitMP3Tools = False;

InitMP3Tools[] := 
 Module[{dir, dlls, MP3ToolsDll, DLLTable}, 
  DLLTable = {"MacOSX-x86" -> {FileNameJoin[{$InstallationDirectory, 
        "SystemFiles", "Links", "MP3Tools", "LibraryResources", 
        "MacOSX-x86"}], {"libmp3lame.dylib", "libmad.dylib"}}, 
    "MacOSX-x86-64" -> {FileNameJoin[{$InstallationDirectory, 
        "SystemFiles", "Links", "MP3Tools", "LibraryResources", 
        "MacOSX-x86-64"}], {"libmp3lame.dylib", "libmad.dylib"}}, 
    "Linux" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles", 
        "Links", "MP3Tools", "LibraryResources", 
        "Linux"}], {"libmp3lame.so", "libmad.so"}}, 
    "Linux-x86-64" -> {FileNameJoin[{$InstallationDirectory, 
        "SystemFiles", "Links", "MP3Tools", "LibraryResources", 
        "Linux-x86-64"}], {"libmp3lame.so", "libmad.so"}},
	"Linux-ARM" -> {FileNameJoin[{$InstallationDirectory, 
        "SystemFiles", "Links", "MP3Tools", "LibraryResources", 
        "Linux-ARM"}], {}},
    "Windows" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles",
         "Links", "MP3Tools", "LibraryResources", 
        "Windows"}], {"libmp3lame.dll", "libmad.dll"}}, 
    "Windows-x86-64" -> {FileNameJoin[{$InstallationDirectory, 
        "SystemFiles", "Links", "MP3Tools", "LibraryResources", 
        "Windows-x86-64"}], {"libmp3lame.dll", "libmad.dll"}}};
  dlls = Cases[DLLTable, ($SystemID -> l_) :> l];
  If[dlls === {},
  	Message[MP3Tools::sys, "Incompatible SystemID"];
  	$InitMP3Tools = $Failed;
  	Return[$Failed]];
  {dir, dlls} = First@dlls;
  Needs["CCompilerDriver`"];
  LibraryLoad[FileNameJoin[{dir, #}]] & /@ dlls;
  MP3ToolsDll = FileNameJoin[{dir, "MP3Tools"}];
  $encodeMP3 = LibraryFunctionLoad[MP3ToolsDll,"EncodeMP3", {{_Real, _}, "UTF8String", _Integer, _Integer, _Real}, "UTF8String"];
  $decodeMP3 = LibraryFunctionLoad[MP3ToolsDll,"DecodeMP3", {"UTF8String"}, {_Real, _}];
  $readMetadataMP3 = LibraryFunctionLoad[MP3ToolsDll,"ReadMetadataMP3", {"UTF8String"}, {_Integer, _}];
  $InitMP3Tools = If[ $encodeMP3 === $Failed || $decodeMP3 === $Failed || $readMetadataMP3 === $Failed, $Failed, True];
  $InitMP3Tools
]

End[]
EndPackage[]
