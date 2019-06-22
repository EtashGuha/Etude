BeginPackage["DTWTools`"]
Begin["`Private`"]

InitDTWTools[] := 
  Module[{dir, DTWToolsDll, DirTable}, 
   DirTable = {"MacOSX-x86-64" -> FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", 
         "DTWTools", "LibraryResources", "MacOSX-x86-64"}], 
     "Linux" -> FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", 
     	"DTWTools", "LibraryResources", "Linux"}], 
     "Linux-x86-64" -> FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", 
     	"DTWTools", "LibraryResources", "Linux-x86-64"}], 
     "Linux-ARM" -> FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", 
     	"DTWTools", "LibraryResources", "Linux-ARM"}], 
     "Windows" -> FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", 
     	"DTWTools", "LibraryResources", "Windows"}], 
     "Windows-x86-64" -> FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", 
     	"DTWTools", "LibraryResources", "Windows-x86-64"}]};
   dir = Cases[DirTable, ($SystemID -> l_) :> l];
   If[dir === {}, Message[DTWTools::sys, "Incompatible SystemID"];
    $InitDTWTools = $Failed;
    Return[$Failed]];
   dir = dir[[1]];
   Needs["CCompilerDriver`"];

   DTWToolsDll = FileNameJoin[{dir, "DTWTools"}];
   $errorDescription = LibraryFunctionLoad[DTWToolsDll, "errorDescription", {}, "UTF8String"];
   $fastDTWDist = LibraryFunctionLoad[DTWToolsDll,"fastDTWDistance", 
   		(* TimeSeries 1, TimeSeries 2, DistanceFunction name, StepPattern name, Search radius *)
   		{{Real, _,"Constant"}, {Real, _,"Constant"}, "UTF8String", "UTF8String", Integer}, Real];
   $strictDTWDist = LibraryFunctionLoad[DTWToolsDll,"strictDTWDistance", 
   		(* TimeSeries 1, TimeSeries 2, DistanceFunction name, StepPattern name, SearchWindow name, SearchWindow parameter *)
   		{{Real, _,"Constant"}, {Real, _,"Constant"}, "UTF8String", "UTF8String", "UTF8String", Real}, Real];
   $openEndDTWDist = LibraryFunctionLoad[DTWToolsDll, "openEndDTWDistance", 
   		(* TimeSeries 1, TimeSeries 2, DistanceFunction name, StepPattern name, SearchWindow name, SearchWindow parameter, use OpenBegin *)
   		{{Real, _,"Constant"}, {Real, _,"Constant"}, "UTF8String", "UTF8String", "UTF8String", Real, "Boolean"}, {Real, 1}];
   $fastDTWPath = LibraryFunctionLoad[DTWToolsDll,"fastDTWCorrespondence", 
   		{{Real, _,"Constant"}, {Real, _,"Constant"}, "UTF8String", "UTF8String", Integer}, {Integer, 2}];
   $strictDTWPath = LibraryFunctionLoad[DTWToolsDll,"strictDTWCorrespondence", 
   		{{Real, _,"Constant"}, {Real, _,"Constant"}, "UTF8String", "UTF8String", "UTF8String", Real}, {Integer, 2}];
   $openEndDTWPath = LibraryFunctionLoad[DTWToolsDll,"openEndDTWCorrespondence", 
   		{{Real, _,"Constant"}, {Real, _,"Constant"}, "UTF8String", "UTF8String", "UTF8String", Real, "Boolean"}, {Integer, 2}];
   $InitDTWTools = If[$errorDescription === $Failed || $fastDTWDist === $Failed || $strictDTWDist === $Failed
   					|| $fastDTWPath === $Failed || $strictDTWPath === $Failed || $openEndDTWDist === $Failed || $openEndDTWPath === $Failed,
   					 $Failed, True];
   $InitDTWTools
];

End[]
EndPackage[]