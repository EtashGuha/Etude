BeginPackage["NumericArrayUtilities`"]

Begin["Private`"]

Get @ FileNameJoin[{DirectoryName @ $InputFileName, "Common.m"}];

LoadLibraries[];

(* Forces library initialization. Wrap in CatchFailure to prevent
   Panic from calling Abort[] if ReturnErrorString cannot be loaded *)
GeneralUtilities`CatchFailure @ ReturnErrorString[];

End[]
EndPackage[]