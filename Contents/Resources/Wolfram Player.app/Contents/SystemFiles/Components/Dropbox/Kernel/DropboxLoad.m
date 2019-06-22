(* Mathematica Package *)
  
BeginPackage["DropboxLoad`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

If[!ListQ[System`$Services],Get["OAuth`"]]

Block[{dir=DirectoryName[System`Private`$InputFileName]},
	OAuthClient`addOAuthservice["Dropbox",dir]
]


End[] (* End Private Context *)
EndPackage[]
