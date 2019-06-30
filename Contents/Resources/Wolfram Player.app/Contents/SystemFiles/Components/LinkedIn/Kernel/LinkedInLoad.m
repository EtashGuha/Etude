(* Mathematica Package *)
  
BeginPackage["LinkedInLoad`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

If[!ListQ[System`$Services],Get["OAuth`"]]

Block[{dir=DirectoryName[System`Private`$InputFileName]},
	OAuthClient`addOAuthservice["LinkedIn",dir]
]


End[] (* End Private Context *)
EndPackage[]
