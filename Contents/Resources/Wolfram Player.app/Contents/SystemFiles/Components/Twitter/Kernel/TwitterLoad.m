(* Mathematica Package *)
  
BeginPackage["TwitterLoad`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

If[!ListQ[System`$Services],Get["OAuth`"]]

Block[{dir=DirectoryName[System`Private`$InputFileName]},
	OAuthClient`addOAuthservice["Twitter",dir]
]


End[] (* End Private Context *)
EndPackage[]
