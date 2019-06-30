(* Mathematica Package *)
  
BeginPackage["GoogleContactsLoad`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

If[!ListQ[System`$Services],Get["OAuth`"]]

Block[{dir=DirectoryName[System`Private`$InputFileName]},
	OAuthClient`addOAuthservice["GoogleContacts",dir]
]


End[] (* End Private Context *)
EndPackage[]
