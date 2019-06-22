(* Mathematica Package *)
  
BeginPackage["BingSearchLoad`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

If[!ListQ[System`$Services],Get["OAuth`"]]

Block[{dir=DirectoryName[System`Private`$InputFileName]},
	KeyClient`addKeyservice["BingSearch",dir]
]


End[] (* End Private Context *)
EndPackage[]
