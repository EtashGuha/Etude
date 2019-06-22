(* Mathematica Package *)
  
BeginPackage["SeatGeekLoad`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

If[!ListQ[System`$Services],Get["OAuth`"]]

Block[{dir=DirectoryName[System`Private`$InputFileName]},
	KeyClient`addKeyservice["SeatGeek",dir]
]


End[] (* End Private Context *)
EndPackage[]
