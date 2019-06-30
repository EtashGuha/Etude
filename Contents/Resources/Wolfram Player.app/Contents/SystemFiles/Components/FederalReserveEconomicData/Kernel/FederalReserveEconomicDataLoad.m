(* Mathematica Package *)
  
BeginPackage["FederalReserveEconomicDataLoad`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

If[!ListQ[System`$Services],Get["OAuth`"]]

Block[{dir=DirectoryName[System`Private`$InputFileName]},
	KeyClient`addKeyservice["FederalReserveEconomicData",dir]
]


End[] (* End Private Context *)
EndPackage[]
