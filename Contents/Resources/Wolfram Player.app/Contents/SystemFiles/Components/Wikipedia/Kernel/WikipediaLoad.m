(* Mathematica Package *)

Begin["WikipediaLoad`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

If[!ListQ[System`$Services],Get["OAuth`"]]

Block[{dir=DirectoryName[System`Private`$InputFileName]},
	(*OtherClient`addOtherservice["Wikipedia",dir]*)
	KeyClient`addKeyservice["Wikipedia",dir]
]


End[] (* End Private Context *)
End[]