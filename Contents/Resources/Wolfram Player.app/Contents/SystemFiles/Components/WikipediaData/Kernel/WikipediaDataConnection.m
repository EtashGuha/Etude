(* Mathematica Package *)

BeginPackage["WikipediaData`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

Needs["PacletManager`"];
wikipedia = ServiceConnect["Wikipedia"];

End[] (* End Private Context *)

EndPackage[]
