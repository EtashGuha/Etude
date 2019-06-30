BeginPackage["Compile`Utilities`ZipList`"]

ZipList
ZipListWith

Begin["`Private`"]

(* Internal`GetUnboundSymbols[Function[{}, x]] *)
getUnboundSymbols := Internal`GetUnboundSymbols

ZipList[lists__] := Thread[List[lists]]
ZipListWith[f_, lists__] := Thread[f[lists]]


End[]


EndPackage[]
