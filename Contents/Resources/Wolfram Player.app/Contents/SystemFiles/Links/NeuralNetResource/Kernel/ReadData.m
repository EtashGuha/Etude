
(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryResourceExecute[$NeuralNetResourceType,id_,info_,rest___]:=
    readResourceElementContent[id,info,rest]

End[] (* End Private Context *)

EndPackage[]