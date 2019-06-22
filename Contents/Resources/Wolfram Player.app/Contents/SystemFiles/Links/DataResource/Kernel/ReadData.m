
(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryResourceExecute[$DataResourceType,id_,info_,rest___]:=
   readResourceElementContent[id,info,rest]
   
End[] (* End Private Context *)

EndPackage[]