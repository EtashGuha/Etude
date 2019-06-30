(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositorystandardizeContentMetadata[$NeuralNetResourceType,id_, info_]:=
	standardizeContentMetadataWithElements["NeuralNet",id, info]

standardizeContentMetadataContentInfo["NeuralNet",default_, locations_, contentelements_,moreinfo_]:=
	contentElementSize[locations,contentelements, moreinfo,default]

ResourceSystemClient`Private`resourceElementStorageSizeLimit["NeuralNet"]=0;


ResourceSystemClient`Private`repositorysaveresourceobject[$NeuralNetResourceType,info_]:=saveresourceobjectwithelements[info]


End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];