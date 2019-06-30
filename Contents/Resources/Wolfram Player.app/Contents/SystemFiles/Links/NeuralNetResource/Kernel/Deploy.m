(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryclouddeployResourceContent[$NeuralNetResourceType,id_,  info_, rest___]:=cloudDeployResourceContentElements[id, info,rest]

ResourceSystemClient`Private`repositoryclouddeployResourceInfo[$NeuralNetResourceType,id_,  localinfo_, newinfo_, rest___]:=
	clouddeployResourceInfoWithElements["NeuralNet",id,  localinfo, newinfo, rest]

ResourceSystemClient`Private`repositoryBundleResourceObject[$NeuralNetResourceType,id_, localinfo_Association]:=
	bundleResourceObjectWithElementFunctions["NeuralNet"_,id, localinfo]/;MemberQ[$localResources, id]


End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];