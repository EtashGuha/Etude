(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryclouddeployResourceContent[$DataResourceType,id_,  info_, rest___]:=cloudDeployResourceContentElements[id, info,rest]

ResourceSystemClient`Private`repositoryclouddeployResourceInfo[$DataResourceType,id_,  localinfo_, newinfo_, rest___]:=
	clouddeployResourceInfoWithElements["DataResource",id,  localinfo, newinfo, rest]

ResourceSystemClient`Private`repositoryBundleResourceObject[$DataResourceType,id_, localinfo_Association]:=
	bundleResourceObjectWithElementFunctions["DataResource",id, localinfo]/;MemberQ[$localResources, id]


End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];