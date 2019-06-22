(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryresourcedownload[$FunctionResourceTypes,args___]:=functionResourceDownload[args]

functionResourceDownload[id_String,as_]:=Block[{format, location},
	{location, format}=If[KeyExistsQ[as,"FunctionLocation"],
        functionresourceDownload[id,as["FunctionLocation"]]
        ,
        {$Failed,$Failed}
    ];
    ResourceSystemClient`Private`storeDownloadVersion[id,as, Association["FunctionLocation"->location,"Format"->format]];
    Association[as,"FunctionLocation"->location, "Format"->format,"UUID"->id]
]

functionResourceDownload[___]:=$Failed

functionresourceDownload[id_,location_CloudObject]:=functionresourcedownload[id,location]
neuralnetresourcedownload[__]:=$Failed

functionresourcedownload[id_,location_]:=ResourceSystemClient`Private`resourcefiledownload["Function", id, "Binary", location, Automatic]


ResourceSystemClient`Private`updateRepositoryResourceInfo[$FunctionResourceTypes,id_,info_, as_]:=updateFunctionResourceInfo[id, info,as]

updateFunctionResourceInfo[id_, info_,as_]:=Association[info,KeyTake[as,"FunctionLocation"]]


End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];