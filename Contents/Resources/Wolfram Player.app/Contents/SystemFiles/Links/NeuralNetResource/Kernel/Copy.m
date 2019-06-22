(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryresourcedownload[$NeuralNetResourceType,args___]:=resourcedownload0[args]

resourcedownload0[id_String,res_]:=Block[{format, location, elem=Lookup[res,"Element",Automatic], dlres},
    storeContentFunctions[id, res];
	dlres=
	If[KeyExistsQ[res,"ContentFormat"],
        Flatten[DataResource`Private`dataresourceexport[id,Lookup[res,"ContentFormat"],elem,res]]
		,
		If[KeyExistsQ[res,"DownloadInfo"],
	        neuralnetresourcedownload[id,Lookup[res,"DownloadInfo",Throw[$Failed]],elem]
	        ,
	        $Failed
	    ]
	];
    If[ListQ[dlres],
    	{location, format}=dlres,
    	Return[$Failed]
    ];
    ResourceSystemClient`Private`storeDownloadVersion[id,res, 
    	Association[res,"Element"->elem,"Location"->location,"Format"->format]]
]

resourcedownload0[___]:=$Failed

neuralnetresourcedownload[id_,downloadinfo_List,elem_]:=neuralnetresourcefileDownload[id,First[downloadinfo], elem]
neuralnetresourcedownload[id_,downloadinfo_,elem_]:=neuralnetresourcefileDownload[id,downloadinfo, elem]
neuralnetresourcedownload[__]:=$Failed

neuralnetresourcefileDownload[id_,fileinfo_,elem_]:=Block[{format, cos},
	{format,cos}=Lookup[fileinfo,{"Format","ContentElementLocations"}];
	neuralnetresourcefiledownload[id,format,cos,elem, fileinfo]
]


$nnCopyFileFormats=("NB"|"Package");
neuralnetresourcefiledownload[id_,fmt:$nnCopyFileFormats,co:HoldPattern[_CloudObject|_URL], elem_,_]:=resourcecopyfiledownload["NeuralNet",id,fmt,co,elem]

neuralnetresourcefiledownload[id_,fmt_,co:HoldPattern[_CloudObject|_URL], elem_,fileinfo_]:=resourcefiledownload["NeuralNet", id, fmt, co, elem,fileinfo]

neuralnetresourcefiledownload[___]:=$Failed

ResourceSystemClient`Private`updateRepositoryResourceInfo[$NeuralNetResourceType,id_,info_, as_]:=updateResourceInfoElements["NeuralNet", id, info,as]

ResourceSystemClient`Private`repositorycloudResourceDownload[$NeuralNetResourceType, info_, as_]:=
	cloudresourceDownload["NeuralNet",info,info["ContentElementLocations"], Lookup[as,"Element"]]/;KeyExistsQ[info,"ContentElementLocations"]


ResourceSystemClient`Private`determineSubmissionContentFormat[$NeuralNetResourceType,key_,val_]:=determineNNSubmissionContentType[key,val]
determineNNSubmissionContentType["UninitializedEvaluationNet"|"EvaluationNet", _]:="WLNet"
determineNNSubmissionContentType[_, net_]:="WLNet"/;(NetChain;NeuralNetworks`ValidNetQ[net])
determineNNSubmissionContentType[_, _]:="Binary"


ResourceSystemClient`Private`resourceDownloadStorageInfo[$NeuralNetResourceType, location_, format_]:=Association["Location"->location,"Format"->format];

End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];