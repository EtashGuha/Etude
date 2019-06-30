(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryresourcedownload[$DataResourceType,args___]:=resourcedownload0[args]

resourcedownload0[id_String,res_]:=Block[{formats, locations, elem=Lookup[res,"Element",Automatic]},
	storeContentFunctions[id, res];
	{locations, formats}=If[KeyExistsQ[res,"ContentFormat"],
        dataresourceexport[id,Lookup[res,"ContentFormat"],elem,res]
        ,
        If[KeyExistsQ[res,"DownloadInfo"],
            dataresourcedownload[id,Lookup[res,"DownloadInfo",Throw[$Failed]],elem]
            ,
            {$Failed,$Failed}
        ]
    ];
    ResourceSystemClient`Private`storeDownloadVersion[id,res, 
    	Association[res,"Element"->elem,"Locations"->locations,
    		"Formats"->formats]]
    
]

resourcedownload0[___]:=$Failed

dataresourceexport[id_,format_,elem_,res_]:=Block[{content, file, dir,lo},
	content=Lookup[res,"Content",Throw[$Failed]];
	content=Switch[format,
		"Compressed",Uncompress,
		"PlainText",ToExpression,
		_,Identity][content];
	If[$CacheResourceContent===False,
		Throw[content,"NoCacheResourceDownload"]
		,
		dir=resourceCopyDirectory[id,"MX",elem];
		createDirectory[dir];
		lo=localObject[FileNameJoin[{dir,"data"}]];
		Export[lo, content,"MX"];
		Put[Association[
			{
				"Location"->lo,
				"DownloadDate"->DateObject[],
				"Format"->"MX",
				"Size"->bytecountQuantity[ByteCount[content]]
			}
		   ],
		   resourcecopyInfoFile[dir,"MX"]
		];
		{{lo},{"MX"}}
	]
]

dataresourcedownload[id_,downloadinfo_List,elem_]:=Transpose[dataresourcefileDownload[id,#, elem]&/@downloadinfo]
dataresourcedownload[id_,downloadinfo_,elem_]:=dataresourcedownload[id,{downloadinfo},elem]
dataresourcedownload[__]:=$Failed

dataresourcefileDownload[id_,fileinfo_,elem_]:=dataresourcefiledownload[id,elem,KeyTake[fileinfo,{"Format","ContentElementLocations","Hash"}]]


dataresourcefiledownload[id_,elem_,dlinfo_]:=Block[{n, progress=0, 
	dir,fmt,cos,lo,los={},size, temp},
	n=Length[cos];
	fmt=Lookup[dlinfo,"Format"];
	cos=Lookup[dlinfo,"ContentElementLocations"];
	dir=resourceCopyDirectory[id,fmt,elem];
	If[!ListQ[cos],Throw[$Failed]];
	If[Length[cos]>0,
		ResourceSystemClient`Private`$ProgressIndicatorContent=ResourceSystemClient`Private`downloadingMessageWindow[];
		temp=printTempOnce[ResourceSystemClient`$progressID];
		createDirectory[dir];
		({progress,lo}=dataresourcechunkdownload[id,fmt,dir,#,progress];
			AppendTo[los,lo])&/@cos;
		{lo,size}=mergeChunks[id,fmt,los];
		If[$CacheResourceContent===False,
			clearTempPrint[temp];
			Throw[Import[lo],"NoCacheResourceDownload"]
			,
			DeleteDirectory[dir,DeleteContents->True];
			Put[Association[
	        	{
	            "Location"->lo,
	            "DownloadDate"->DateObject[],
	            "Format"->"MX",
	            "Size"->bytecountQuantity[size]
	            }
	        ],
	        resourceCopyInfoFile[id,"MX",elem]
	        ];
	        clearTempPrint[temp];
			{lo,"MX"}
		]
		,
		Throw[$Failed]
	]
]/;MatchQ[Lookup[dlinfo,"Format"],("MXChunks"|"CompressedChunks")]

dataresourcechunkdownload[id_,fmt_,dir_,co:HoldPattern[_CloudObject],i_]:=With[{
	raw=fetchContent[ResourceSystemClient`ResourceDownload,Automatic,
		co, {"StatusCode","ContentData"},"VerifyPeer" -> False,"CredentialsProvider" -> None],
	lo=localObject[FileNameJoin[{dir,"data",ToString[i]}]]},
	If[!ListQ[raw],Throw[$Failed]];
    If[raw[[1]]===200,
    	Switch[fmt,
    		"MXChunks",
    		ResourceSystemClient`Private`resourcelocalrawexport["DataResource",lo,raw[[2]],"MX"],
    		"CompressedChunks",
    		Export[lo,raw,"String"],
    		_,
    		Throw[$Failed]
    	]
        ,
        Throw[$Failed]
    ];
    {i+1,lo}
]

mergeChunks[id_,"MXChunks",los_]:=Block[{dir=resourceCopyDirectory[id,"MX"],
	lo=localObject[FileNameJoin[{resourceCopyDirectory[id,"MX"],"data"}]]},
	(Export[lo,#,"MX"];{lo,ByteCount[#]})&@(Join@@(Import/@los))
]

mergeChunks[id_,"CompressedChunks",los_]:=Block[{dir=resourceCopyDirectory[id,"MX"],
	lo=localObject[FileNameJoin[{resourceCopyDirectory[id,"MX"],"data"}]]},
	(Export[lo,#,"MX"];{lo,ByteCount[#]})&@(Join@@(Uncompress[Import[#,"String"]]&/@los))
]

cloudexportfmts=("MX"|"PNG"|"Compressed");

dataresourcefiledownload[id_, elem_, dlinfo_]:=dataresourcefiledownload[id, elem, dlinfo["Format"],dlinfo["ContentElementLocations"],dlinfo]

dataresourcefiledownload[id_,elem_,fmt:cloudexportfmts,co:HoldPattern[_CloudObject|_URL], dlinfo_]:=
	resourcefiledownload["DataResource", id, elem, dlinfo]

dataresourcefiledownload[id_,elem_,Automatic|_String,co:HoldPattern[_URL], dlinfo_]:=
	resourcefiledownload["DataResource", id, elem, dlinfo]
	
dataresourcefiledownload[id_,elem_,_Missing,co:HoldPattern[_CloudObject], dlinfo_]:=ResourceSystemClient`Private`resourceimportFileDownload[
	"DataResource",id,Missing[],co, elem]

dataresourcefiledownload[___]:=$Failed
  
ResourceSystemClient`Private`updateRepositoryResourceInfo[rtype:$DataResourceType,id_,info0_, locations_, formats_, as_Association]:=
	ResourceSystemClient`Private`updateRepositoryResourceInfo[rtype,id,info0, Association[as,"Locations"->locations, "Formats"->formats]]

ResourceSystemClient`Private`updateRepositoryResourceInfo[$DataResourceType,id_,info_, as_]:=updateResourceInfoElements["DataResource", id, info,as]


ResourceSystemClient`Private`repositorycloudResourceDownload[$DataResourceType, info_, as_]:=
	cloudresourceDownload[info,info["ContentElementLocations"], Lookup[as,"Element"]]/;KeyExistsQ[info,"ContentElementLocations"]

ResourceSystemClient`Private`defaultResourceTypeDownloadFormat[$DataResourceType]="MX";
ResourceSystemClient`Private`resourceDownloadStorageInfo[$DataResourceType,location_,format_]:=Association["Locations"->{location},"Formats"->{format}];
        
End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];