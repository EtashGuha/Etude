(* Wolfram Language Package *)


BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  
Begin["`Private`"] (* Begin Private Context *) 

standardizeCustomResourceInfo[ro:rpat]:=Block[{info, id},
	info=First[ro];
	If[!AssociationQ[info],Message[System`ResourceObject::invro];Return[ro]];
	If[KeyExistsQ[info,"UUID"],
		id=Lookup[info, "UUID"];
		If[MemberQ[id,$loadedResources],
			info=Join[resourceInfo[id],info]
		]
	];
	System`ResourceObject[usableResourceInfo[standardizecustomResourceInfo[info]]]
]


standardizecustomResourceInfo[info0_]:=Block[{id,info=info0},
	info=standardizeContentMetadata[id, info];
	sortBasicInfo[fillresourceInfo[info]]
]

standardizeContentMetadata[id_, info_]:=With[{rtype=getResourceType[info]},
	If[StringQ[rtype],
		loadResourceType[rtype];
		repositorystandardizeContentMetadata[rtype,id, info]
		,
		info
	]
	
]

repositorystandardizeContentMetadata[_,_, info_]:=info

autoloadResource[info_]:=With[{id=Lookup[info, "UUID",Throw[$Failed]]},
	If[MemberQ[$localResources,id],
		System`ResourceObject[id],
		resourceAcquire[id,False,resourcerepositoryBase[info]]
	]
]/;marketplacebasedResourceQ[info]

autoloadResource[info_]:=With[{id=Lookup[info, "UUID",Throw[$Failed]]},
	If[MemberQ[$localResources,id],
		System`ResourceObject[id],
		loadresource[id,standardizeContentMetadata[id, KeyDrop[info,"Autoload"]]];
		ResourceObject[standardizeContentMetadata[id, KeyDrop[info,"Autoload"]]]
	]
]/;userdefinedResourceQ[info]

fillResourceInfo[ro:rpat]:=Block[{info, id},
	info=First[ro];
	If[!AssociationQ[info],Message[System`ResourceObject::invro];Return[ro]];
	If[KeyExistsQ[info,"UUID"],
		id=Lookup[info, "UUID"];
		If[MemberQ[id,$loadedResources],
			info=Join[resourceInfo[id],info]
		]
	];
	System`ResourceObject[usableResourceInfo[fillresourceInfo[info]]]
]

fillresourceInfo[as_Association]:=Block[{info=as, moreinfo},
	If[!KeyExistsQ[info,"Name"],
		Message[ResourceObject::crname];Throw[$Failed]
	];
	If[!KeyExistsQ[info,"UUID"],
		info["UUID"]=CreateUUID[]
	];
	If[!KeyExistsQ[info,"ResourceType"],
		info["ResourceType"]=determineResourceType[info]
	];
	If[!KeyExistsQ[info,"Version"],
		info["Version"]=None
	];
	loadresource[info];
	info
]

fillresourceInfo[___]:=Throw[$Failed]

determineResourceType[info_]:=None

saveResourceObject[ro:rpat]:=saveresourceObject[First[ro]]
saveresourceObject[as_Association]:=$Failed/;!KeyExistsQ[as,"UUID"]

saveresourceObject[as_Association]:=saveresourceObject[Lookup[as,"UUID"],as]
saveresourceObject[id_String, as_]:=If[MemberQ[$localResources,id],
	ResourceObject[id],
	saveresourceobject[resourceInfo[id]]
]/;MemberQ[$loadedResources,id]

saveresourceObject[id_String, as_]:=ResourceObject[id]/;MemberQ[$localResources,id]

saveresourceObject[_,as_Association]:=saveresourceobject[as]
saveresourceObject[_]:=(Message[ResourceObject::invro];$Failed)
(*
saveresourceobject[info_]:=(Message[ResourceObject::exists];Throw[$Failed])/;!FreeQ[Lookup[info,"ResourceLocations",{}],_LocalObject]
*)
saveresourceobject[info_]:=saveresourceobject[fillresourceInfo[info]]/;!filledResourceQ[info]

saveresourceobject[info_]:=Block[{rtype=getResourceType[info],id},
	id=If[StringQ[rtype],
		loadResourceType[rtype];
		repositorysaveresourceobject[rtype,info]
		,
		cacheresourceinfo[info]["UUID"]
	];
	System`ResourceObject[id]
]

repositorysaveresourceobject[_,info0_] :=
If[KeyExistsQ[info0,"UUID"],
	  Module[ { info = info0},
	      info[ "RepositoryLocation" ] = localObject @ resourceCacheDirectory[ ];
	      cacheresourceinfo @ info;
	      id
	  ],
	  $Failed
]



End[] (* End Private Context *)

EndPackage[]
