(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {System`ResourceRegister}
BeginPackage["ResourceSystemClient`"]

System`ResourceRegister

ResourceSystemClient`ResourceUnregister

ResourceSystemClient`$CacheCloudRegistries

Begin["`Private`"] (* Begin Private Context *) 

System`ResourceRegister[args___]:=Catch[resourceRegister[args]]

Options[System`ResourceRegister]=Options[resourceRegister]=Options[resourceregister]={"StoreContent"->Automatic}

resourceRegister[ro_, opts:OptionsPattern[]]:=resourceRegister[ro, defaultResourceRegistries[],opts]


resourceRegister[ro_ResourceObject, locations_List,opts:OptionsPattern[]]:=With[{id=resourceObjectID[ro]},
	resourceregister[id, getResourceInfo[id], persistenceLocation/@locations, opts]
]

resourceRegister[id_String, locations_List,opts:OptionsPattern[]]:=resourceRegister[ResourceObject[id], locations, opts]/;uuidQ[id]

resourceRegister[expr_, locations_List,opts:OptionsPattern[]]:=resourceRegisterNonResource[expr, locations, opts]

resourceRegister[ro_, expr_,opts:OptionsPattern[]]:=resourceRegister[ro,{expr}, opts]

resourceregister[id_, info_Association, locations_, opts:OptionsPattern[]]:=With[{rtype=info["ResourceType"]},
	If[StringQ[rtype],
		loadResourceType[rtype];
		If[!FailureQ[repositoryResourceRegister[rtype, id, info, locations, Association[opts]]],
			ResourceObject[id]
			,
			$Failed
		]
		,
		$Failed
	]
]

defaultResourceRegistries[]:=With[{pb=$PersistenceBase},
	If[FreeQ[pb,"KernelSession"],
		Flatten[{"KernelSession",pb}],
		pb
	]
]

persistenceLocation[str_String]:=With[{pl=PersistenceLocation[str]},
	If[Head[pl]=!=PersistenceLocation,
		Message[ResourceRegister::badpl,str];
		Throw[$Failed]
		,
		pl
	]
]

persistenceLocation[pl_PersistenceLocation]:=pl
persistenceLocation[expr_]:=(Message[ResourceRegister::badpl,expr];Throw[$Failed])


repositoryResourceRegister[rtype_, id_, info_, locations_, as_]:=(
	persistResourceName[id,info["Name"], locations];
	If[Lookup[as,"StoreContent",True]=!=False,
		storeResource[rtype,id,info,#]&/@locations
	];
	registerResourceNameExtensions[rtype, id, info, locations,as]
	)

registerResourceNameExtensions[___]:=Null

$ResourceNamePersistentValuePrefix="ResourceNames/";
$ResourceObjectPersistentValuePrefix="ResourceObjects/";

encodeResourcePersistentObjectName[name_]:=$ResourceNamePersistentValuePrefix<>Hash[name,"Expression","HexString"]

persistResourceName[id_,name_String, locations___]:=
	With[{res=persistresourceName[id,encodeResourcePersistentObjectName[name],locations]},
		If[!FailureQ[res],
			res
			,
			Throw[$Failed]
		]
	]

persistresourceName[id_,name_,locations_]:=(PersistentValue[name,locations]=id)

storeResource[rtype_,id_,info_,pl_PersistenceLocation]:=With[{ptype=plType[pl]},
	storeResource[rtype,ptype,id,info,pl]
]

storeResource[rtype_,"Local",id_,info_,pl_]:=saveresourceObject[info]
storeResource[rtype_,"Cloud",id_,info_,pl_]:=With[{res=If[MemberQ[$loadedResources, id],
			clouddeployResourceObject[id, resourceInfo[id], None]
			,
			autoloadResource[info];
			If[MemberQ[$loadedResources, id],
				clouddeployResourceObject[id, resourceInfo[id], None]
				,
				Throw[$Failed]
			]
		]},
		If[Head[res]===CloudObject,
			res,
			$Failed
		]		
	]

storeResource[_,"KernelSession",_,_,_]:=Null
storeResource[rtype_,ptype_,id_,info_,pl_]:=With[{info1=Association[info,"Autoload"->True]},
	Block[{ResourceObject, res},
		res=(PersistentValue[$ResourceObjectPersistentValuePrefix<>id,ptype]=ResourceObject[
			info1
		]);
		If[!FailureQ[res],
			res
			,
			Throw[$Failed]
		]
	]
]
storeResource[___]:=Null

plType[HoldPattern[PersistenceLocation]["Cache", 
	{HoldPattern[PersistenceLocation][type_, ___], ___}, ___]] := type
plType[HoldPattern[PersistenceLocation]["Cache", 
	HoldPattern[PersistenceLocation][type_, ___], ___]] := type
plType[HoldPattern[PersistenceLocation][_, ___, 
   Hold[___, "DisplayName" -> type_, ___], ___]] := type
plType[HoldPattern[PersistenceLocation][type_, ___]] := type

Clear[checkRegistryLookup]

localRegistryLookup[type_,name_]:=checkRegistryLookup[type,name,{"Local"}]
cloudRegistryLookup[type_,name_]:=checkRegistryLookup[type,name,{"Cloud"}]

checkRegistryLookup[type_,names_List,rest___]:=Catch[
	checkRegistryAlternativeName[type,#,rest]&/@names;
	None,
	"checkRegistryLookup"
]

checkRegistryAlternativeName[type_,name_,rest___]:=With[{res=checkRegistryLookup[type,name,rest]},
	If[uuidQ[res],
		Throw[res,"checkRegistryLookup"]
	]
]

checkRegistryLookup[type_,name_String]:=checkRegistryLookup[type,name,$PersistencePath]
checkRegistryLookup[All,name_String,locations_]:=With[{uuid=registryLookup[name,locations]},
	If[uuidQ[uuid],
		With[{ro=Quiet[ResourceObject[uuid], All, {ResourceObject::updav,ResourceObject::updavb}]},
			If[localResourceQ[uuid]||resourceObjectQ[ro],
				If[verifyResourceRegistryName[ro,name,locations],
					uuid,
					None
				],
				None
			]
		]
		,
		None		
	]
]


checkRegistryLookup[types_List,name_String,locations_]:=With[{uuid=registryLookup[name,locations]},
	If[uuidQ[uuid],
		With[{ro=Quiet[ResourceObject[uuid], All, {ResourceObject::updav,ResourceObject::updavb}]},
			If[MemberQ[types,Quiet[ro["ResourceType"], All]],
				If[verifyResourceRegistryName[ro,name,locations],
						uuid,
						None
					]
				,
				None
			]
		]
		,
		None		
	]
]

verifyResourceRegistryName[ro_,name_String,_]:=True/;Quiet[ro["Name"]===name]
verifyResourceRegistryName[ro_,name_String,_]:=True/;Quiet[ro["ShortName"]===name]
verifyResourceRegistryName[ro_,name_String, locations_]:=(unregisterResource[name, locations];False)

checkRegistryLookup[type_,name_String,locations_]:=checkRegistryLookup[{type},name,locations]

registryLookup[name_,locations_List]:=registrylookup[name,persistenceLocation/@locations]

$ResourceNameRegistryUsed=False;

registrylookup[name_,locations_List]:=With[
	{uuid=persistentValueFast[encodeResourcePersistentObjectName[name],locations]},
	If[uuidQ[uuid],
		$ResourceNameRegistryUsed=True;
		uuid
		,
		None		
	]
]

persistentValueFast[name_,locations_]:=Catch[
	With[{val = persistentValueRO[name, #]}, 
	    If[!MissingQ[val],If[
	    	FreeQ[#,"KernelSession"],
	    	persistresourceName[val,name,{"KernelSession"}]
	    ];Throw[val, "merged"], val]
	    ] & /@ locations, 
	    	"merged"]

persistentValueRO[name_,HoldPattern[PersistenceLocation]["Cloud",___]]:=(
		printTempOnce[$progressID,Dynamic[$ProgressIndicatorContent]];
		PersistentValue[name,"Cloud"]
		)
		
persistentValueRO[name_,location_]:=PersistentValue[name,location]

ResourceSystemClient`ResourceUnregister[args___]:=Catch[resourceUnregister[args]]

resourceUnregister[ro_]:=resourceUnregister[ro, defaultResourceRegistries[]]

resourceUnregister[ro_ResourceObject, locations_List]:=With[{id=resourceObjectID[ro]},
	resourceunregister[id, getResourceInfo[id], persistenceLocation/@locations]
]

resourceUnregister[id_String, locations_List,opts:OptionsPattern[]]:=resourceUnregister[ResourceObject[id], locations, opts]/;uuidQ[id]

resourceUnregister[name_String,locations_List,opts:OptionsPattern[]]:=With[{ro=Quiet[ResourceObject[name]]},
	unRegisterResource[name,locations];
	localResourceNameMap=KeyDrop[localResourceNameMap,name];
	If[resourceObjectQ[ro],
		resourceUnregister[ro,locations,opts],
		$Failed
	]
]

resourceUnregister[expr_, l_List]:=(Message[ResourceRegister::noro,expr];$Failed)

resourceUnregister[ro_, expr_]:=resourceUnregister[ro,{expr}]

resourceunregister[id_, info_Association, locations_]:=With[{rtype=info["ResourceType"]},
	If[StringQ[rtype],
		loadResourceType[rtype];
		If[!FailureQ[repositoryResourceUnregister[rtype, id, info, locations]],
			ResourceObject[id]
			,
			$Failed
		]
		,
		$Failed
	]
]

repositoryResourceUnregister[rtype_, id_, info_, locations_]:=(
	unRegisterResource[info["Name"],locations];
	unregisterResourceNameExtensions[rtype, id, info, locations]
	)

unregisterResourceNameExtensions[___]:=Null

$unregisterPath:=$PersistencePath;

unRegisterResource[name_]:=unRegisterResource[name,$unregisterPath]
unRegisterResource[name_,locations_List]:=unregisterResource[name,persistenceLocation/@locations]
unRegisterResource[name_,location_]:=unRegisterResource[name,{location}]

unregisterResource[name_,locations_List]:=Quiet[
	Remove[PersistentValue[encodeResourcePersistentObjectName[name],locations]],
		{DeleteObject::nso}]

resourceUnregister[___]:=$Failed


resourceRegisterNonResource[___]:=$Failed


End[] (* End Private Context *)

EndPackage[]
SetAttributes[{ResourceRegister},
   {ReadProtected, Protected}
];