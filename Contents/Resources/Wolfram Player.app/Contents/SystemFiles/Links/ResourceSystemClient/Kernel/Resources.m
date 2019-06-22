(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

Quiet[If[Length[PacletManager`PacletFind["InstalledResourceObjects"]]>0,
	Needs["InstalledResourceObjects`"];
]];

$localResourcesCompiled=False;
$localResources:=($localResourcesCompiled=True;$localResources=importlocalResourceIDs[])
$cloudResources:=($cloudResources=importCloudResourceIDs[])/;$CloudConnected&&$importCloudResources
$cloudResources:={}

$importCloudResources=False;

$myResourceSystemResources={};

$loadedResources={};

localResourceNameMap=Association[];

importLocalResourceIDs[]:=($localResources=importlocalResourceIDs[])

importlocalResourceIDs[]:=With[{files=FileNames["metadata",resourceCacheDirectory[],3]},
    FileNameTake[#,{-2}]&/@files
    ]

localResourceQ[id_]:=MemberQ[$localResources,id]/;$localResourcesCompiled
localResourceQ[id_]:=FileExistsQ[resourceInfoFile[id]]

importCloudResourceIDs[]:=(importLocalResourceIDs[];
    $cloudResources=$localResources;
    $localResources
)/;$CloudEvaluation

importCloudResourceIDs[]:=With[{
	res=CloudEvaluate[
		With[{files=FileNames["metadata",$CloudResourceBase,3]},
	    FileNameTake[#,{-2}]&/@files
	    ]]},
	If[ListQ[res],
		$cloudResources=res
		,
		{}
	]
]

importSystemResourceIDs[]:=With[{res=myResourceSystemResources[False]},
	If[ListQ[res],
		$myResourceSystemResources=res
	]
]

myResourceSystemResources[includeMetadataQ_:True]:=Block[{res},
    res=apifun["UserResources",{"IncludeMetadata"->includeMetadataQ}, ResourceSearch];
    Lookup[res,"Resources",{}]
]/;requestBaseConnected[]

myResourceSystemResources[args___]:=(
    cloudConnect[ResourceSearch];
    myResourceSystemResources[args]
    )

importLocalResourceInfo[]:=With[{ids=importLocalResourceIDs[]},
      Select[getResourceInfo/@ids,AssociationQ]
    ]


importCloudResourceInfo[]:=importLocalResourceInfo[]/;$CloudEvaluation
importCloudResourceInfo[]:=With[{
	res=CloudEvaluate[
		Block[{files=FileNames["metadata",$CloudResourceBase,3], ids,
			resourceCacheDirectory, localObject=Identity},
			resourceCacheDirectory[]=$CloudResourceBase;
	   		ids=FileNameTake[#,{-2}]&/@files;
	    	getResourceInfo/@ids
	    ]]},
	If[ListQ[res],
		res		
		,
		{}
	]
]
    
(* Syncing resource info *)

$ResourceSystemSyncedQ=True/;$CloudEvaluation
$ResourceSystemSyncedQ=False

syncResources[]:=Block[{temp},
	$ProgressIndicatorContent=progressPanel["Retrieving acquired and deployed resources\[Ellipsis]"];
	temp=printTemporaryFrontEnd[Dynamic[$ProgressIndicatorContent]];
    syncWithSystemResources[];
    syncWithCloudResources[];
	clearTempPrint[temp];
    $ResourceSystemSyncedQ=True
]

syncWithSystemResources[]:=Block[{marketplaceresourceInfo, allinfo=Association[], ids={},cloudresouceInfo},
    marketplaceresourceInfo=myResourceSystemResources[True];
    marketplaceresourceInfo=Select[marketplaceresourceInfo,AssociationQ];
    marketplaceresourceInfo=fillResourceMetadata[#,Association["RepositoryLocation"->URL[$resourceSystemRequestBase],"MyAccount"->True]]&/@marketplaceresourceInfo;
    $myResourceSystemResources=Select[Lookup[marketplaceresourceInfo,"UUID"],StringQ];
    cacheresourceInfo[Select[marketplaceresourceInfo,AssociationQ], False]
]

syncWithCloudResources[]:=Block[{cloudresources,$importCloudResources=True},
    cloudresources=CloudEvaluate[importLocalResourceInfo[]];
    cacheresourceInfo[cloudresources];
    $cloudResources=With[{localresources=importLocalResourceInfo[]},
    CloudEvaluate[syncwithCloudResources[localresources]]
    ];
    importLocalResourceIDs[]
]/;$CloudConnected&&(!$CloudEvaluation)

syncWithCloudResources[]:=Null

syncwithCloudResources[localresources_]:=(cacheresourceInfo[localresources];
        importlocalResourceIDs[])

(* loading resources *)
loadResource[id_String]:=resourceInfo[id]/;MemberQ[$loadedResources,id]
loadResource[id_String]:=With[{info=getResourceInfo[id]},
	If[AssociationQ[info],
        loadresource[id, info];
		info
		,
		$Failed
	]]/;localResourceQ[id]

loadResource[name_String]:=loadResource[
	Lookup[localResourceNameMap,name,$Failed]]/;KeyExistsQ[localResourceNameMap,name]
	
loadResource[id_String]:=With[{
	info=CloudEvaluate[
		Block[{resourceCacheDirectory, localObject=Identity},
			resourceCacheDirectory[]=$CloudResourceBase;
			getResourceInfo[id]
		]
		]},
    If[AssociationQ[info],
        cacheresourceinfo[info]
        ,
        $Failed
    ]]/;MemberQ[$cloudResources,id]
    
loadResource[id_String]:=With[{info=importresourceInfo[id, False]},
    If[AssociationQ[info],
        info
        ,
        $Failed
    ]]/;MemberQ[$myResourceSystemResources,id]
    
loadResource[uuid_String]:=loadResourceUUID[uuid]/;uuidQ[uuid]
    
loadResourceUUID[uuid_]:=loadResourceUUID[uuid,
	persistentValueFast[$ResourceObjectPersistentValuePrefix<>uuid,$PersistencePath]]

loadResourceUUID[uuid_,registered_]:=registered[All]/;resourceObjectQ[registered]

loadResourceUUID[uuid_,_]:=With[{info=importresourceInfo[uuid, False]},
    If[AssociationQ[info],
        info
        ,
        $Failed
    ]]
    
loadResource[name_String]:=With[{info=importresourceInfo[name, False]},
    If[Quiet[KeyExistsQ[info,"UUID"]],
        info
        ,
        $Failed
    ]]

loadResource[___]:=$Failed

loadresource[info_]:=loadresource[Lookup[info,"UUID"],info]
loadresource[id_String,info_Association]:=(
    resourceInfo[id]=checkResourceInfo[info,True];
	If[marketplacebasedResourceQ[info],
    	AppendTo[localResourceNameMap,info["Name"]->id];
	];
    $loadedResources=DeleteDuplicates[Append[$loadedResources,id]];
)
loadresource[__]:=$Failed

clearresource[id_, name_]:=(
	If[KeyExistsQ[$cachedContentElements,id],
		clearContentCache[id,resourceInfo[id]]
	];
    Quiet[resourceInfo[id]=.];
    If[StringQ[name],
    	localResourceNameMap=KeyDrop[localResourceNameMap,name];
    	unRegisterResource[name]
    ];
    $cachedContentElements[id]=.;
    $loadedResources=DeleteCases[$loadedResources,id];
)

clearContentCache[id_,as_Association]:=clearContentCache[info["ResourceType"],id,as]

clearContentCache[___]:=Null

$issueAcquisitionWarnings=True;
checkResourceInfo[info0_, checkRequired_]:=Block[{rsbase=resourcerepositoryBase[info0], updated,temp, info=info0},
	If[KeyExistsQ[info,"WolframLanguageVersionRequired"],
		checkResourceWLVersion[info["WolframLanguageVersionRequired"]]
	];
	If[$issueAcquisitionWarnings&&TrueQ[mustAcquireResourceQ[info["UUID"],info]],
		summaryPurchaseStatus[info0["UUID"]]=summaryMustPurchase[info0];
		Message[ResourceObject::unacq,info["Name"]]
	];
	info=resourceRequiredUpdateCheck[info, rsbase];
	Block[{publicResourceInfoUpdating},
		publicResourceInfoUpdating[__]=False;
		publicResourceInfoUpdating["Versions",rsbase]=True;
		If[KeyExistsQ[resourceVersionsAvailable[rsbase],Lookup[info,"UUID"]],
			resourceUpdateCheck[info]
			,
			info
		]
	]
]


checkResourceWLVersion[version_String]:=If[newerversionQ[{
	StringReplace[version, {"+" -> "", "-" -> ""}], ToString[$VersionNumber]}]
	,
	Message[ResourceObject::version,versionnumberstring[ToString[$VersionNumber]], version]
]
checkResourceWLVersion[version_?NumberQ]:=checkResourceWLVersion[ToString[version]]

myResources[args___]:=If[$ResourceSystemSyncedQ,
        myresources[args]
        ,
        syncResources[args];
        myresources[args]
    ]

myresources[___]:=System`ResourceObject/@$localResources

alternativeResourceNames[__]:={}

findResourceObject[type_, name_, opts___]:=Block[{res, temp, id,requestbase,
	alternativeNames=alternativeResourceNames[type,name], $progressID=Unique[]},
	$ProgressIndicatorContent=progressPanel["Finding resource\[Ellipsis]"];
	res=lookupResourceNameMap[type, name,alternativeNames];
	If[uuidQ[res],
		clearTempPrint[temp];
		Return[ResourceObject[res,opts]]
	];
	If[StringContainsQ[name,"://"] || StringStartsQ[name, "user:"],
		temp=printTempOnce[$progressID,Dynamic[$ProgressIndicatorContent]];
		res=findResourceObjectByURL[name];
		clearTempPrint[temp];
		If[res=!=$Failed,
			Return[res]
		]
	];
	res=If[Length[alternativeNames]>0,
		checkRegistryLookup[type, Flatten[{name,alternativeNames}]],
		checkRegistryLookup[type, name]
	];
	If[uuidQ[res],
		clearTempPrint[temp];
		Return[ResourceObject[res,opts]]
	];
	temp=printTempOnce[$progressID,Dynamic[$ProgressIndicatorContent]];
	If[$CloudConnected&&FreeQ[$PersistencePath,"Cloud"],
		res=If[Length[alternativeNames]>0,
			checkRegistryLookup[type, Flatten[{name,alternativeNames}],{"Cloud"}],
			checkRegistryLookup[type, name,{"Cloud"}]
		];
		If[uuidQ[res],
			clearTempPrint[temp];
			Return[ResourceObject[res,opts]]
		]
	];
	requestbase=OptionValue[ResourceObject,{opts},System`ResourceSystemBase];
	res=If[requestbase=!=System`$ResourceSystemBase,
		resourceAcquire[name, False,requestbase, Association[resourceAquireTypeOption[type],opts]]
		,
		resourceAcquire[name, False, Automatic, Association[resourceAquireTypeOption[type],opts]]
	];
	clearTempPrint[temp];
	res
]

resourceAquireTypeOption[types:{_String...}]:=With[{str=StringRiffle[types,","]},
	Association["ResourceType"->str]
]
resourceAquireTypeOption[type_String]:=Association["ResourceType"->type]
resourceAquireTypeOption[___]:=Association[]


findResourceObjectByURL[co:HoldPattern[_CloudObject]]:=With[{res=Get[co]},
	If[Head[res]===ResourceObject,
		autoloadResource[First[res]];
		res
		,
		$Failed
	]
]

findResourceObjectByURL[ url_String ? shortURLQ ] :=
	findResourceObjectByURL @ URLExpand @ url;

findResourceObjectByURL[url_URL]:=findResourceObjectByURL[First[url]]
findResourceObjectByURL[url_String]:=With[{as=URLParse[url]},
	If[AssociationQ[as],
		repositoryResourceByURL[as["Domain"],as["Path"]]
	]
]/;repositoryURLQ[url]

findResourceObjectByURL[url_String]:=findResourceObjectByURL[CloudObject[url]]/;cloudURLQ[url]

findResourceObjectByURL[url_String]:=With[{res=Import[url, "Plaintext"]},
	If[StringContainsQ[res,"ResourceObject"],
		checkResourceString[res]
		,
		$Failed
	]
]

shortURLQ // ClearAll;
shortURLQ[ url_String ] := Quiet[ URLParse[ url, "Domain" ] === "wolfr.am" ];

checkResourceString[str_]:=With[{res=ToExpression[str]},
	If[Head[res]===ResourceObject,
		autoloadResource[First[res]];
		res,
		$Failed]	
]

repositoryResourceByURL[_,{___,"resources"|"Resources",name_}]:=findResourceObject[All, name]/;StringFreeQ[name,"://"]
repositoryResourceByURL[___]:=$Failed

repositoryURLQ[url_]:=StringMatchQ[url,"*datarepository*wolfram*.com*"~~("resources"|"Resources")~~"*"]||
	StringMatchQ[url,"*resources*wolfram*.com*/*"]

cloudURLQ[url_]:=True/;(URLParse[url]["Domain"]===URLParse[$CloudBase]["Domain"])&&StringContainsQ[url,"object"]
cloudURLQ[url_]:=StringContainsQ[url,"wolframcloud"] || StringStartsQ[url, "user:"]

lookupResourceNameMap[All, name_,_]:=Lookup[localResourceNameMap,name,$Failed]/;KeyExistsQ[localResourceNameMap,name]

lookupResourceNameMap[type_, name_,{}]:=lookupResourceNameMap[type, name]
lookupResourceNameMap[type_, name_,alternatives_List]:=
	Catch[
		checklookupResourceNameMap[type,#]&/@Flatten[{name,alternatives}];
		$Failed
		,
		"lookupResourceNameMap"
	]

checklookupResourceNameMap[type_,name_]:=With[{res=lookupResourceNameMap[type,name]},
	If[StringQ[res],Throw[res,"lookupResourceNameMap"]]
]

lookupResourceNameMap[type_, name_]:=With[{id=Lookup[localResourceNameMap,name,$Failed]},
	If[Quiet[MemberQ[Flatten[{type}],Lookup[getResourceInfo[id],"ResourceType"]]],
		id,
		$Failed
	]
	]/;KeyExistsQ[localResourceNameMap,name]

lookupResourceNameMap[__]:=$Failed	

cacheCloudResourcesAsynchronous[]:=RunScheduledTask[
 Block[{$importCloudResources=True,timingLog}, 
  	$cloudResources;
  	RemoveScheduledTask[$ScheduledTask]
  ], {5}]/;$CloudConnected&&(!$CloudEvaluation)&&ResourceSystemClient`$AsyncronousResourceInformationUpdates




End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];