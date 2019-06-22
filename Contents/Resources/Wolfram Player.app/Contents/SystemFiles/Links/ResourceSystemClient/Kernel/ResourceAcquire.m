(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {System`ResourceAcquire,System`ResourceUpdate,System`ResourceRemove}

BeginPackage["ResourceSystemClient`"]

System`ResourceAcquire
System`ResourceRemove
System`ResourceUpdate

ResourceSystemClient`$ResourceSystemAutoUpdate

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`$ResourceSystemAutoUpdate:=(
	ResourceSystemClient`$ResourceSystemAutoUpdate=If[TrueQ[$CloudEvaluation],True,Automatic]
	)
(*** ResourceAcquire ***)

System`ResourceAcquire[args___]:=
	Catch[iResourceAcquire[args]]

Options[System`ResourceAcquire]={System`ResourceSystemBase:>Automatic, "AddToAccount"->True,
	"Version"->Automatic,"WolframLanguageVersion"->Automatic}

iResourceAcquire[ro_,as_Association]:=Block[{$issueAcquisitionWarnings=False, res, info},
	res=resourceAcquire[ro,as];
	If[resourceObjectQ[res],
		info=resourceInfo[resourceObjectID[res]];
		If[mustAcquireResourceQ[info["UUID"],info],
			launchAcquisition[info,as]
		];		
	];
	res
]

iResourceAcquire[ro_,opts:OptionsPattern[System`ResourceAcquire]]:=iResourceAcquire[ro,Association[opts]]

iResourceAcquire[___]:=$Failed

resourceAcquire[resource:rpat, rest___]:=resourceAcquire[resourceObjectID[resource],rest]

resourceAcquire[id_,as_Association]:=(loadResource[id];resourceAcquire[id, as])/;MemberQ[$localResources,id]&&!MemberQ[$loadedResources,id]
resourceAcquire[id_,as_Association]:=If[optionMismatchQ[resourceInfo[id], as],
	Block[{$forceCacheUpdate=True},
		noUpdate[resourceacquire[id,as]]
	]
	,
	System`ResourceObject[id]
]/;MemberQ[$localResources, id]

resourceAcquire[id_,as_Association]:=With[{info=loadResource[id]},
	If[AssociationQ[info],
        cacheresourceinfo[info];
		resourceAcquire[id, as]
		,
		$Failed
	]]/;MemberQ[$cloudResources,id]||MemberQ[$myResourceSystemResources,id]

resourceAcquire[str_String,as_Association]:=resourceacquire[str,as]

resourceAcquire[str_,addToAccount_:True]:=resourceAcquire[str,Association["AddToAccount"->addToAccount,System`ResourceSystemBase->Automatic]]
resourceAcquire[str_,addToAccount_, requestbase_]:=resourceAcquire[str,Association["AddToAccount"->addToAccount,System`ResourceSystemBase->requestbase]]
resourceAcquire[str_,addToAccount_, requestbase_, as_]:=resourceAcquire[str,Association[as,"AddToAccount"->addToAccount,System`ResourceSystemBase->requestbase]]

resourceacquire[str_,addToAccount_,requestbase_]:=resourceAcquire[str,Association["AddToAccount"->addToAccount,System`ResourceSystemBase->requestbase]]

resourceacquire[str_,as_Association]:=Block[{info, id},
	info=importresourceInfo[str,
		Lookup[as,"AddToAccount",True],
		Lookup[as,System`ResourceSystemBase,Automatic],
		KeyDrop[as,{"AddToAccount",System`ResourceSystemBase}]];
	id=info["UUID"];
	persistResourceName[id,info["Name"], {"KernelSession","Local"}];
	System`ResourceObject[id, Normal@KeyTake[as,{System`ResourceSystemBase,"Version","WolframLanguageVersion"}]]
]

resourceAcquire[___]:=Throw[$Failed]
resourceacquire[___]:=Throw[$Failed]

importresourceInfo[str_,addToAccount_]:=importresourceInfo[str,addToAccount,Automatic]
importresourceInfo[str_,addToAccount_,requestbase_]:=importresourceInfo[str,addToAccount,requestbase,Association[]]

importresourceInfo[str_,addToAccount_,requestbase_, as_]:=With[{info=importResourceInfo1[str,addToAccount,requestbase,as]},
	noUpdate[cacheresourceinfo[info]]
]

importResourceInfo1[str_,addToAccount_,requestbase_, as_]:=Block[{params, info, resourcebase=resourceSystemBase[requestbase]},
	params=Flatten[{If[uuidQ[str],
        "UUID"->str,
        "Name"->str
    ],"RecordUserAccess"->addToAccount,
    	Normal[Select[as,StringQ]],
    	"Elements"->"True","ContentElementFunctions"->"True"}];
	info=importresourceinfo[params, resourcebase];
	If[Quiet[TrueQ[KeyExistsQ[info,"UUID"]]],
		fillResourceMetadata[info, Association["RepositoryLocation"->URL[resourcebase],"MyAccount"->addToAccount]]
		,
		Message[ResourceObject::notf];
		Throw[$Failed]
	]
]


importresourceinfo[params_, resourcebase_]:=With[{res=apifun["AcquireResource",params, System`ResourceAcquire,resourcebase]},
	standardizeResourceInfo[res]
]


myAccountQ[id_]:=myaccountQ[resourceinfo[id]]
myaccountQ[as_Association]:=Lookup[as,"MyAccount",False]
myaccountQ[_]:=False




(*** ResourceRemove ***)

System`ResourceRemove[args___]:=Catch[resourceRemove[args]]

resourceRemove[resource:rpat]:=resourceRemove[resourceObjectID[resource]]

resourceRemove[id_]:=Block[{info},
	info=resourceInfo[id];
	If[AssociationQ[info],
		deleteresourcecache[info];
		If[myaccountQ[info]&&$CloudConnected,
			info=removeResourceFromAccount[If[uuidQ[id],
		        {"UUID"->id},
		        {"Name"->id}
		    ],resourcerepositoryBase[info]]
		];
	    If[KeyExistsQ[info,"UUID"],
			id
			,
			Throw[$Failed]
		]
		,
		If[MemberQ[$loadedResources,id]||MemberQ[$localResources,id],
			deleteresourcecache[id, Missing[]];
			id
			,
			Message[ResourceRemove::unkro, id];
			$Failed
		]
	]
	
]

resourceRemove[___]:=$Failed

removeResourceFromAccount[params_, resourcebase_]:=With[{res=apifun["RemoveResource",params, System`ResourceAcquire,resourcebase]},
	res
]

deleteResourceObject[resource:rpat]:=deleteResourceObject[resourceObjectID[resource],resource]

deleteResourceObject[id_String, resource_]:=deleteResourceObject[id, resource, resourceInfo[id]]/;MemberQ[$loadedResources,id]
deleteResourceObject[id_String, resource_]:=deleteResourceObject[id, resource, getResourceInfo[id]]/;MemberQ[$localResources,id]
deleteResourceObject[id_String, resource_]:=deleteResourceObject[id, resource, None]

deleteResourceObject[id_String,resource_, info_Association]:=(resourceRemove[id];Null)

deleteResourceObject[id_String,resource_, None]:=(Message[DeleteObject::nso, resource];$Failed)

System`ResourceUpdate[args___]:=Catch[resourceUpdate0[args]]

Options[System`ResourceUpdate]=Options[resourceUpdate0]=Options[resourceUpdate]={"Version"->Automatic,
	System`ResourceSystemBase->Automatic,"WolframLanguageVersion"->Automatic};

resourceUpdate0[args___]:=noUpdate[resourceUpdate[args]]

resourceUpdate[All,opts:OptionsPattern[]]:=updateAllLocalResources[All,Association[opts]]
resourceUpdate[HoldPattern[Rule][rtype_,All],opts:OptionsPattern[]]:=updateAllLocalResources[rtype,Association[opts]]

resourceUpdate[resource_ResourceObject, rest___]:=Block[{ro=resourceUpdate[resourceObjectID[resource],rest], id},
	If[resourceObjectQ[ro],
		System`ResourceObject[ro["UUID"]]
		,
		$Failed
	]
]

resourceUpdate[id_String,rest___]:=Block[{info=resourceUpdate[getResourceInfo[id], rest]},
	If[AssociationQ[info],
		System`ResourceObject[info["UUID"]]
		,
		$Failed
	]
]/;uuidQ[id]

resourceUpdate[name_String,rest___]:=With[{ro=ResourceObject[name,rest]},
	If[resourceObjectQ[ro],
		resourceUpdate[ro,rest]
		,
		$Failed
	]
]


resourceUpdate[info_Association, rest___]:=resourceUpdate[resourcerepositoryBase[info],Lookup[info,"UUID"], info, rest]/;marketplacebasedResourceQ[info]

resourceUpdate[_Association]:=(Message[ResourceUpdate::notressys];$Failed)

resourceUpdate[rsbase_String,id_String, info_Association,opts:OptionsPattern[]]:=Block[{rsVersion,localVersion, res},
	If[
		newerversionQ[{Lookup[resourceVersionsAvailable[rsbase],id,"0.0"],Lookup[info,"Version","0.0"]}]
		||StringQ[OptionValue["Version"]],
		res=resourceupdate[rsbase,id, Association[opts]];
		If[!FailureQ[res],
			res
			,
			info
		]
		,
		Message[ResourceObject::newest,id];info
	]
]
	
resourceupdate[rsbase_,id_, as_]:=Block[{info},
	info=importResourceInfo1[id,False,rsbase,as];
	If[AssociationQ[info],
		deleteresourcecache[resourceInfo[id]];
		cacheresourceinfo[info]
		,
		$Failed
	]
]

resourceUpdate[___]:=$Failed

$resourceUpdating=False;

resourceUpdateCheck[info_]:=resourceUpdateCheck[Lookup[info,"AutoUpdate", Automatic],info]

resourceUpdateCheck[False,info_]:=info
resourceUpdateCheck[False,info_,_]:=info
resourceUpdateCheck[autoUpdate_,info_]:=resourceUpdateCheck[autoUpdate,info, resourcerepositoryBase[info]]
resourceUpdateCheck[autoUpdate_,info_, rsbase_]:=With[{id=Lookup[info,"UUID"]},
	If[!$resourceUpdating&&KeyExistsQ[resourceVersionsAvailable[rsbase],id],
		resourceupdateCheck[autoUpdate,id,info, rsbase,resourceVersionsAvailable[rsbase][id]]
		,
		info
	]
]

resourceupdateCheck[autoUpdate_,id_,info_, rsbase_, availableversion_]:=(
	resourceupdatecheck[autoUpdate,id,info, rsbase]
	)/;newerversionQ[{availableversion,Lookup[info,"Version","0.0"]}]&&(!$forceCacheUpdate)
resourceupdateCheck[_,_,info_, _, _]:=info

resourceupdatecheck[True,id_,info_, rsbase_]:=Block[{$resourceUpdating=True},resourceUpdateWithProgress[id,info, rsbase]]
resourceupdatecheck[Automatic,id_,info_, rsbase_]:=resourceupdatecheck[Automatic,ResourceSystemClient`$ResourceSystemAutoUpdate,id,info, rsbase]
resourceupdatecheck[_,_,info_, _]:=(resourceUpdateMessage[info];info)

resourceupdatecheck[Automatic,True,id_,info_, rsbase_]:=Block[{$resourceUpdating=True},resourceUpdateWithProgress[id,info, rsbase]]
resourceupdatecheck[Automatic,False,id_,info_, rsbase_]:=(resourceUpdateMessage[info];info)
resourceupdatecheck[Automatic,Automatic,id_,info_, rsbase_]:=If[TrueQ[autoUpdateResourceQ[info]],
	resourceupdatecheck[Automatic,True,id,info, rsbase],
	resourceupdatecheck[Automatic,False,id,info, rsbase]
]

resourceUpdateMessage[info_]:=With[{name=Lookup[info,"Name",""],uuid=Lookup[info,"UUID",""]},
	If[uuidQ[uuid],
		Message[ResourceObject::updavb,name,Button["Click here to update", ResourceUpdate[uuid], ContentPadding -> False]],
		Message[ResourceObject::updav,name]
	];
	info
]

autoUpdateResourceQ[info_]:=With[{rtype=info["ResourceType"]},
	loadResourceType[rtype];
	autoUpdateResourceQ[rtype,info]
]
autoUpdateResourceQ[_,_]:=False

resourceRequiredUpdateCheck[info_]:=resourceRequiredUpdateCheck[info, resourcerepositoryBase[info]]
resourceRequiredUpdateCheck[info_, rsbase_]:=With[{id=Lookup[info,"UUID"]},
	If[KeyExistsQ[resourceVersionsRequired[rsbase],id],
		resourcerequiredUpdateCheck[id,info, rsbase,resourceVersionsRequired[rsbase][id]]
		,
		info
	]
]

resourcerequiredUpdateCheck[id_,info_, rsbase_, requiredversion_]:=resourcerequiredupdateCheck[id,info, rsbase]/;newerversionQ[{requiredversion,Lookup[info,"Version","0.0"]}]
resourcerequiredUpdateCheck[_,info_, _, _]:=info

resourcerequiredupdateCheck[id_,info_, rsbase_]:=resourceUpdateWithProgress[id,info, rsbase]/;ResourceSystemClient`$ResourceSystemAutoUpdate=!=False
resourcerequiredupdateCheck[id_,info_,_]:=(Message[ResourceObject::requpdate,id];info)

resourceUpdateWithProgress[id_,info_, rsbase_]:=Block[{temp, updated},
	temp=installPrintTemp[];
	updated=resourceUpdate[info];
	clearTempPrint[temp];
	Sow[updated,"UpdatedResourceInfo"]
]

(* update all *)
updateAllLocalResources[All,as_Association]:=updateAllLocalResources[All,Lookup[as,System`ResourceSystemBase,$ResourceSystemBase],as]
updateAllLocalResources[rtype_,Automatic,as_]:=updateAllLocalResources[rtype,$ResourceSystemBase,as]
updateAllLocalResources[rtype_,rsbase_String,as_]:=With[{infos=localResourcesInfoByTypeAndBase[rtype,rsbase]},
	publicResourceInformation["Versions",rsbase, True];
	With[{ids=Flatten@Last@Reap[resourceUpdateCheck[True,#, rsbase]&/@infos,"UpdatedResourceInfo"]},
		If[Length[ids]>0,
			ResourceObject/@Select[Lookup[ids,"UUID"],uuidQ],
			{}
		]
	]
]

updateAllLocalResources[___]:={};

localResourcesInfoByTypeAndBase[rtypes_,rbase_String]:=filterInfoByTypeAndBase[getResourceInfo[#],rtypes,rbase]&/@$localResources

filterInfoByTypeAndBase[as_Association,rtype_String,rbase_String]:=as/;(as["RepositoryLocation"]===URL[rbase])&&(as["ResourceType"]===rtype)
filterInfoByTypeAndBase[as_Association,All,rbase_String]:=as/;(as["RepositoryLocation"]===URL[rbase])
filterInfoByTypeAndBase[as_Association,rtype:{_String...},rbase_String]:=as/;(as["RepositoryLocation"]===URL[rbase])&&(MemberQ[rtype,as["ResourceType"]])
filterInfoByTypeAndBase[___]:=Nothing

$UseResourceAcquireDialog=False;
$UseServiceCreditDialog=True;

launchAcquisition[info_,as_]:=If[
	recheckAcquisition[info],
	launchacquisition[info,as],
	setResourceAcquired[info["UUID"]]
]

launchacquisition[info_,as_]:=createResourceAcquireDialog[info,as]/;$UseResourceAcquireDialog
launchacquisition[info_,as_]:=(resourceAcquireSend[info["UUID"], purchasingLocation[info]] &)

recheckAcquisition[info_Association]:=recheckAcquisition[info["UUID"],resourcerepositoryBase[info],info]

recheckAcquisition[id_String,rsbase_,info_]:=With[{
	res=apifun["AcquireResource",{"UUID"->id,"RecordUserAccess"->"False",
    	"Elements"->"False","ContentElementFunctions"->"False"}, System`ResourceAcquire,rsbase]},
    TrueQ[Lookup[res,"MustAcquire",False]]	
]
recheckAcquisition[___]:=False

createResourceAcquireDialog[info_,as_]:=Block[{CloudObject,IntegratedServices`Private`$windowtitle="Acquire a Resource"},
   IntegratedServices`RemoteServiceExecute;
   IntegratedServices`Private`CreateIntegratedServicesDialog[
    "Acquire a Resource", 
    "Acquire "<>dialogResourceName[info["Name"]]<>" from the Wolfram Marketplace to start using it.", 
    "Acquire", 
    (resourceAcquireSend[info["UUID"], purchasingLocation[info]] &), 
    (SystemOpen[learnMoreLocation[info]] &)]
   ]

dialogResourceName[str_String]:=ToString[str,InputForm]
dialogResourceName[_]:="the resource"

resourceAcquireSend[id_, url_]:=(
	SystemOpen[url];
	createResourceChannel[id];
	DialogReturn["Sent"]
)

serviceCreditWarning[rtype_]:=serviceCreditWarning[rtype,Automatic, Automatic,$CloudUserUUID]

serviceCreditWarning[rtype_,info_, as_,None]:=With[{res=CloudConnect[]},
	If[StringQ[$CloudUserUUID],
		serviceCreditWarning[rtype,info,as,$CloudUserUUID],
		res
	]	
]

serviceCreditWarning[rtype_,info_,as_, user_String]:=If[serviceCreditWarned[rtype,user],
	True,
	createResourceServiceCreditDialog[rtype,info,as,user]
]

createResourceServiceCreditDialog[rtype_,info_,as_,user_]:=Block[{IntegratedServices`Private`$windowtitle="Resource Service Credits"},
   IntegratedServices`RemoteServiceExecute;
   IntegratedServices`Private`CreateIntegratedServicesDialog[
   	resourceIcon[rtype], "Using this resource requires service credits.", "Continue",
   	(	
   		storeServiceCreditApproval[rtype,user];
   		serviceCreditWarned[rtype,user]=True;
   		DialogReturn[True]
   	)&, 
   	SystemOpen[ResourceSystemClient`Private`$ServiceCreditInformationURL]&]
]


serviceCreditWarned[rtype_String,user_String]:=With[{res=apifun["CheckServiceCreditConsentStatus",{"ResourceType"->rtype},ResourceObject]},
	If[KeyExistsQ[res,"ConsentStatus"],
		If[TrueQ[res["ConsentStatus"]]&&res["UserUUID"]===user,
			serviceCreditWarned[rtype,user]=True,
			False			
		],
		False
	]
]

storeServiceCreditApproval[rtype_,user_]:=apifun["ApproveServiceCredits",{"ResourceType"->rtype},ResourceObject]

serviceCreditWarned[__]:=False

$ResourceAcquireChannelName="ResourceAcquisitionChannel";

$DefaultResourceAcquistionURL="https://fakebilling.wolfram.com/fakepath";

learnMoreLocation[_]:="https://resources.wolframcloud.com/learn-more"
purchasingLocation[info_Association]:=Lookup[info,"AcquisitionURL",purchasinglocation[info]]
purchasinglocation[info_Association]:=purchasinglocation[info["RepositoryLocation"], info["UUID"]]

purchasinglocation[_, id_]:=URLBuild[$DefaultResourceAcquistionURL,{"UUID"->id}]

createResourceChannel[id_]:=(
	ChannelListen[ChannelObject[$ResourceAcquireChannelName, Permissions -> "Public"], 
 	resourceAcquireCallback[id,#]&, AutoRemove->True]
)

resourceAcquireCallback[id_,resp_]:=Quiet[
	resourceAcquireCallback[id,resp, resp["Message"]]
]

resourceAcquireCallback[id_,resp_,mess_String]:=resourceAcquireCallback[id,resp,URLQueryDecode[mess]]
resourceAcquireCallback[id_,resp_,KeyValuePattern[{"resource" -> id_, "status" -> "approved"}]]:=setResourceAcquired[id]

setResourceAcquired[id_]:=(
	summaryPurchaseStatus[id]=$summaryPurchased;
	updateResourceInfo[id,Association["MustAcquire"->False]]
	)

End[] 

EndPackage[]

SetAttributes[{ResourceAcquire,ResourceUpdate,ResourceRemove},
   {ReadProtected, Protected}
];