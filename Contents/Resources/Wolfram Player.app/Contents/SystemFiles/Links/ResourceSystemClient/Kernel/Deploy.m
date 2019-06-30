(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}



BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  

ResourceSystemClient`resourceLocalDeploy

ResourceSystemClient`$DeployResourceContent=True;
ResourceSystemClient`$DeployResourceExamples=False;
ResourceSystemClient`$CreateCloudResourceCache=True;

Begin["`Private`"] (* Begin Private Context *) 

cloudDeployResource[ro_System`ResourceObject, rest___]:=clouddeployResourceObject[resourceObjectID[ro],ro,rest]

cloudDeployResource[___]:=$Failed

clouddeployResourceObject[id_, _,rest___]:=clouddeployresourceObject[id, getResourceInfo[id], rest]/;MemberQ[$localResources, id]
clouddeployResourceObject[id_, _,rest___]:=(importandclouddeployresourceObject[id, resourceInfo[id], rest])/;MemberQ[$loadedResources, id]
clouddeployResourceObject[id_, ro_,rest___]:=loadandclouddeployresourceObject[id, ro, rest]

clouddeployResourceObject[___]:=Throw[$Failed]

importandclouddeployresourceObject[id_, info_,rest___]:=Block[{acquired},
	acquired=resourceAcquire[id,False, resourcerepositoryBase[info]];
	If[Head[acquired]===System`ResourceObject,
		clouddeployresourceObject[id, resourceInfo[id], rest]
	]
]/;marketplacebasedResourceQ[info]

importandclouddeployresourceObject[id_, info_,rest___]:=clouddeployresourceObject[id, info, rest]/;userdefinedResourceQ[info]

importandclouddeployresourceObject[___]:=Throw[$Failed]

loadandclouddeployresourceObject[id_, ro_, rest___]:=Block[{loaded},
	loaded=loadResource[id];
	If[AssociationQ[loaded]&&MemberQ[$loadedResources, id],
		clouddeployResourceObject[id, ro, rest]
	]
]

loadandclouddeployresourceObject[___]:=Throw[$Failed]


clouddeployresourceObject[id_, info_, rest___]:=clouddeployresourceobject[id,info, rest]/;cloudConnectedDeployQ[rest]

clouddeployresourceObject[id_, info_, rest___]:=(
	cloudDeployConnect[rest];
	clouddeployresourceObject[id, info, rest]
)

	
clouddeployresourceobject[id_,localinfo_, rest___]:=Block[{type,newinfo=Association[],cloudinfo, res, 
	fullinfo,cloudinfoco,$importCloudResources=True,autoloadResourceQ,nb=Automatic},
	autoloadResourceQ[___]:=False;
	type=getResourceType[localinfo];
	loadResourceType[type];
	fullinfo=repositoryBundleResourceObject[type,id, localinfo];
	If[TrueQ[ResourceSystemClient`$DeployResourceContent],
		newinfo=repositoryclouddeployResourceContent[type, id, fullinfo, rest];
		,
		If[containsLocalFileContentQ[localinfo],
			Message[ResourceObject::nocdep]
		]
	];
	If[TrueQ[ResourceSystemClient`$DeployResourceExamples],
		{nb,newinfo}=repositoryCloudDeployResourceExamples[type, id, fullinfo, rest]
	];
	newinfo["Autoload"]=True;
	fullinfo["Autoload"]=True;
	If[TrueQ[ResourceSystemClient`$CreateCloudResourceCache],
		{cloudinfo, cloudinfoco}=repositoryclouddeployResourceInfo[type,id,  fullinfo,newinfo, rest];
		addToCloudResourceIndex[cloudinfo];
		AppendTo[$cloudResources,id];
		,
		cloudinfo=Join[fullinfo, newinfo];
		cloudinfo["RepositoryLocation"]=None;
		cloudinfo["ResourceLocations"]={};
	];
	If[MatchQ[{rest},{None,___}],
		cloudinfoco
		,
		cloudDeployResourceObjectWithShingle[System`ResourceObject[Association[cloudinfo,"ExampleNotebookData"->nb]], rest]
	]
]

repositoryclouddeployResourceInfo[_,id_,  localinfo_, newinfo_, rest___]:=Block[{cloudinfo=localinfo,infoco},
	cloudinfo["ResourceLocations"]={CloudObject[cloudpath[resourceDirectory[id]]]};
	infoco=cloudResourceDirectoryObject[FileNameJoin[{StringTake[id,3], id,"metadata"}]];
	cloudinfo=Join[cloudinfo,newinfo];
	CloudPut[cloudinfo,infoco,takeFunctionOptions[CloudPut,rest]];
	{cloudinfo,infoco}
]

repositoryclouddeployResourceContent[___]:=Association[]
repositoryclouddeployResourceInfo[___]:=Throw[$Failed]

repositoryCloudDeployResourceExamples[type_, id_, fullinfo_, rest___]:=repositorycloudDeployResourceExamples[type, id, fullinfo, fullinfo["ExampleNotebook"],rest]

repositorycloudDeployResourceExamples[type_, id_, fullinfo_, file:(_File|_LocalObject|_String?FileExistsQ),rest___]:={
	Import[file],
	Association[
	fullinfo,
	"ExampleNotebook"->CopyFile[file,cloudDeployExampleLocation[type,id,rest]]
]}

repositorycloudDeployResourceExamples[type_, id_, fullinfo_, nb:(_Notebook|_NotebookObject),rest___]:={
nb,
Association[
	fullinfo,
	"ExampleNotebook"->CloudDeploy[nb,cloudDeployExampleLocation[type,id,rest]]
]}

repositorycloudDeployResourceExamples[type_, id_, fullinfo_, co:HoldPattern[_CloudObject],rest___]:={
	Import[co],
	Association[fullinfo]
}

repositorycloudDeployResourceExamples[_, _, fullinfo_,___]:=fullinfo

cloudDeployExampleLocation[___]:=CloudObject[]

repositoryBundleResourceObject[_,_, localinfo_]:=localinfo

containsLocalFileContentQ[localinfo_]:=containslocalFileContentQ[Lookup[localinfo,{"ContentElementLocations","ContentLocation"}]]

containslocalFileContentQ[locations_]:=containslocalfileContentQ[Flatten[locations /. as_Association :> Values[as]]]

containslocalfileContentQ[locations_]:=(!FreeQ[locations,LocalObject|System`File])||AnyTrue[Select[locations,StringQ],FileExistsQ]

ResourceSystemClient`Private`$resourceObjectShingleDeploy=True;

cloudDeployResourceObjectWithShingle[ro_, rest___]:=(
	loadDeployResourceShingle[];
	DeployedResourceShingle`CloudDeployResourceWithShingle[ro, rest]
)/;TrueQ[ResourceSystemClient`Private`$resourceObjectShingleDeploy]


cloudDeployResourceObjectWithShingle[ro_, rest___]:=Block[{System`ResourceObject, res},
    res=CloudDeploy[ExportForm[ro,"WL"], rest];
	If[Head[res]===CloudObject,
		res
		,
		$Failed
	]
]

loadDeployResourceShingle[]:=(loadDeployResourceShingle[]=Get["ResourceSystemClient`WebpageDeployment`DeployedResourceShingle`"])


resourceLocalCache[ro_ResourceObject]:=resourcelocalDeploy[ro, LocalObject[]]
resourceLocalCache[ro_ResourceObject, lo:(HoldPattern[_LocalObject]|Automatic)]:=resourcelocalDeploy[ro, lo]

ResourceSystemClient`resourceLocalDeploy[args___]:=Catch[resourcelocalDeploy[args]]

resourcelocalDeploy[ro_ResourceObject,Automatic]:=resourcelocalDeploy[ro, defaultLocalDeployLocation[ro]]
resourcelocalDeploy[ro_ResourceObject]:=resourcelocalDeploy[ro, None]

resourcelocalDeploy[ro_ResourceObject, location_]:=Block[{res=saveResourceObject[ro]},
	If[!FailureQ[res],
		res=Switch[location,
			None, Null,
			_File,Put[ro, location[[1]]],
			_String|_LocalObject,Put[ro, location],
			_,(Message[ResourceRegister::invloc, location]);$Failed
		];
		If[!FailureQ[res],
			If[location=!=None,location]
			,
			$Failed
		]
	]
]


resourcelocalDeploy[___]:=$Failed

defaultLocalDeployLocation[HoldPattern[ResourceObject][info:KeyValuePattern[{"ResourceType"->rtype_String}]]]:=defaultLocalDeployLocation[rtype,info]
defaultLocalDeployLocation[___]:=None

defaultLocalDeployLocation[rtype_String,info_Association]:=defaultLocalDeployLocation[rtype,Lookup[info,"ShortName",Lookup[info,"Name"]]]

defaultLocalDeployLocation[rtype_,name_String]:=With[{safename=safeLocalObjectName[name]},
	If[StringLength[safename]>0,
		LocalObject[FileNameJoin[{localDeployDefaultPath[rtype],safename},OperatingSystem->"Unix"]],
		None
	]
]

safeLocalObjectName[name_]:=StringReplace[name, {Whitespace -> "-"}]

localDeployDefaultPath[rtype_]:=FileNameJoin[{"DeployedResources",rtype},OperatingSystem->"Unix"]

cloudConnectedDeployQ[rest___]:=cloudconnectedDeployQ[Cases[{rest}, HoldPattern[Rule | RuleDelayed][CloudBase, base_] :>base, 5]]

cloudconnectedDeployQ[{}]:=$CloudConnected
cloudconnectedDeployQ[{cloudbase}]:=cloudbaseConnected[cloudbase]

cloudconnectedDeployQ[___]:=False

cloudDeployConnect[rest___]:=clouddeployConnect[Cases[{rest}, HoldPattern[Rule | RuleDelayed][CloudBase, base_] :>base, 5]]

clouddeployConnect[{}]:=cloudConnect[ResourceObject]
clouddeployConnect[{cloudbase_}]:=cloudConnect[ResourceObject,cloudbase]

End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];