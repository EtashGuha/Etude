BeginPackage["DataResource`"]

Begin["`Private`"] 


resourcedataResourceSystem[{id_, info_},elem_,rest___]:=
	resourcedataresourceSystem[id,info,elem, rest]/;MemberQ[
		Keys[Lookup[info,"ContentElementLocations",Association[]]],elem]
		
$reacquired=False;

resourcedataResourceSystem[{id_, info_},elem_,rest___]:=Block[{$reacquired=True, res},
	res=ResourceSystemClient`Private`resourceacquire[id, 
		Association["AddToAccount"->True,System`ResourceSystemBase->ResourceSystemClient`Private`resourcerepositoryBase[info]]];
	If[resourceObjectQ[res],
		resourceData[res, elem, rest]
	]
]/;!TrueQ[$reacquired]

resourcedataResourceSystem[___]:=$Failed

resourcedataresourceSystem[id_,info_,elem_,rest___]:=
	Block[{$messages={}},
		ResourceSystemClient`Private`$CDNFailure=False;
		With[{res=directDataResourceDownloadQuiet[cacheResourceQ[info],id, info, elem,rest]},
		If[FailureQ[res]&&TrueQ[ResourceSystemClient`Private`$CDNFailure],
			Block[{freeResourceDataDownloadElementQ,ResourceSystemClient`Private`$ClientInformation},
				ResourceSystemClient`Private`$ClientInformation={
					"PacletVersion"->"1.0",
					"WLVersion"->ToString[ResourceSystemClient`Private`$VersionNumber],
					"MachineType"->ToString[$MachineType],
					"WordLength"->ToString[$SystemWordLength]
				};
				freeResourceDataDownloadElementQ[___]:=False;
				resourcedataresourceSystem[id,info,elem,rest]
			],
			If[MemberQ[messages, HoldForm[ResourceData::baddl]],
				Message[ResourceData::baddl]
			];
			res
		]
]]/;freeResourceDataDownloadElementQ[info,elem]

directDataResourceDownloadQuiet[args___]:=Quiet[With[{res=directDataResourceDownload[args]},
	$messages=$MessageList;
	res
],{ResourceData::baddl}]

directDataResourceDownload[True,id_,info_,elem_,rest___]:=
Block[{downloadinfo},
	downloadinfo=ResourceSystemClient`Private`repositoryresourcedownload[Lookup[info,"ResourceType","DataResource"],
		id,Association[
			"Element"->elem,
			"Version"->Lookup[info,"Version"],
			"DownloadInfo"->
				{Association[
					"Format"->info["Format"][elem],
					"Hash"->Lookup[Lookup[info,"Hash",Association[]],elem,None],
					"ContentElementLocations"->info["ContentElementLocations"][elem]]}
				]];
		If[AssociationQ[downloadinfo],
			resourcedataLocal[id,elem,rest],
			$Failed
		]
]

directDataResourceDownload[False,id_,info_,elem_,rest___]:=Block[
	{ResourceSystemClient`Private`$CacheResourceContent=False},
	directdataResourceDownload[id, info, elem, rest]
]

directdataResourceDownload[id_, info_, elem_, rest___]:=With[{res=
	Catch[ResourceSystemClient`Private`repositoryresourcedownload[Lookup[info,"ResourceType","DataResource"],
		id,Association[
			"Element"->elem,
			"Version"->Lookup[info,"Version"],
			"DownloadInfo"->
				{Association[
					"Format"->info["Format"][elem],
					"Hash"->Lookup[Lookup[info,"Hash",Association[]],elem,None],
					"ContentElementLocations"->info["ContentElementLocations"][elem]]}
				]],
				"NoCacheResourceDownload"
		]
	},
	If[cacheResourceContentInMemory[Lookup[info,"ResourceType"],info],
		If[!FailureQ[res],
			setElementCached[id,elem];
			directdataResourceDownload[id,_,elem,rest]=res
			,
			res		
		]
		,
		res
	]
]


resourcedataresourceSystem[id_,info_,elem_,rest___]:=Block[{resource, PrintTemporary},
	If[cacheResourceQ[info],
		resource=ResourceSystemClient`ResourceDownload[id, Association["Element"->elem,"ResponseMethod"->"Download"]];
		If[resourceObjectQ[resource],
			resourcedataLocal[resource,elem,rest]
		]
		,
		ResourceSystemClient`Private`readresource[id, Association["Element"->elem,"ResponseMethod"->"Download"]]
	]
]

resourcedataLocal[ro_System`ResourceObject,rest___]:=resourcedataLocal[resourceObjectID[ro],rest]

resourcedataLocal[id_String,rest___]:=resourcedataLocal[resourceInfo[id],rest]/;MemberQ[$loadedResources,id]

resourcedataLocal[id_String,rest___]:=resourcedataLocal[ResourceSystemClient`Private`getResourceInfo[id],rest]

resourcedataLocal[info_]:=resourcedataLocal[info, Automatic]

elempattern0=(_String|Automatic|All);
elempattern=(elempattern0|{elempattern0...});

resourcedataLocal[info_Association,elem:elempattern,___]:=resourcedatalocal[info,elem]

resourcedataLocal[___]:=Throw[$Failed]

resourcedatalocal[info_,elem_]:=resourcedatalocal[Lookup[info,"ResourceType",Throw[$Failed]],info, elem]

resourcedatalocal[rtype_,info_, elem_]:=With[{res=ResourceSystemClient`Private`readResourceElementLocal[Lookup[info,"UUID",Throw[$Failed]], info, elem],
	id=info["UUID"]},
	If[!FailureQ[res],
		setElementCached[id,elem];
		resourcedatalocal[rtype,info, elem]=res
		,
		res		
	]
]/;cacheResourceContentInMemory[rtype,info]

resourcedatalocal[_,info_, elem_]:=ResourceSystemClient`Private`readResourceElementLocal[Lookup[info,"UUID",Throw[$Failed]], info, elem]

resourcedataCloud[rtype_,info_Association,elem_,rest___]:=With[{newinfo=cloudresourceDownload[rtype,
	info,Lookup[info,"ContentElementLocations",Throw[$Failed]], elem]},
	If[localStoredQ[newinfo, elem],
		resourcedataLocal[newinfo,elem, rest]
	]
]


resourcedataUncached[{_, info_Association}, Automatic|"Content",rest___]:=info["Content"]/;KeyExistsQ[info,"Content"]
resourcedataUncached[{_, info_Association}, elem_String,rest___]:=info["ContentValues",elem]/;keyExistsQ[info,{"ContentValues",elem}]
resourcedataUncached[{_, info_Association}, elem_String,rest___]:=info["ContentElements",elem]/;keyExistsQ[info,{"ContentElements",elem}]
resourcedataUncached[{id_, info_Association}, elem_String,rest___]:=resourcedatauncached1[id,info,elem,info["ContentElementLocations",elem]]/;keyExistsQ[info,{"ContentElementLocations",elem}]
resourcedataUncached[{id_, info_Association}, elem_String,rest___]:=produceResourceElementContent[id, info, elem, info["ContentElementFunctions",elem]]/;keyExistsQ[info,{"ContentElementFunctions",elem}]

resourcedataUncached[___]:=$Failed

resourcedatauncached1[id_,info_,elem_,loc_]:=resourcedatauncached[info["ResourceType"],id,elem,info["ElementInformation"][elem],loc]/;TrueQ[Quiet[KeyExistsQ[info["ElementInformation"],elem]]]
resourcedatauncached1[__,loc_]:=resourcedatauncached[loc]

$importProgressBarSize=10^7;
resourcedatauncached[lo:HoldPattern[_LocalObject]]:=(
	If[fileByteCount[lo]>$importProgressBarSize,
		printTempOnce[ResourceSystemClient`$progressID]
	];
	importlocal[lo]
	)
	
resourcedatauncached[file:File[path_]]:=(
	If[fileByteCount[lo]>$importProgressBarSize,
		printTempOnce[ResourceSystemClient`$progressID]
	];
	Import[path]
	)
	
resourcedatauncached[co:HoldPattern[_CloudObject]]:=(
	printTempOnce[ResourceSystemClient`$progressID];
	CloudImport[co]
	)
	
resourcedatauncached[url:HoldPattern[_URL]]:=(
	printTempOnce[ResourceSystemClient`$progressID];
	Import[url]
	)


resourcedatauncached[str_String]:=resourcedatauncached[File[str]]/;FileExistsQ[str]
resourcedatauncached[_]:=$Failed


resourcedatauncached[rtype_,id_,elem_,eleminfo_,co:HoldPattern[_CloudObject]]:=(
	printTempOnce[ResourceSystemClient`$progressID];
	ResourceSystemClient`Private`noCacheCloudDownload[rtype,id,elem,eleminfo["Format"],co]
	)/;KeyExistsQ[eleminfo,"Format"]

resourcedatauncached[__,loc_]:=resourcedatauncached[loc]



freeResourceDataDownloadElementQ[info_,elem_]:=False/;KeyExistsQ[info,"PricingInformation"]
freeResourceDataDownloadElementQ[info_,elem_]:=False/;!TrueQ[Lookup[info,"Public",True]]
freeResourceDataDownloadElementQ[info_,elem_]:=freeresourceDataDownloadQ[info["ContentElementLocations"],info["Format"],elem]
freeResourceDataDownloadElementQ[__]:=False

freeresourceDataDownloadQ[locations_Association,formats_Association,elem_]:=
	freeresourcedataDownloadQ[locations[elem],formats[elem]]
freeresourceDataDownloadQ[___]:=False

freeresourcedataDownloadQ[HoldPattern[_CloudObject|_URL],fmt:(_String|Automatic)]:=True
freeresourcedataDownloadQ[___]:=False

cacheResourceContentInMemory[__]:=False

clearInMemoryElementCache[rtype_,id_, info_]:=Quiet[
	(resourcedatalocal[rtype,info,#]=.;)&/@info["ContentElements"];
	(directdataResourceDownload[id,info,#]=.;)&/@info["ContentElements"];
	,Unset::norep
]



End[] 

EndPackage[]
