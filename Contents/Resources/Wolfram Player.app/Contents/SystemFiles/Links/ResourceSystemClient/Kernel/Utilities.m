(* Wolfram Language Package *)


BeginPackage["ResourceSystemClient`"]

ResourceSystemClient`Private`CompressedInformationElement
ResourceSystemClient`$AsyncronousResourceInformationUpdates

Begin["`Private`"] (* Begin Private Context *) 
If[!MatchQ[ResourceSystemClient`$AsyncronousResourceInformationUpdates,True|False],
	ResourceSystemClient`$AsyncronousResourceInformationUpdates=True
];

$ResourceBase="Resources";
$CloudResourceBase="Resources";

$rscDirectory=DirectoryName[System`Private`$InputFileName];

localObject:=Identity/;$CloudEvaluation
localObject:=(localobject[toRelativePath[#],resourceCacheDirectory[]]&)

localobject["",base_]:=System`LocalObject[base]
localobject[path_,base_]:=System`LocalObject[path,base]

resourceCacheDirectory[]:=$ResourceBase/;$CloudEvaluation;
resourceCacheDirectory[]:=(checkWordLength[];resourcecacheDirectory[]);

resourcecacheDirectory[]:=FileNameJoin[{LocalObjects`PathName[LocalObject[$DefaultLocalBase]], $ResourceBase}]

$ResourceSystemClientDebug=False;

$loadedResources={};

marketPlaceReviewerGroup:=marketPlaceReviewerGroup=PermissionsGroup[$ResourceSystemAdminUser,"Reviewers"];

(* Directories *)
toRelativePath[co:HoldPattern[_CloudObject]]:=toRelativePath[First[co]]
toRelativePath[path_]:=path/;StringFreeQ[path,$ResourceBase]
toRelativePath[path_]:=FileNameJoin[Replace[FileNameSplit[path],{___,$ResourceBase,rest___}:>{rest},{0}],OperatingSystem->"Unix"]

createDirectory[dir_]:=Quiet[CreateDirectory[dir, CreateIntermediateDirectories->True],CreateDirectory::filex]/;!DirectoryQ[dir]

resourceDirectory[id_]:=FileNameJoin[{resourceCacheDirectory[],StringTake[id,3], id}]

resourceInfoFile[id_]:=localObject[FileNameJoin[{resourceDirectory[id],"metadata"}]]
resourceinfoFile[dir_]:=localObject[FileNameJoin[{dir,"metadata"}]]

cloudResourceDirectoryObject[path_]:=FileNameJoin[Flatten[{$CloudRootDirectory,$CloudResourceBase, FileNameSplit[path]}]]

Attributes[cloudpath]={HoldFirst};
cloudpath[expr_]:=FileNameJoin[(FileNameSplit[expr]/.{___,$ResourceBase,rest___}:>{$CloudResourceBase,rest}),OperatingSystem->"Unix"]

(* common utility functions *)

rpat=(_System`ResourceObject);

resourceObjectQ[ro_System`ResourceObject]:=AssociationQ[First[ro]]
resourceObjectQ[___]:=False

getResourceInfo[id_,keys_]:=With[{info=getResourceInfo[id]},
	If[AssociationQ[info],
		KeyTake[info,keys],
		Missing["NotAvailable"]
	]
]


getResourceInfo[id_]:=With[{info=resourceInfo[id]},
	Block[{CloudObject},
		If[AssociationQ[info],info,getresourceInfo[id]]
	]
]

$usableResourceInfoKeys={
	"Name","UUID","ResourceType",
	"Version","Description","RepositoryLocation",
	"WolframLanguageVersionRequired"
};

usableResourceInfo[as_]:=usableresourceInfo[as,getResourceType[as]]
usableresourceInfo[as_, rtype_]:=KeyTake[as,usableResourceInfoKeys[rtype]]

usableResourceInfoKeys[rtype_]:=DeleteDuplicates[Flatten[{$usableResourceInfoKeys,usableresourceInfoKeys[rtype]}]]

usableresourceInfoKeys[rtype_String]:=(loadResourceType[rtype];usableresourceinfoKeys[rtype])

usableresourceinfoKeys[_]:={}

getresourceInfo[id_]:=Block[{lo=resourceInfoFile[id], info},
	If[fileExistsQ[lo],
		info=Quiet[Get[lo]];
		If[AssociationQ[info],
			info=standardizeResourceInfo[info];
			resourceInfo[id]=info;
			info
			,
			importresourceInfo[id,False]
		]
		,
		Missing["NotAvailable"]
	]
]

standardizeResourceInfo[info_Association]:=$Failed/;!KeyExistsQ[info,"UUID"]


standardizeResourceInfo[info_Association]:=With[{rtype=getResourceType[info]},
	loadResourceType[rtype];
	sortBasicInfo[
	repositorystandardizeResourceInfo[rtype,info]]
]
standardizeResourceInfo[l_List]:=standardizeResourceInfo[Association[l]]
standardizeResourceInfo[expr_]:=expr

repositorystandardizeResourceInfo[_,info_Association]:=info

setResourceInfo[id_, as_Association]:=Block[{lo=resourceInfoFile[id], info},
	If[fileExistsQ[lo],
		info=Get[lo];
		If[AssociationQ[info],
			info=Join[info,as],
			info=as
		];
		resourceInfo[id]=info;
		Put[info, lo];
		info
		,
		If[AssociationQ[resourceInfo[id]],
			resourceInfo[id]=sortBasicInfo[Join[resourceInfo[id],as]]
		]
	]
]

setResourceInfo[id_, opts__?OptionQ]:=setResourceInfo[id, Association[opts]]
setResourceInfo[___]:=$Failed

cloudConnect[head_]:=cloudConnect[head, $CloudBase]
cloudConnect[head_, cloudbase_]:=(
    CloudConnect[CloudBase->cloudbase];
    If[!cloudbaseConnected[cloudbase],
        Message[head::cloudc];
        Throw[$Failed]
    ]
)

requestBaseConnected[]:=requestBaseConnected[$resourceSystemRequestBase]

requestBaseConnected[requestbase_String]:=cloudbaseConnected[tocloudbase[requestbase]]

cloudbaseConnected[cloudbase_]:=True/;$CloudEvaluation&&URLParse[$EvaluationCloudBase]["Domain"]===URLParse[cloudbase]["Domain"]

cloudbaseConnected[cloudbase_]:=(Needs["CloudObject`"];TrueQ[
	CloudObject`Internal`CloudConnectStatus[withTrailingSlash[cloudbase]]
	||CloudObject`Internal`CloudConnectStatus[withoutTrailingSlash[cloudbase]]])
	
requestBaseConnected[Automatic]:=$CloudConnected
requestBaseConnected[___]:=False

withTrailingSlash[str_]:=StringReplace[str, (char:Except["/"] ~~ EndOfString) :> char <> "/"]
withoutTrailingSlash[str_]:=StringReplace[str, ("/"~~ EndOfString) -> ""]

requestBaseConnect[head_]:=requestBaseConnect[head,$resourceSystemRequestBase]
requestBaseConnect[head_, Automatic]:=requestBaseConnect[head,$resourceSystemRequestBase]
requestBaseConnect[head_,requestbase_]:=requestbaseConnect[head,tocloudbase[requestbase]]

requestbaseConnect[head_,cloudbase_]:=cloudConnect[head,cloudbase]

tocloudBase[as_Association]:=tocloudbase[resourcerepositoryBase[as]]
tocloudbase[requestbase_String]:=(tocloudbase[requestbase]=URLBuild@KeyDrop[URLParse[requestbase], "Path"])
tocloudbase[Automatic]:=$CloudBase

resourceObjectID[object_]:=Lookup[First[object],"UUID",$Failed]
resourceObjectName[object_]:=Lookup[First[object],"Name",
	Lookup[getResourceInfo[resourceObjectID[object]],"Name",Missing["NotAvailable"]]]

resourceRepositoryBase[object_]:=resourcerepositoryBase[First[object]]
resourcerepositoryBase[as_Association]:=resourcerepositorybase[Lookup[as,"RepositoryLocation",$resourceSystemRequestBase]]
resourcerepositoryBase[_]:=$resourceSystemRequestBase

resourcerepositorybase[str_String]:=str
resourcerepositorybase[url_URL]:=First[url]
resourcerepositorybase[_]:=$resourceSystemRequestBase

uuidQ[str_String]:=StringMatchQ[str, RegularExpression["\\w{8}-\\w{4}-\\w{4}-\\w{4}-\\w{12}$"]]
uuidQ[_]:=False

newerversionQ[{x_, x_}]:=False
newerversionQ[{new_String, old_String}]:=newerversionQ[ToExpression[StringSplit[new,"."]],ToExpression[StringSplit[old,"."]]]
newerversionQ[{new_String, old_}]:=True
newerversionQ[{new_, old_String}]:=False
newerversionQ[_]:=False

newerversionQ[new:{__Integer},old:{__Integer}]:=Catch[
    (If[new[[#]]>old[[#]],Throw[True,"newer"],If[old[[#]]>new[[#]],Throw[False,"newer"]]
    ]&/@Range[Min[Length/@{new,old}]];False),"newer"]/;Length[new]<=Length[old]

newerversionQ[new_List,old_List]:=newerversionQ[new,PadRight[old,Length[new]]]

versionnumberstring[str_]:=str<>"0"/;StringMatchQ[str,"*."]
versionnumberstring[str_]:=str/;StringContainsQ[str,"."]
versionnumberstring[str_]:=str<>".0"

fillResourceMetadata[info_,more_]:=Join[more, info]

fileExistsQ[HoldPattern[lo_LocalObject]]:=FileExistsQ[LocalObjects`PathName[lo]]
fileExistsQ[expr_]:=TrueQ[Quiet[FileExistsQ[expr]]]

createParentDirectory[lo_LocalObject]:=Null
createParentDirectory[file_String]:=createDirectory[FileNameDrop[file]]

cacheresourceInfo[l_, updateIndex_:True]:=Block[{$updateResourceSearchIndex=False, res},
	res=cacheresourceinfo/@l;
	If[$AllowResourceTextSearch&&updateIndex,
		updateLocalResourceSearchIndex[]
	];
	res
]

cacheresourceinfo[info_]:=Block[{dir, id=Lookup[info,"UUID",Throw[$Failed]], newinfo=info,rtype},
	dir=resourceDirectory[id];
	newinfo["ResourceLocations"]=DeleteDuplicates[Flatten[{Lookup[newinfo,"ResourceLocations",{}],localObject[dir]}]];
	rtype=getResourceType[info];
	loadResourceType[rtype];
	newinfo=repositoryCacheResourceInfo[rtype,id, newinfo, dir];
	If[$AllowResourceTextSearch&&userdefinedResourceQ[info],
		addToLocalSearchIndex[id, newinfo,dir]
	];
	If[$localResourcesCompiled,
		$localResources=Cases[DeleteDuplicates[Append[$localResources,id]],_String]
	];
	If[Head[info["ExampleNotebook"]]===NotebookObject,
		saveExampleNotebook[id, info["ExampleNotebook"]]
	];
	loadresource[newinfo];
	newinfo

]

repositoryCacheResourceInfo[_,id_, info_, dir_]:=Block[{newinfo},
	If[DirectoryQ[dir],
		newinfo=updatecachedresourceinfo[dir, id, info]
		,
		createDirectory[dir];
		newinfo=Append[info,"DownloadedVersion"->None];
		Put[newinfo,resourceinfoFile[dir]];
	];
	newinfo
]



deleteresourcecache[info_Association]:=With[{id=Lookup[info,"UUID",Throw[$Failed]], name=Lookup[info,"Name"]},
	If[FreeQ[Lookup[info,"ResourceLocations"],_CloudObject],
		deleteresourcecache[id,name],
		Block[{$unregisterPath},
			$unregisterPath:=Prepend[$PersistencePath,"Cloud"];
			deleteresourcecache[id,name]
		]
	]
]

deleteresourcecache[id_String,name_:Missing[]]:=With[{dir=FileNameJoin[{resourceCacheDirectory[],StringTake[id,3], id}]},
	If[DirectoryQ[dir],
		DeleteDirectory[dir, DeleteContents->True]
	];
	If[$localResourcesCompiled,
		$localResources=Cases[DeleteCases[$localResources,id],_String]
	];
	clearresource[id, name]
]

$forceCacheUpdate=False;
updatecachedresourceinfo[dir_, id_, info_]:=Block[{infofile,oldinfo, olddownloadversion, newinfo},
	infofile=resourceinfoFile[dir];
	If[fileExistsQ[infofile],
		oldinfo=Quiet[Get[infofile]];
		If[!AssociationQ[oldinfo],
			oldinfo=Association[]
		];
		,
		oldinfo=Association[]
	];
	If[$forceCacheUpdate||newerversionQ[Lookup[{info,oldinfo},"Version"]],
		olddownloadversion=If[fileExistsQ[infofile],
			Lookup[oldinfo,"DownloadedVersion",None],
			None
		];
		newinfo=Append[info,"DownloadedVersion"->olddownloadversion];
		Put[newinfo,infofile];
		newinfo
		,
		newinfo=mergeResourceInfo[info, oldinfo];
		Put[newinfo,infofile];
		newinfo
	]
]

mergeResourceInfo[info_Association, oldinfo_Association]:=Merge[{info, oldinfo},mergeinfofun]
mergeResourceInfo[info_Association, oldinfo_]:=info
mergeResourceInfo[info_, oldinfo_Association]:=oldinfo
mergeResourceInfo[__]:=Association[]


mergeinfofun[l:{_Association..}]:=Join@@l
mergeinfofun[l:{_List..}]:=DeleteDuplicates[Flatten[l]]
mergeinfofun[{str1_String,str2_String}]:=str1
mergeinfofun[l_]:=Last[l]

$SortingPropertyCloudObject:=($SortingPropertyCloudObject=CloudObject[URLBuild[MapAt[Append[Drop[#, -2], "SortingPropertyList"] &, URLParse[$resourceSystemRequestBase], "Path"]]])

$sortingPropertiesAcquired=False;
$resourceSortingProperties:=($resourceSortingProperties=getResourceSortingProperties[])

resourceSortingProperties[All]:=$resourceSortingProperties

resourceSortingProperties[rtype_, False]:=If[TrueQ[$sortingPropertiesAcquired],
	resourceSortingProperties[rtype],
	If[$CloudConnected,
		resourceSortingProperties[rtype]
		,
		loadResourceType[rtype];
		defaultSortingProperties[rtype]
	]
]

resourceSortingProperties[rtype_,___]:=Lookup[$resourceSortingProperties,rtype,Association[]]

defaultSortingProperties[rtype_]:=Association[rtype->{}]

getResourceSortingProperties[]:=With[{res=CloudGet[$SortingPropertyCloudObject]},
	If[AssociationQ[res],
		$sortingPropertiesAcquired=True;
		res
		,
		Association[]
	]
]


publicResourceInformationObject["Versions",rsbase_,wlv_String]:=(
	publicResourceInformationObject["Versions",rsbase,wlv]=
	CloudObject[URLBuild[MapAt[Join[Drop[#, -2],
		{"VersionInformation", StringReplace[wlv,"."->"-"],"ResourceVersions"}] &, URLParse[rsbase], "Path"]]])
	
publicResourceInformationObject["Versions",rsbase_,___]:=CloudObject[URLBuild[MapAt[Join[Drop[#, -2],
		{"ResourceVersions"}] &, URLParse[rsbase], "Path"]]]

publicResourceInformationObject["Names",rsbase_,___]:=(
	publicResourceInformationObject["Names",rsbase,___]=CloudObject[URLBuild[MapAt[Append[Drop[#, -2], "ResourceNames"] &, 
	URLParse[rsbase], "Path"]]])

Clear[publicInfoAcquired];
Clear[publicResourceInformationKernelCache];
publicInfoAcquired[__]:=False;

resourceVersionsAvailable[args___]:=With[{rvs=publicResourceInformation["Versions",args]},
	If[AssociationQ[rvs],
		Lookup[rvs,"Versions",Association[]]
		,
		Association[]
	]
]

resourceVersionsRequired[args___]:=With[{rvs=publicResourceInformation["Versions",args]},
	If[AssociationQ[rvs],
		Lookup[rvs,"RequiredUpdates",Association[]]
		,
		Association[]
	]
]
publicResourceInfoUpdating[__]:=False

publicResourceInformation[infotype_]:=publicResourceInformation[infotype,$ResourceSystemBase]

publicResourceInformation[infotype_,rsbase_]:=publicResourceInformation[infotype,rsbase, Automatic]

publicResourceInformation[infotype_,rsbase_String, updateCache_]:=If[TrueQ[publicInfoAcquired[infotype,rsbase]],
	publicResourceInformationKernelCache[infotype,rsbase],
	Switch[updateCache,
			Automatic,
			With[{res=publicResourceInformationLocalCache[infotype,rsbase]},
				If[!TrueQ[publicResourceInfoUpdating[infotype,rsbase]],
					updatePublicResourceInformationCacheAsynchronous[infotype,rsbase]
				];
				res
			],
			True,
			If[cloudbaseConnected[tocloudbase[rsbase]],
				getPublicResourceInformation[infotype,rsbase],
				publicResourceInformationLocalCache[infotype,rsbase]
			],
			False,
			publicResourceInformationLocalCache[infotype,rsbase]
	]
]

getPublicResourceInformation[infotype_,rsbase_]:=With[{res=getpublicResourceInformation[infotype,rsbase]},
	If[AssociationQ[res],
		publicResourceInfoCallback[infotype,rsbase,res]
		,
		Association[]
	]
]

getpublicResourceInformation["Versions",rsbase_]:=Quiet[
	With[{v=Import[publicResourceInformationObject["Versions",rsbase,versionnumberstring[ToString@$VersionNumber]],"RawJSON"]},
	If[AssociationQ[v],
		v,
		Import[publicResourceInformationObject["Versions",rsbase],"RawJSON"]
	]
]]

getpublicResourceInformation[infotype_,rsbase_]:=Quiet[Import[publicResourceInformationObject[infotype,rsbase],"RawJSON"]]


getPublicResourceInformationAsynchronous[infotype_,rsbase_]:=URLSubmit[First[publicResourceInformationObject[infotype,rsbase]], 
	HandlerFunctions -> Association["BodyReceived" -> 
		(Quiet[publicResourceInfoCallback[infotype,rsbase,#]]&)], HandlerFunctionsKeys -> {"BodyByteArray"}]

publicResourceInfoCallback[infotype_,rsbase_,as_Association]:=publicresourceInfoCallback[infotype,rsbase,as["BodyByteArray"]]/;Quiet[KeyExistsQ[as,"BodyByteArray"]]
publicResourceInfoCallback[infotype_,rsbase_,as_Association]:=publicresourceInfoCallback[infotype,rsbase,as]

publicresourceInfoCallback[infotype_,rsbase_,ba_ByteArray]:=publicresourceInfoCallback[infotype,rsbase,Quiet[ImportByteArray[ba, "RawJSON"]]]
publicresourceInfoCallback[infotype_,rsbase_,as_Association]:=With[{info=importPublicResourceInfo[infotype,as]},
		storeResourceInformationCache[infotype,rsbase,info];
		publicInfoAcquired[infotype,rsbase]=True;
		If[rsbase===$ResourceSystemBase&&infotype==="Names",
			$lastPublicResourceNameBase=rsbase;
			setResourceNameAutocomplete[info]
		];
		publicResourceInformationKernelCache[infotype,rsbase]=info
]

publicresourceInfoCallback[__]:=Null

importPublicResourceInfo["Names",as_Association]:=as["ResourceNames"]
importPublicResourceInfo[_,info_]:=info

publicResourceInformationLocalCache[infotype_,rsbase_]:=With[{res=publicresourceInformationLocalCache[localResourceInformationCache[infotype,rsbase]]},
		If[rsbase===$ResourceSystemBase&&infotype==="Names"&&AssociationQ[res],
			setResourceNameAutocomplete[res]
		];
		res
]

publicresourceInformationLocalCache[file_]:=With[{cache=Quiet[Get[file]]},
	If[AssociationQ[cache],
		cache,
		$Failed
	]
]/;FileExistsQ[file]

publicresourceInformationLocalCache[_]:=Association[]

updatePublicResourceInformationCacheAsynchronous[infotype_,rsbase_]:=getPublicResourceInformation[infotype,rsbase]/;$CloudEvaluation&&$CloudConnected

updatePublicResourceInformationCacheAsynchronous[infotype_,rsbase_]:=SessionSubmit[
	ScheduledTask[If[cloudbaseConnected[tocloudbase[rsbase]],
		getPublicResourceInformationAsynchronous[infotype,rsbase];
	], {1}]]/;ResourceSystemClient`$AsyncronousResourceInformationUpdates
	
localResourceInformationCache["Versions",rsbase_]:=localResourceCacheFile["Versions",rsbase]
localResourceInformationCache["Names",rsbase_]:=localResourceNamesCache[localResourceCacheFile["Names",rsbase]]
localResourceInformationCache[__]:=$Failed

localResourceCacheFile["Versions",rsbase_]:=FileNameJoin[{resourceCacheDirectory[],"versionscache",stringhash[{rsbase,$VersionNumber}]}]
localResourceCacheFile["Names",rsbase_]:=FileNameJoin[{resourceCacheDirectory[],"namescache",stringhash[rsbase]}]

localResourceCacheWordLengthFile[]:=FileNameJoin[{resourcecacheDirectory[],"wordlength"}]

checkWordLength[]:=checkWordLength[localResourceCacheWordLengthFile[]]
checkWordLength[file_String]:=checkWordLength[Quiet[Get[file]]]/;FileExistsQ[file]
checkWordLength[int_Integer]:=(checkWordLength[]=Null)/;int===$SystemWordLength
checkWordLength[int_Integer]:=($ResourceBase=$ResourceBase<>ToString[$SystemWordLength])/;int>$SystemWordLength
checkWordLength[int_Integer]:=With[{file=localResourceCacheWordLengthFile[]},
	Put[$SystemWordLength,file];
	checkWordLength[file]
]/;int<$SystemWordLength

checkWordLength[_]:=(checkWordLength[]=Null)

checkWordLength[file_String]:=(
	createDirectory[FileNameDrop[file]];
	Put[$SystemWordLength,file];
	checkWordLength[]=Null;
	)

localResourceNamesCache[file_]:=file/;fileExistsQ[file]
localResourceNamesCache[_]:=$installedResourceNamesFile

$installedResourceNamesFile:=FileNameJoin[{$rscDirectory,"Data","DefaultResourceNames"}]

storeResourceInformationCache[infotype_,rsbase_,cache_Association]:=storeresourceInformationCache[localResourceCacheFile[infotype,rsbase],cache]

storeresourceInformationCache[file_,cache_]:=(createDirectory[FileNameDrop[file]];Quiet[Put[cache,file]])

bytecountQuantity[n_?NumberQ]:=Which[
	n<2000,
	Quantity[n,"Bytes"],
	n<2*10^6,
	Quantity[N[n]/10^3,"Kilobytes"],
	n < 2*10^9, 
	Quantity[N[n]/10^6, "Megabytes"], 
	True, 
	Quantity[N[n]/10^9, "Gigabytes"]
]

bytecountQuantity[expr_]:=expr


fileByteCount[HoldPattern[lo_LocalObject]]:=With[{file=localObjectDataFile[lo]},
	If[MissingQ[file],
		file,
		FileByteCount[file]
	]
]
fileByteCount[expr_]:=FileByteCount[expr]
filebyteCount[___]:=Missing["NotAvailable"]

localObjectDataFile[HoldPattern[lo_LocalObject]]:=(LocalObject;localObjectDataFile[lo,LocalObjects`PathName@lo])
localObjectDataFile[lo_,dir_]:=localObjectDataFile[lo,LocalObjects`AuxPathName[lo],LocalObjects`BundlePathName[lo]]/;DirectoryQ[dir]
localObjectDataFile[_,file_]:=file/;FileExistsQ[file]
localObjectDataFile[_,file_,_]:=file/;FileExistsQ[file]
localObjectDataFile[_,_,file_]:=file/;FileExistsQ[file]
localObjectDataFile[___]:=Missing["NotAvailable"]


marketplacebasedResourceQ[info_]:=Head[Lookup[info,"RepositoryLocation"]]===System`URL
userdefinedResourceQ[info_]:=MatchQ[Lookup[info,"RepositoryLocation", None],(_LocalObject|None)]

verifyReviewerPermissions[co_]:=If[TrueQ[checkReviewerPermissions[co]],
		co
		,
		If[ListQ[setReviewerPermissions[co]],
			co,
			(Message[ResourceSubmit::appperms,co];Throw[$Failed])
		]
	]

$ResourceSystemAdminUser =
  If[ ! StringQ @ $ResourceSystemAdminUser,
      "marketplace-admin@wolfram.com",
      $ResourceSystemAdminUser
  ];

setReviewerPermissions[co:HoldPattern[_CloudObject]]:=Block[{res},
	res=Quiet[SetPermissions[co,$ResourceSystemAdminUser->"Read"]];
	If[ListQ[res],
		res
		,
		$Failed
    ]
]

checkReviewerPermissions[co_]:=checkreviewerPermissions[Quiet[CloudObjectInformation[co,"Permissions"]]]

checkreviewerPermissions[l_List]:=True/;MemberQ[Lookup[l, "All",{}],"Read"]
checkreviewerPermissions[l_List]:=True/;MemberQ[Lookup[l, $ResourceSystemAdminUser,{}],"Read"]

checkreviewerPermissions[_]:=False

importlocal[HoldPattern[lo_LocalObject]]:=With[{objectinfo=LocalObjects`getLocalObject[lo]},
	If[AssociationQ[objectinfo],
		Switch[Lookup[objectinfo,"Type"],
			"Export",
			Import[lo],
			"Expression"|Expression,
			Get[lo],
			_,
			Import[lo]
		],
		$Failed
	]
]

importlocal[File[file_]]:=Import[file]

importlocal[file_]:=Import[file]

getResourceType[info_Association, default_:Missing["NotAvailable"]]:=
	getresourceType[Lookup[info,"ResourceType",default]]
getResourceType[__]:=Missing[]

getresourceType[expr_]:=expr

stringhash[expr_]:=IntegerString[Hash[expr],16]

allSortingProperties[]:=DeleteDuplicates[Flatten[sortingProperties/@$availableResourceTypes]]

sortingProperties[_]:={}

uRL[str_String]:=URL[str]
uRL[expr_]:=expr

mapAt[as_, rules : {_Rule ..}] := Fold[mapAt[#2[[2]], #1, #2[[1]]] &, as, rules]
mapAt[f_,as_,key_]:=MapAt[f,as,key]/;KeyExistsQ[as,key]
mapAt[_,as_,_]:=as

attributeCheck[info_,att_]:=MemberQ[info["Attributes"],att]/;KeyExistsQ[info, "Attributes"]
attributeCheck[__]:=False

keyExistsQ[as_Association, {first_, rest__}]:=If[KeyExistsQ[as, first],
	If[AssociationQ[as[first]],
		keyExistsQ[as[first],{rest}],
		False
	]
	,
	False
]

keyExistsQ[as_Association, {first_}]:=KeyExistsQ[as, first]
keyExistsQ[___]:=False

stringFileExistsQ[str_String]:=FileExistsQ[str]
stringFileExistsQ[___]:=False

 
noCacheImportBytes[bytes_,fmt_]:=
	nocacheImportBytes[bytes,CreateFile[],fmt]   

nocacheImportBytes[bytes_,file_,fmt_]:=With[{content=Import[Export[file,bytes,"Byte"],fmt]},
	DeleteFile[file];
	If[FailureQ[content],
		$Failed,
		content
	]
]


takeFunctionOptions[f_, rest___?OptionQ]:=Sequence@@FilterRules[Flatten[{rest}], Keys[Options[f]]]
takeFunctionOptions[f_, expr_,rest___?OptionQ]:=takeFunctionOptions[f,rest]

optionMismatchQ[info_, opts_]:=With[{as=Association[opts]},
	versionOptionMismatchQ[info,as]||
	cloudOptionMismatchQ[info,as]||
	wlVersionOptionMismatchQ[info,as]
]
	
versionOptionMismatchQ[info_,opts_]:=((opts["Version"]=!=Automatic)&&(opts["Version"]=!=None)&&(opts["Version"]=!=info["Version"]))/;KeyExistsQ[opts,"Version"]
versionOptionMismatchQ[___]:=False
cloudOptionMismatchQ[info_,opts_]:=(
	(opts[System`ResourceSystemBase]=!=Automatic)&&
	(opts[System`ResourceSystemBase]=!=resourcerepositoryBase[info]))/;KeyExistsQ[opts,System`ResourceSystemBase]
cloudOptionMismatchQ[___]:=False
wlVersionOptionMismatchQ[info_,opts_]:=wlversionOptionMismatchQ[
	info["WolframLanguageVersionRequired"],
	opts["WolframLanguageVersion"]]/;(KeyExistsQ[opts,"WolframLanguageVersion"]&&KeyExistsQ[info,"WolframLanguageVersionRequired"])
wlVersionOptionMismatchQ[___]:=False

wlversionOptionMismatchQ[roRequirement_String,userReq_String]:=newerversionQ[{roRequirement,userReq}]
wlversionOptionMismatchQ[__]:=False

$DownloadResourceFromFilesURL=False;

makeFilesURL[co_] := Block[{url = URLParse[co], id},
	id=System`CloudObjectInformation[co, "UUID"];
	If[StringQ[id],
		url["Path"] = {"", "files", id}
	];
	URLBuild[url]
  	]/;$DownloadResourceFromFilesURL

makeFilesURL[co_]:=First[co]

localObjectPathName[lo_LocalObject]:=LocalObjects`PathName[lo]
localObjectPathName[dir_String]:=dir/;DirectoryQ[dir]
localObjectPathName[file_String]:=FileNameDrop[file]
  	
deleteFile[lo_LocalObject]:=DeleteFile[lo]/;fileExistsQ[lo]
deleteFile[dir_String]:=DeleteDirectory[dir,DeleteContents->True]/;DirectoryQ[dir]
deleteFile[file_]:=DeleteFile[file]/;fileExistsQ[file]
	
importLocalObject[lo_LocalObject,_]:=Import[lo]
importLocalObject[dir_String,fmt_]:=With[{file=FileNameJoin[{dir,"data"<>dataFileExtension[fmt]}]},
	importLocalObject[file, fmt]]/;DirectoryQ[dir]
importLocalObject[file_String,_]:=Import[file]/;FileExistsQ[file]&&!DirectoryQ[file]
importLocalObject[file_String,fmt_String]:=With[{f=file<>"."<>fmt},
	If[FileExistsQ[f],
		Import[f],
		$Failed
	]
]

acquiredResourceQ[_]:=False

mustAcquireResourceQ[info_Association]:=mustAcquireResourceQ[info["UUID"]]
mustAcquireResourceQ[id_String]:=False/;acquiredResourceQ[id]
mustAcquireResourceQ[id_String]:=mustAcquireResourceQ[id,getResourceInfo[id]]

mustAcquireResourceQ[id_String,_]:=False/;acquiredResourceQ[id]
mustAcquireResourceQ[id_,info_Association]:=mustAcquireResourceQ[info["ResourceType"],id,info]
mustAcquireResourceQ[_,id_,info_Association]:=TrueQ[info["MustAcquire"]]

costPerUseResourceQ[info_Association]:=costPerUseResourceQ[info["UUID"]]
costPerUseResourceQ[id_String]:=costPerUseResourceQ[id,getResourceInfo[id]]

costPerUseResourceQ[id_,info_Association]:=costPerUseResourceQ[info["ResourceType"],id,info]
costPerUseResourceQ[_,id_,info_Association]:=costperUseResourceQ[info["PricingInformation"]]
costPerUseResourceQ[___]:=False

costperUseResourceQ[as_Association]:=costperuseResourceQ[as["BaseUsagePrice"]]/;Quiet[as["MarketplaceBilling"]==="PerUse"]
costperUseResourceQ[___]:=False
costperuseResourceQ[0|_Missing]:=False
costperuseResourceQ[_]:=True

dataFileExtension["Binary"]:=".bin"
dataFileExtension[Automatic]:=""
dataFileExtension[fmt_String]:="."<>fmt
dataFileExtension[_]:=""

Attributes[noUpdate]={HoldAll};
noUpdate[expr_]:=Quiet[expr, {ResourceObject::updav,ResourceObject::updav2}]

verifyDownloadHash[file_String, dlinfo_Association]:=
	verifyDownloadHash[FileHash[file,"MD5",All,"HexString"],dlinfo["Hash"]]/;FileExistsQ[file]&&KeyExistsQ[dlinfo,"Hash"]

verifyDownloadHash[HoldPattern[resp_HTTPResponse], dlinfo_Association]:=
	verifyDownloadHash[Hash[resp["BodyByteArray"],"MD5","HexString"],dlinfo["Hash"]]/;KeyExistsQ[dlinfo,"Hash"]
	
verifyDownloadHash[hash_String,hash_String]:=Null
verifyDownloadHash[hash1_String,hash2_String]:=(Message[ResourceData::baddl];Throw[$Failed])


notebookObjectQ[ nb_NotebookObject ] :=
  FailureQ @ NotebookInformation @ nb === False;

notebookObjectQ[ ___ ] :=
  False;
  
End[] (* End Private Context *)

EndPackage[]
