(* Wolfram Language Package *)


BeginPackage["ResourceSystemClient`"]

Begin["`Private`"] (* Begin Private Context *) 
$ResourceDeployProgressIndicator=0;

resourceElementInfoFile[id_, elem_]:=localObject@FileNameJoin[{resourceDirectory[id],"download",URLEncode[ToString[elem]],"metadata"}]

resourceCopyDirectory[id_,rest___]:=resourcecopyDirectory[resourceDirectory[id],rest]
resourcecopyDirectory[dir_,format_]:=resourcecopyDirectory[dir,format, Automatic]
resourcecopyDirectory[dir_,format_, elem_]:=resourcecopydirectory[dir,{format,$SystemWordLength}, elem]/;$SystemWordLength=!=64
resourcecopyDirectory[dir_,format_, elem_]:=resourcecopydirectory[dir,format, elem]
resourcecopydirectory[dir_,format_, Automatic]:=FileNameJoin[{dir,"download","Automatic",IntegerString[Hash[format],16]}]
resourcecopydirectory[dir_,format_, elem_]:=FileNameJoin[{dir,"download",URLEncode[ToString[elem]],IntegerString[Hash[format],16]}]

resourceCopyInfoFile[id_,format_]:=resourceCopyInfoFile[id,format, Automatic]
resourceCopyInfoFile[id_,format_, elem_]:=resourcecopyinfoFile[id,{format,$SystemWordLength}, elem]/;$SystemWordLength=!=64
resourceCopyInfoFile[id_,format_, elem_]:=resourcecopyinfoFile[id,format, elem]

resourcecopyinfoFile[id_,format_, Automatic]:=localObject@FileNameJoin[{resourceDirectory[id],"download","Automatic",IntegerString[Hash[format],16],"metadata"}]
resourcecopyinfoFile[id_,format_, elem_]:=localObject@FileNameJoin[{resourceDirectory[id],"download",URLEncode[ToString[elem]],IntegerString[Hash[format],16],"metadata"}]

resourcecopyInfoFile[copydir_,format_]:=localObject@FileNameJoin[{copydir,"metadata"}]

resourceElementDirectory[id_, elem_]:=FileNameJoin[{resourceDirectory[id],"download",URLEncode[ToString[elem]]}]

getElementInfo[id_, elem_]:=getelementInfo[resourceElementInfoFile[id, elem]]
getelementInfo[file_]:=Get[file]/;FileExistsQ[file]
getelementinfo[_]:=None

getElementFormat[rtype_,id_,elem_,loc_]:=repositoryElementFormat[rtype,id,elem,loc,getelementFormat[resourceElementInfoFile[id, elem]]]
getelementFormat[file_]:=With[{as=Get[file]},
	chooseformat[Lookup[as,"Formats",Lookup[as,"Format"]]]]/;fileExistsQ[file]
getelementFormat[_]:=Automatic

getElementCopyInfo[id_, format_, elem_]:=getelementCopyInfo[resourceCopyInfoFile[id,format, elem]]
getelementCopyInfo[file_]:=Get[file]/;FileExistsQ[file]
getelementCopyInfo[_]:=None


repositoryElementFormat["NeuralNet",_,_,_URL,Automatic]:=Automatic (* Delete in 12.1, it's in NNR *)
repositoryElementFormat["NeuralNet",_,_,_,Automatic]:="WLNet" (* Delete in 12.1, it's in NNR *)
repositoryElementFormat[__,fmt_]:=fmt

chooseformat[formats_List]:=First[formats];
chooseformat[format_String]:=format;
chooseformat[_]:=Automatic;

storeElementFormat[id_, elem_, format_,listQ_:False]:=storeelementFormat[resourceElementInfoFile[id, elem],format, listQ]
storeelementFormat[file_, format_, listQ_]:=storeelementformat[Get[file], file,format,listQ]/;fileExistsQ[file]
storeelementformat[as_Association, file_,Missing["Deleted"], _]:=Block[{newas=as},
	newas["Format"]=.;
	newas["Formats"]=.;
	Put[newas,file]	
]

storeelementformat[as_Association, file_,format_, listQ_]:=Block[{newas=as},
	If[listQ,
		newas["Format"]=format,
		newas["Formats"]=DeleteDuplicates[Flatten[{format,newas["Formats"]}]]
	];
	Put[newas,file]	
]

storeelementFormat[file_, Missing["Deleted"], False]:=Null
storeelementFormat[file_, format_, False]:=(
	createParentDirectory[file];
	Put[Association["Format"->format],file]
	)
storeelementFormat[file_, format_, True]:=(
	createParentDirectory[file];
	Put[Association["Formats"->Flatten[{format}]],file]
	)
storeelementformat[_,file_, format_, False]:=Put[Association["Format"->format],file]
storeelementformat[_,file_, format_, True]:=Put[Association["Formats"->Flatten[{format}]],file]

getAllElementFunction[info_]:=getAllElementFunction[Lookup[info,"UUID"],Lookup[info,"ContentElements"]]
getAllElementFunction[id_String,elems_List]:=With[{as=AssociationMap[getElementFunction[id, #]&, elems]},
	DeleteCases[as,$Failed]
	]
getAllElementFunction[__]:=Association[]

getElementFunction[id_, elem_]:=getelementFunction[resourceElementInfoFile[id, elem]]
getelementFunction[file_]:=getelementfunction[Get[file]]/;fileExistsQ[file]
getelementFunction[_]:=$Failed
getelementfunction[as_Association]:=Lookup[as,"ContentFunction",$Failed]
getelementfunction[_]:=$Failed

storeContentFunctions[id_, res_]:=storecontentFunctions[id,Lookup[res,"ContentElementFunctions"]]
storecontentFunctions[id_,as_Association]:=AssociationMap[storeElementFunction[id,#[[1]],#[[2]]]&,as]
storecontentFunctions[id_,str_String]:=With[{unc=Quiet[Uncompress[str]]},
	If[AssociationQ[str],storeContentFunctions[id, unc]]
]
storeElementFunction[id_, elem_, func_]:=storeelementFunction[resourceElementInfoFile[id, elem],func]
storeelementFunction[file_, func_]:=storeelementfunction[Get[file], file,func]/;fileExistsQ[file]
storeelementfunction[as_Association, file_,func_]:=Block[{newas=as},
	newas["ContentFunction"]=func;
	Put[newas,file]	
]
storeelementFunction[file_, func_]:=(createParentDirectory[file];Put[Association["ContentFunction"->func],file])
storeelementfunction[_,file_, func_]:=Put[Association["ContentFunction"->func],file]


multipartResourceQ[info_]:=AssociationQ[Lookup[info,"ContentElementLocations", None]]||
	AssociationQ[Lookup[info,"InformationElements", None]]
cloudStoredQ[info_]:=MatchQ[Lookup[info,"ContentElementLocations", None],_CloudObject]/;!multipartResourceQ[info]
cloudStoredQ[info_]:=MatchQ[Values[Lookup[info,"ContentElementLocations", None]],{(_CloudObject)...}]
cloudStoredQ[info_]:=False
cloudStoredQ[info_,Automatic]:=cloudStoredQ[info]
cloudStoredQ[info_,elems_List]:=And@@(cloudStoredQ[info,#]&/@elems)
cloudStoredQ[info_,elem_]:=With[{locations=Lookup[info,"ContentElementLocations"]},
	If[AssociationQ[locations],
		If[MatchQ[Lookup[locations, elem],(_CloudObject|_URL)],
			True,False]
		,
		False
	]]

localStoredQ[info_]:=MatchQ[Lookup[info,"ContentElementLocations", None],_LocalObject|_File]/;!multipartResourceQ[info]
localStoredQ[info_]:=MatchQ[Values[Lookup[info,"ContentElementLocations", None]],{(_LocalObject|_File)...}]
localStoredQ[info_,Automatic|All]:=localStoredQ[info]
localStoredQ[info_,elems_List]:=And@@(localStoredQ[info,#]&/@elems)
localStoredQ[info_, elem_]:=localstoredQ[info,Lookup[info,"ContentElementLocations"],elem]

localstoredQ[info_,locations_Association,elem_]:=MatchQ[Lookup[locations, elem],
	(_LocalObject?fileExistsQ|None|_String?FileExistsQ)]/;KeyExistsQ[locations,elem]
localstoredQ[info_,locations_Association,elem_]:=MemberQ[Lookup[info,"ContentElements",{}],elem]&&!MemberQ[Keys[locations],elem]
localstoredQ[info_,HoldPattern[_Missing|_CloudObject],elem_]:=False
localstoredQ[info_,_,elem_]:=MemberQ[Lookup[info,"ContentElements",{}],elem]
localstoredQ[___]:=False

contentElementQ[_,Automatic]=True;
contentElementQ[info_,l_List]:=And@@(MemberQ[Lookup[info,"ContentElements",{}],#]&/@l)
contentElementQ[info_,elem_]:=MemberQ[Lookup[info,"ContentElements",{}],elem]

cacheResourceInfoWithElementFunctions[rtype_,id_, info_, dir_]:=Block[{cfs, newinfo=info},
	If[KeyExistsQ[newinfo,"ContentElementFunctions"],
		cfs=newinfo["ContentElementFunctions"];
		newinfo=KeyDrop[newinfo,"ContentElementFunctions"]
	];
	If[DirectoryQ[dir],
		newinfo=ResourceSystemClient`Private`updatecachedresourceinfo[dir, id, newinfo]
		,
		createDirectory[dir];
		newinfo=Append[newinfo,"DownloadedVersion"->None];
		Put[Append[newinfo,"DownloadedVersion"->None],resourceInfoFile[id]];
	];
	storecontentFunctions[id, cfs];
	newinfo
]


updateResourceInfoElements[rtype_, id_, info0_,as_]:=Block[
	{info=info0, elem=Lookup[as,"Element",Automatic],
		location=First[Flatten[{Lookup[as,"Locations",Lookup[as,"Location",Automatic]]}]],
		formats=Lookup[as,"Formats",Lookup[as,"Format",Automatic]]
	},
	If[elem===Automatic,
		If[location===Missing["Deleted"],
			info["ContentElementLocations"]=.,
			info["ContentElementLocations"]=location
		]
		,
		If[AssociationQ[info["ContentElementLocations"]],
			If[location===Missing["Deleted"],
				info["ContentElementLocations",elem]=Missing[],
				info["ContentElementLocations",elem]=location
			],
			If[location===Missing["Deleted"],
				info["ContentElementLocations"]=.,
				info["ContentElementLocations"]=Association[elem->location]
			]
		]
	];
	If[ListQ[formats],
        storeElementFormat[id, elem, formats, True]
        ,
        If[StringQ[formats]||formats===Automatic||MissingQ[formats],
	        storeElementFormat[id, elem, formats, False]
		]
	];
	info
]

resourceElementStorageSizeLimit[_]:=2*10^5;(* Bytes *)

standardizeContentMetadataWithElements[_,_, info_]:=(Message[ResourceObject::twodef];Throw[$Failed])/;KeyExistsQ[info,"Content"]&&KeyExistsQ[info,"ContentLocation"]

standardizeContentMetadataWithElements[rtype_,id_, info_]:=Block[{elements, locations,
	functions,moreinfo, contentelements, default=Lookup[info,"DefaultContentElement",Automatic],storedelements},
	locations=DeleteCases[Lookup[info,"ContentElementLocations",Association[]],None];
	contentelements=Lookup[info,"ContentElements",Association[]];
	If[ListQ[contentelements],
		If[KeyExistsQ[info,"ContentValues"],
			contentelements=info["ContentValues"];
			If[!AssociationQ[contentelements],
				contentelements=Association[]
			],
			contentelements=Association[]
		]
	];
	If[!AssociationQ[contentelements],
		Message[ResourceObject::invas,"ContentElements"]; Throw[$Failed]
	];
	If[!AssociationQ[locations],
		Message[ResourceObject::invas,"ContentElementLocations"]; Throw[$Failed]
	];
	If[KeyExistsQ[info,"Content"],
		If[default=!=Automatic,
			Message[ResourceObject::twodef];Throw[$Failed]
		];
		If[KeyExistsQ[contentelements,"Content"],
			Message[ResourceObject::twocont];Throw[$Failed]
		];
		default="Content";
		contentelements=Prepend[contentelements,"Content"->info["Content"]]			
	];
	If[KeyExistsQ[info,"ContentLocation"],
		If[default=!=Automatic,
			Message[ResourceObject::twodef];Throw[$Failed]
		];
		If[KeyExistsQ[locations,"Content"],
			Message[ResourceObject::twocont];Throw[$Failed]
		];
		default="Content";
		locations=Prepend[locations,"Content"->info["ContentLocation"]]			
	];
	functions=Lookup[info,"ContentElementFunctions",Association[]];
	If[!AssociationQ[functions],
		Message[ResourceObject::invas,"ContentElementFunctions"]; Throw[$Failed]
	];
	
	If[KeyExistsQ[info,"InformationElements"],
		moreinfo=Lookup[info,"InformationElements",Association[]];
		If[!AssociationQ[moreinfo],
			Message[ResourceObject::invas,"InformationElements"]; Throw[$Failed]
		]
		,
		moreinfo=Select[contentelements,(ByteCount[#]<resourceElementStorageSizeLimit[rtype])&];
		contentelements=KeyComplement[{contentelements,moreinfo}];
	];
		
	elements=Flatten[Keys/@{contentelements,locations,functions,moreinfo}];
	If[!DuplicateFreeQ[elements],
		Message[ResourceObject::elemcon,Cases[Tally[elements], {el_, _?(# > 1 &)} :> el, {1}]];Throw[$Failed]
	];
	elements=Sort[elements];
	storedelements=Sort[Flatten[Keys/@{contentelements,locations}]];
	If[!MemberQ[Append[elements, Automatic],default],
		Message[ResourceObject::invdefa,default]; Throw[$Failed]
	];
	Association[
		KeyDrop[info,{"ContentElements","ContentElementLocations","Content","ContentLocation","InformationElements"}]
		,
		Association[
			"ResourceType"->rtype,
			"ContentElements"->elements,
			"ContentElementLocations" -> AssociationMap[Lookup[locations,#,None]&,storedelements],
			"RepositoryLocation"->None,
			"InformationElements"->moreinfo,
			"ContentValues"->contentelements
		]
		,
		standardizeContentMetadataContentInfo[rtype,default, locations, contentelements,moreinfo]
	]
]

standardizeContentMetadataWithElements[___]:=(Message[ResourceObject::nocont]; Throw[$Failed])

saveresourceobjectwithelements[info_]:=saveMultipartResource[info]/;AssociationQ[Lookup[info,"ContentElementLocations"]]

saveresourceobjectwithelements[info_]:=saveresourceobjectwithcontent[info]/;Lookup[info,"ContentElementLocations"]==None

saveresourceobjectwithelements[info_]:=saveResourceObjectWithLocation[Lookup[info,"ResourceType"],info]/;KeyExistsQ[info,"ContentElementLocations"]

saveresourceobjectwithelements[info_]:=saveresourceobjectwithcontent[info]/;KeyExistsQ[info,"Content"]

saveresourceobjectwithelements[info_]:=(Message[ResourceObject::nocont];Throw[$Failed])

saveresourceobjectwithcontent[info0_]:=Block[{content, id=Lookup[info0,"UUID",Throw[$Failed]], 
	info=KeyDrop[info0,"Content"]},
	content=info0["Content"];
	info["RepositoryLocation"]=localObject[resourceCacheDirectory[]];
	cacheresourceinfo[info];
	
	ResourceSystemClient`Private`repositoryresourcedownload[Lookup[info,"ResourceType"],Lookup[info,"UUID"],Join[
		KeyTake[info,{"UUID","ContentElementFunctions","ResourceType","Version"}],
		Association["Content"->content,"ContentFormat"->"WDF"]]];
	id
	]
	
saveResourceObjectWithLocation[rtype_,info0_]:=Block[{id=Lookup[info0,"UUID",Throw[$Failed]],format, 
	info=info0,copyinfo, loc=Lookup[info0,"ContentElementLocations",Throw[$Failed]]},
	If[!MatchQ[loc,(_CloudObject|_LocalObject|_File|_?fileExistsQ)],Message[ResourceObject::invloc];Throw[$Failed]];
	info["RepositoryLocation"]=localObject[resourceCacheDirectory[]];
	cacheresourceinfo[info];
	saveresourceObjectWithLocation[rtype,id, info, loc];
	id
]
	
saveresourceObjectWithLocation[rtype_,id_, info_, loc_, elem_:Automatic]:=Block[{copyinfo, fmt},
	fmt=Lookup[Lookup[info,"Formats",Lookup[info,"Format",<||>]],elem,Automatic];
	copyinfo=resourcedownloadInfo[rtype,id, info,fmt/.Automatic->defaultResourceTypeDownloadFormat[rtype],loc];
	If[AssociationQ[copyinfo],
        Put[copyinfo,resourceCopyInfoFile[id,Automatic,elem]]
        ,
        Throw[$Failed]
	];
	storecontentFunctions[id,Lookup[info,"ContentElementFunctions"]];
	storeDownloadVersion[id,Association["Version"->None],
		Association["Element"->elem,
			resourceDownloadStorageInfo[rtype,loc,fmt]]];
	id
	]
	
saveMultipartResource[info_]:=saveMultipartResource[Lookup[info,"ResourceType"],info]

saveMultipartResource[rtype_,info0_]:=Block[{
	id=Lookup[info0,"UUID",Throw[$Failed]],format, 
	info=KeyDrop[info0,"ContentValues"],copyinfo, 
	locations=Lookup[info0,"ContentElementLocations",Throw[$Failed]], 
	rawelements=Lookup[info0,"ContentValues",Throw[$Failed]]
	},
	info["RepositoryLocation"]=localObject[resourceCacheDirectory[]];
	cacheresourceinfo[info];
	
	AssociationMap[savemultipartResource[rtype,info,#]&, rawelements];
	AssociationMap[savemultipartResourceLocation[rtype,id,info, #]&, DeleteCases[locations, None]];

	storecontentFunctions[id,Lookup[info,"ContentElementFunctions"]];
	id
	]
	

savemultipartResource[rtype_,info_,_[elem_,value_]]:=ResourceSystemClient`Private`repositoryresourcedownload[rtype,
	Lookup[info,"UUID"],
	Join[
		KeyTake[info,{"UUID","ResourceType","Version"}],
		Association["Format"->"WDF","Content"->value,"ContentFormat"->"WDF","Element"->elem]]
	]

savemultipartResourceLocation[rtype_,id_,info_,_[elem_,value_]]:=saveresourceObjectWithLocation[rtype,id, info, value, elem]




bundleResourceObjectWithElementFunctions[rtype_,id_, localinfo_]:=Block[{fullinfo=localinfo},
	fullinfo["ContentElementFunctions"]=getAllElementFunction[localinfo];	
	fullinfo
]


clouddeployResourceInfoWithElements[rtype_,id_,  localinfo_, newinfo_, rest___]:=Block[{cloudinfo=localinfo,infoco},
	cloudinfo["ContentValues"]=.;
	cloudinfo["ResourceLocations"]={CloudObject[cloudpath[resourceDirectory[id]]]};
	cloudinfo["ContentElementLocations"]=Join[Lookup[cloudinfo,"ContentElementLocations",Association[]],Lookup[newinfo,"ContentElementLocations",Association[]]];
	cloudinfo["RepositoryLocation"]=None;
	infoco=cloudResourceDirectoryObject[FileNameJoin[{StringTake[id,3], id,"metadata"}]];
	CloudPut[Join[cloudinfo,newinfo],infoco,takeFunctionOptions[CloudPut,rest] ];
	{cloudinfo,infoco}
]


cloudexportResourceContentElement[id_, elem_, elemlocation_, rest___]:=Block[{targetCO,res},
	targetCO=CloudObject[cloudpath[FileNameJoin[{resourceCopyDirectory[id,"MX",elem],"data"}]]];
	With[{content=Import[elemlocation]},
		If[FailureQ[content],
			Message[ResourceObject::depcf, elem];
			Throw[$Failed]
		];
		res=CloudExport[content,"MX",targetCO,takeFunctionOptions[CloudExport,rest]];
	];
	If[res===$Failed,
		Message[ResourceObject::depcf, elem];
		Throw[$Failed]
	];
	targetCO
]

$DeployDownloadInfo=False;

cloudDeployResourceContentElements[id_, info_,rest___]:=Block[{$ResourceDeployProgressIndicator, elems=Lookup[info,"ContentElements"], temp, res},
	$ProgressIndicatorContent=progressPanel["Deploying the resource content\[Ellipsis]"];
	temp=printTemporaryFrontEnd[Dynamic[$ProgressIndicatorContent]];
	res=DeleteMissing[AssociationMap[cloudDeployResourceContentElement[id, info,#, rest]&,Lookup[info,"ContentElements"]]];
	clearTempPrint[temp];
	Association["ContentElementLocations"->res]
]/;userdefinedResourceQ[info]

cloudDeployResourceContentElements[_,info_,___]:=Association[]/;marketplacebasedResourceQ[info]

cloudDeployResourceContentElement[id_,info_,elem_, rest___]:=Block[{
	cloudcontentlocation=clouddeployResourceContentElement[id,info,elem, resourceElementDirectory[id, elem],rest],
	eleminfo=getElementInfo[id, elem], format,copyinfo, opts},
	format=Lookup[cloudcontentlocation,"Format",None];
	(* element info *)
	opts=takeFunctionOptions[CloudPut,rest];
	If[AssociationQ[eleminfo],
		If[format=!=None,eleminfo["Format"]=format];
		CloudPut[eleminfo,cloudpath[resourceElementInfoFile[id, elem]],opts]
	];
	(* copy info *)
	If[format=!=None&&TrueQ[$DeployDownloadInfo],
		copyinfo=getElementCopyInfo[id, format, elem];
		copyinfo["Location"]=cloudcontentlocation["Location"];
		CloudPut[copyinfo,cloudpath[resourceCopyInfoFile[id,format, elem]],opts]
	];
	$ResourceDeployProgressIndicator++;
	Lookup[cloudcontentlocation,"Location"]
]/;MemberQ[$localResources,id]

cloudDeployResourceContentElement[id_,info_,elem_, rest___]:=Block[{targetCO,res,opts},
	opts=takeFunctionOptions[CloudExport,rest];
	targetCO=CloudObject[cloudpath[FileNameJoin[{resourceCopyDirectory[id,"MX",elem],"data"}]]];
	res=CloudExport[info["ContentValues",elem],"MX",targetCO, opts];
	If[res===$Failed,
		Message[ResourceObject::depcf, elem];
		Throw[$Failed]
	];
	targetCO
]/;keyExistsQ[info,{"ContentValues",elem}]

cloudDeployResourceContentElement[id_,info_,elem_, rest___]:=With[{location=info["ContentElementLocations", elem]},
	Switch[location,
		_CloudObject|_URL, location,
		_LocalObject, cloudexportResourceContentElement[id, elem, location, rest],
		File[__],cloudexportResourceContentElement[id, elem, First[location], rest],
		_String, 
			If[FileExistsQ[location],
				cloudexportResourceContentElement[id, elem, location, rest]
				,
				Missing[]
			],
		_,Missing[]
	]
]/;keyExistsQ[info,{"ContentElementLocations",elem}]

cloudDeployResourceContentElement[_,_,elem_, ___]:=Missing[]


clouddeployResourceContentElement[id_,info_,elem_,dir_, rest___]:=clouddeployresourceContentElement[id, info, elem, dir, Quiet[Lookup[info["ContentElementLocations"],elem]],rest]

clouddeployresourceContentElement[id_, info_, elem_, dir_, elemlocation:(_LocalObject|_String), rest___]:=
	clouddeployresourcecontentElement[id, info, elem, dir, elemlocation,rest]/;fileExistsQ[elemlocation]

clouddeployresourceContentElement[id_, info_, elem_, dir_, elemlocation:HoldPattern[_CloudObject], rest___]:=
	Association[{"Location"-> elemlocation}]

clouddeployresourcecontentElement[id_, info_, elem_, dir_, elemlocation_, rest___]:=Block[{targetCO,format=getElementFormat[info["ResourceType"],id, elem,elemlocation],res},
	targetCO=CloudObject[cloudpath[FileNameJoin[{resourceCopyDirectory[id,format,elem],"data"}]],takeFunctionOptions[CloudObject,rest]];
	res=CopyFile[elemlocation,targetCO];
	If[res===$Failed,
		Message[ResourceObject::depcf, elem];
		Throw[$Failed]
	];
	Association[{"Format"->format,"Location"-> targetCO}]
]

clouddeployresourceContentElement[___]:=Association[]




typesetElementStorageLocation[rtype_,id_]:=DynamicModule[{typesetstorage=typesetstorageLocation, info},
	Dynamic[info=resourceInfo[id];
		If[AssociationQ[info],
			typesetstorage[Lookup[info,"ContentElementLocations",None]]
			,
			Missing["NotAvailable"]
		]
	]
]

typesetstorageLocation[as_Association]:=Row[Riffle[DeleteDuplicates[typesetstorageLocation/@Values[as]]," "]]
typesetstorageLocation[l_System`LocalObject]="Local"
typesetstorageLocation[HoldPattern[_CloudObject]]="Cloud"
typesetstorageLocation[{HoldPattern[_CloudObject]..}]="Cloud"
typesetstorageLocation[str_String]:=Text[Style["Repository", 8, Black], Background -> Red]/;str=System`$ResourceSystemBase
typesetstorageLocation[str_String]=str
typesetstorageLocation[m_Missing]=m
typesetstorageLocation[None]=Style[None,Gray]
typesetstorageLocation[expr_]=Null

readResourceElementContent[id_,info_,rest___]:=Which[
	MemberQ[$localResources,id],readResourceElementLocal[id, info, rest],
	MemberQ[$cloudResources,id],readResourceElementCloud[id, info, rest],
	True,readElementFromResourceSystem[id, info, rest]
]

readResourceElementCloud[id_, info_, rest___]:=(printTempOnce[$progressID];
	CloudEvaluate[readResourceElementLocal[id, info, rest]])

readResourceElementLocal[id_, info_, elem_,rest___]:=If[(Lookup[info,"DownloadedVersion",None]=!=None||localStoredQ[info, elem]),
		readResourceLocal[id, info,elem, rest],
		printTempOnce[$progressID];
		readElementFromResourceSystem[id, info, elem,rest]
	]


readResourceLocal[id_, info_, elem_,rest___]:=Block[{copyinfo, func},
    func=resourceDataPostProcessFunction[rest];
    func@readresourceLocal[id,info,elem,rest]
]

readresourceLocal[id_,info_,l_List, rest___]:=With[{elems=Lookup[info,"ContentElements",{}]},
	If[Complement[l,elems]=!={},
		Message[ResourceData::invelem];Throw[$Failed]
	];
	AssociationMap[readresourceLocal[id,info,#,rest]&,l]
]

readresourceLocal[id_,info_,All, rest___]:=With[{elems=Lookup[info,"ContentElements",{}]},
	readresourceLocal[id,info,elems, rest]
]
readresourceLocal[id_,info_,Automatic, rest___]:=readresourceLocal[id,info,info["DefaultContentElement"], rest]/;KeyExistsQ[info,"DefaultContentElement"]
readresourceLocal[id_,info_,Automatic, rest___]:=readresourcelocal[getElementFormat[info["ResourceType"],id,Automatic,Automatic],info["ContentElementLocations"]]
readresourceLocal[id_,info_,elem_, ___]:=(Message[ResourceData::invelem,elem];Throw[$Failed])/;!MemberQ[info["ContentElements"],elem]
readresourceLocal[id_,info_,elem_, ___]:=info["InformationElements",elem]/;KeyExistsQ[Lookup[info,"InformationElements",Association[]], elem]

readresourceLocal[id_,info_,elem_String,rest___]:=With[{loc=info["ContentElementLocations",elem]},
	readresourcelocal[
	getElementFormat[info["ResourceType"],id,elem,loc],loc
	]]/;Quiet[TrueQ[KeyExistsQ[Lookup[info,"ContentElementLocations",Association[]],elem]]]

$reaquiredResource=False;

readresourceLocal[id_,info_,elem_String,rest___]:=With[{contentfunction=getElementFunction[id, elem]},
	If[contentfunction=!=$Failed,
		produceResourceElementContent[id, info, elem, contentfunction, rest]
		,
		If[$reaquiredResource,
			(Message[ResourceData::invelem1,elem];Throw[$Failed]),
			printTempOnce[$progressID];
			resourceAcquire[id, True];
			Block[{$reaquiredResource=True},
				readresourceLocal[id,resourceInfo[id],elem,rest]
			]]]]

produceResourceElementContent[id_, info_, elem_, contentfunction_, rest___]:=Block[{reqelems, default, cf=contentfunction},
	reqelems=Intersection[Cases[contentfunction, Slot[e_] :> e, Infinity, Heads -> True], Append[Lookup[info,"ContentElements",{}],"Content"]];
	If[MemberQ[reqelems,"Content"]&&!MemberQ[info["ContentElements"],"Content"],
		default=info["DefaultContentElement"];
		If[StringQ[default],
			reqelems=reqelems/."Content"->default;
			cf=ReplaceAll[cf,Slot["Content"]->Slot[default]]
			,
			Return[$Failed]
		]	
	];
	loadResourceType["DataResource"];
	If[Length[reqelems]>0,
		If[reqelems==={"Content"},
			cf[Association["Content"->DataResource`Private`resourcedatawithProgress[{id,info}, "Content"]]]
			,
			cf[DataResource`Private`resourcedatawithProgress[{id,info}, reqelems]]
		]
		,
		cf[Association[]]
	]
]


readElementFromResourceSystem[id_, info_,elem_, rest___]:=Block[{res},
	If[cacheResourceQ[info],
		ResourceSystemClient`ResourceDownload[id, Association["Element"->elem,"ResponseMethod"->"Download"]];
		readResourceLocal[id, resourceInfo[id],elem,rest]
		,
		readresource[id, Association["Element"->elem,"ResponseMethod"->"Download"]]
	]
]

deleteElementContentFile[info_Association,elem_]:=deleteElementContentFile[info["ContentElementLocations",elem]]
deleteElementContentFile[file:(_LocalObject|_File)]:=deleteFile[file]
deleteElementContentFile[___]:=Null

contentElementSize[locations_,contentelements_, moreinfo_,Automatic]:="ContentSize"->bytecountQuantity[ByteCount[{contentelements,moreinfo}]]/;Keys[locations]==={}
contentElementSize[_,_, _,Automatic]:="ContentSize"->Missing["NotAvailable"]
contentElementSize[_,contentelements_, _,default_]:=Association["DefaultContentElement"->default,
	"ContentSize"->bytecountQuantity[ByteCount[contentelements[default]]]]/;KeyExistsQ[contentelements,default]
contentElementSize[_,_, moreinfo_,default_]:=Association["DefaultContentElement"->default,
	"ContentSize"->bytecountQuantity[ByteCount[moreinfo[default]]]]/;KeyExistsQ[moreinfo,default]
contentElementSize[_,_, _,_]:=Association["DefaultContentElement"->default]

$cachedContentElements=Association[];
setElementCached[id_,elem_]:=($cachedContentElements[id]=DeleteMissing[DeleteDuplicates[Flatten[{Lookup[$cachedContentElements, id, {}], elem}]]])
contentCached[id_,elem_]:=MemberQ[Lookup[$cachedContentElements,id,{}],elem]

$wolframCDNBase="https://"~~("www"|"files")~~".wolframcdn.com/";
cdnURLQ[url:HoldPattern[_URL]]:=cdnURLQ[First[url]]
cdnURLQ[url_String]:=StringMatchQ[url,$wolframCDNBase~~"*"]
cdnURLQ[___]:=False

cdnURLtoCO[url:HoldPattern[_URL]]:=cdnURLtoCO[First[url]]
cdnURLtoCO[url_String]:=CloudObject[StringReplace[url,$wolframCDNBase->tocloudbase[$ResourceSystemBase]<>"objects/"]]/;cdnURLQ[url]
cdnURLtoCO[___]:=$Failed


End[] (* End Private Context *)

EndPackage[]