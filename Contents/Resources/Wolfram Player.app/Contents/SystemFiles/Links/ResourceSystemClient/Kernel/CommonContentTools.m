(* Wolfram Language Package *)


BeginPackage["ResourceSystemClient`"]

ResourceSystemClient`ResourceDownloadWithProgress

Begin["`Private`"] (* Begin Private Context *) 


resourcelocalrawexport[rtype_,co:HoldPattern[_CloudObject],bytes_,fmt_]:=CloudDeploy[ExportForm[bytes,{"Byte",fmt}],co]
resourcelocalrawexport[rtype_,lo_LocalObject,bytes_,fmt_]:=exportRawResourceContent[lo, bytes, fmt]

exportRawResourceContent[lo_,bytes_,fmt_]:=Export[lo,rawResourceBytes[fmt,bytes],"Byte"]

rawHandler[fmt_]:= Association[
   "Export" -> exportRaw,
   "Import" -> importRaw,
   "Get" -> importRaw,
   "Format" -> fmt
   ];
   
rawResourceBytes /: LocalObjects`GetHandler[rawResourceBytes[fmt_,_]] := rawHandler[fmt]

exportRaw[rawResourceBytes[fileformat_,bytes_], dataformat_, obj_LocalObject, h0_] :=Module[
	{h = h0, file},
	LocalObject;
	h["Format"] = fileformat;
  	h["Type"] = "Export";
  	h["DirectoryBundle"] = True;
  	LocalObjects`WithLocalBundleDirectory[obj,
   		file = "data"<>dataFileExtension[fileformat];
   	If[FileType[file] =!= None, DeleteFile[file]];
   		Export[file, bytes, dataformat];
   	];
  	h["ExternalData"] = file;
  	KeyDropFrom[h, {"CopyOut", "Import", "Destructor"}];
  	h]

importRaw[assoc_,obj_]:=If[
	assoc["DirectoryBundle"],
	With[{file=LocalObjects`AuxFileName[assoc]},
		LocalObjects`WithLocalBundleDirectory[obj,
			If[FileType[file]===File,
				Import[file,Lookup[assoc,"Format",FileFormat[file]]],
				$Failed]]],
	With[{data=Lookup[assoc,"Data",$Failed]},If[StringQ[data],ImportString[data,Lookup[assoc,Format,Automatic]],$Failed]]
	]


resourcedownloadInfo[rtype_,id_, info_, fmt_,co:HoldPattern[_CloudObject|_URL]]:=Association[
        	{
            "Location"->co,
            "DownloadDate"->DateObject[],
            "Format"->fmt,
            "Size"->Missing["NotAvailable"]
            }
        ]
        
resourcedownloadInfo[rtype_,id_, info_,fmt_, lo_]:=Association[
        	{
            "Location"->lo,
            "DownloadDate"->DateObject[],
            "Format"->fmt,
            "Size"->fileByteCount[lo]
            }
        ]

cloudresourceDownload[rtype_,info_,locations_, elem_]:=
	Switch[locations,
		_CloudObject,
		cloudresourcedownload[rtype,Lookup[info,"UUID",Throw[$Failed]],info,elem, locations],
		_Association,
		If[KeyExistsQ[locations,elem],
			cloudresourcedownload[rtype,Lookup[info,"UUID",Throw[$Failed]],info,elem, locations[elem]]
			,
			Throw[$Failed]
		],
		_,
		Throw[$Failed]	
	]


cloudresourcedownload[rtype_,id_,info_,elem_, co:HoldPattern[_CloudObject]]:=cloudresourcedownload[rtype,id,info,elem, co,defaultResourceTypeDownloadFormat[rtype]]
cloudresourcedownload[rtype_,id_,info_,elem_, url:HoldPattern[_URL]]:=cloudresourcedownload[rtype,id,info,elem, url,"Automatic"]

cloudresourcedownload[rtype_,id_,info_,elem_, co_,fmt_]:=Block[{res,copyinfo,
	dir, lo}
	,
	dir=resourceCopyDirectory[id,fmt, elem];
	createDirectory[dir];
	lo=localObject[FileNameJoin[{dir,"data"}]];
	res=CopyFile[co,lo];
    If[Head[res]===LocalObject,
		copyinfo=resourcedownloadInfo[rtype,id, info,fmt,res];
		If[AssociationQ[copyinfo],
	        Put[copyinfo,resourcecopyInfoFile[dir,fmt]]
	        ,
	        Throw[$Failed]
		];
    	storeDownloadVersion[id,Association["Version"->None],
    		Join[
    			Association["Element"->elem,
    				resourceDownloadStorageInfo[rtype, lo, fmt]
    			]
    		]
    	]
        ,
        Throw[$Failed]
    ]
]

resourceDownloadStorageInfo[___]:=Association[]

cloudresourceDownload[___]:=$Failed
cloudresourcedownload[___]:=$Failed

defaultResourceTypeDownloadFormat[_]:=Automatic

resourcefiledownload[rtype_, id_, elem_, dlinfo_Association]:=
	resourcefiledownload[rtype, id, dlinfo["Format"], dlinfo["ContentElementLocations"], elem,dlinfo]

resourcefiledownload[rtype_, id_, fmt_, co_, elem_]:=resourcefiledownload[rtype, id, fmt, co, elem, Association[]]

resourcefiledownload[rtype_, id_, "Compressed", co:HoldPattern[_CloudObject], elem_, dlinfo_]:=Block[{
	raw=fetchContentByteArray[ResourceSystemClient`ResourceDownload,Automatic,
		co, {"StatusCode","Body"},"CredentialsProvider" -> None],
	dir=resourceCopyDirectory[id,"MX",elem],
	lo=localObject[FileNameJoin[{resourceCopyDirectory[id,"MX",elem],"data"}]], content},
	If[!AssociationQ[raw],Throw[$Failed]];
	createDirectory[dir];
    If[raw["StatusCode"]===200,
		content=Uncompress[Lookup[raw,"Body",$Failed]];
		If[!FailureQ[content],
			If[$CacheResourceContent===False,
			    Throw[content,"NoCacheResourceDownload"]
			    ,
		        Export[lo,content,"MX"];
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
		        {lo,"MX"}
			]
	        ,
	        Throw[$Failed]
		]
        ,
        Throw[$Failed]
    ]
]


$CDNFailure=False;
				
resourcefiledownload[rtype_, id_, fmt_, url:HoldPattern[_URL], elem_, dlinfo_]:=With[{progressreset=$ProgressIndicatorContent},
	$ProgressIndicatorContent=downloadingMessageWindow[];
	Block[{res, co},
		res=Catch[resourcefiledownloadWithProgress[rtype,id,fmt,url,elem,dlinfo,progressreset]];
		If[FailureQ[res],
			co=cdnURLtoCO[url];
			If[fileExistsQ[co],
				Catch[resourcefiledownloadWithProgress[rtype,id,fmt,co,elem,dlinfo,progressreset]],
				$CDNFailure=True;
				$Failed
			],
			res
		]
	]
]/;cdnURLQ[url]

resourcefiledownload[rtype_, id_, fmt_, co:HoldPattern[_CloudObject|_URL], elem_, dlinfo_]:=With[{progressreset=$ProgressIndicatorContent},
	$ProgressIndicatorContent=downloadingMessageWindow[];
	resourcefiledownloadWithProgress[rtype,id,fmt,co,elem,dlinfo,progressreset]
]

resourcefiledownload[rtype_, id_, fmt_, co_, elem_,_]:=resourcefiledownload[rtype, id, fmt, co, elem]

$ResourceDownloadBar=True;

resourcefiledownloadWithProgress[rtype_,id_,fmt_,co_,elem_,progressreset_]:=
	resourcefiledownloadWithProgress[rtype,id,fmt,co,elem,Association[],progressreset]

resourcefiledownloadWithProgress[rtype_, id_, fmt_, co:HoldPattern[_CloudObject|_URL], elem_,dlinfo_,progressreset_]:=
Block[{dir=resourceCopyDirectory[id,fmt,elem],
	lo=localObject[FileNameJoin[{resourceCopyDirectory[id,fmt,elem],"data"}]], res},
	createDirectory[dir];
	resourcelocalrawexport[rtype,lo,{1},fmt];
	res=urlDownloadWithProgress[co,FileNameJoin[{localObjectPathName[lo],"data"<>dataFileExtension[fmt]}],
		estimateContentLength[rtype,id,elem,fmt, co]];
	verifyDownloadHash[res, dlinfo];
	If[$CacheResourceContent===False,
		With[{content=importLocalObject[lo, fmt]},
			deleteFile[lo];
			If[FailureQ[content],
				Throw[$Failed],
				Throw[content,"NoCacheResourceDownload"]
			]
		]
	];
	$ProgressIndicatorContent=progressreset;
	If[FileExistsQ[res],
		{lo,fmt},
		Throw[$Failed]
	]
]/;$ResourceDownloadBar

noCacheCloudDownload[rtype_,id_,elem_,fmt_,co_]:=
Block[{$CacheResourceContent=False},
	Catch[
		resourcefiledownloadWithProgress[rtype, id, fmt, co, elem,Association[],downloadingMessageWindow[]]
		,
		"NoCacheResourceDownload"
	]
]



estimateContentLength[__, co_]:=Quiet[CloudObjectInformation[co,"FileByteCount"]]

estimateContentLength["DataResource"|"NeuralNet",id_,elem_,fmt_,co_]:=With[{info=getResourceInfo[id]},
		If[elem===info["DefaultContentElement"],
			If[IntegerQ[info["ByteCount"]],
				info["ByteCount"]
				,
				estimateContentLength[Automatic,id, elem, fmt,co]
			]
			,
			estimateContentLength[Automatic,id, elem, fmt,co]
		]
	]

ResourceSystemClient`ResourceDownloadWithProgress[args___]:=Catch[resourceDownloadWithProgress[args]]

resourceDownloadWithProgress[args___]:=Block[{temp, res,$progressID=Unique[]},
	$ProgressIndicatorContent=downloadingMessageWindow[];
	temp=printTempOnce[$progressID,Dynamic[$ProgressIndicatorContent]];
	res=urlDownloadWithProgress[args];
	clearTempPrint[temp];
	res	
]

urlDownloadWithProgress[co_]:=urlDownloadWithProgress[co,CreateFile[]]
urlDownloadWithProgress[co_, file_]:=urlDownloadWithProgress[co,file,Automatic]

urlDownloadWithProgress[co_,file_,size_]:=Block[{
	task, url = makeFilesURL[co], auth},
	Needs["CloudObject`"];
	If[!StringQ[url],
		Message[ResourceObject::nocofile];
		Throw[$Failed]
	];
	auth=cloudAuthenticationHeader[url];
	If[urldownloadWithProgress[url,file,auth,size]==="Success",
 		file
 		,
 		Throw[$Failed]
	]
 ]
 
urldownloadWithProgress[url_,file_,auth_String,size_]:=urldownloadWithProgress[url,file,None,size]/;StringContainsQ[auth,"oauth_token=\"\""]
	
urldownloadWithProgress[url_,file_,auth_String,size_]:=urldownloadwithProgress[url,file,
	Association["Headers" -> Association["Authorization" -> auth]],size]

promptConnectOnDownloadQ[]:=(!$CloudEvaluation)&&$Notebooks

urldownloadWithProgress[url_,file_,_,size_]:=With[{res=urldownloadwithProgress[url,file,
	Association[],size]},
	If[res==="Unauthorized"&&!$CloudConnected&&promptConnectOnDownloadQ[],
		Block[{promptConnectOnDownloadQ},
			promptConnectOnDownloadQ[]:=False;
			cloudConnect[ResourceObject];
			urldownloadWithProgress[url,file,Quiet[CloudObject`Private`makeOAuthHeader[url, "GET"]],size]
		]
		,
		res
	]
]

urldownloadwithProgress[url_,file_,reqas_,size_]:=urldownloadsafelock[url, file, reqas,size]

urldownloadwithProgress[url_,file_,reqas_,size_,tempfile_, cleanupfun_] := Block[
	{$total = size, $startTime = SessionTime[], task, $status = "Success", 
	$current=-1,$current0,$throttle=0},
	TextString; (* autoload GU *)
	GeneralUtilities`ComputeWithProgress[<|
		"Summary" -> $DefaultResourceDownloadMessage,
		"DetailedSummary" -> "`current` of `total`",
		"DynamicContainer" :> $ProgressIndicatorContent,
		"TimeEstimate" -> Ceiling[size/500000], (* assume 500 kilobytes a second *)
		"Body" -> Function[callback,
			Quiet[deleteFile[file];	createDirectory[FileNameDrop[file]]];
			task = URLDownloadSubmit[
				HTTPRequest[url, reqas], 
				tempfile,
				HandlerFunctions -> Association[
					"TaskStatusChanged"->(($status = handleStatus[#, $status])&),
					"HeadersReceived" -> (($total = handleHeaders[#, $total]) &),
					"TaskProgress" -> (({$throttle,$current,$total}=throttledCallback[#,callback,{$throttle,$current,$total}]) &)
				],
				HandlerFunctionsKeys -> {"ByteCountTotal", "ByteCountDownloaded", "Headers","StatusCode"},
				FollowRedirects->False
			];
			TaskWait[task];
			tempfile;
		],
		"Cleanup" -> Function[
			If[!MatchQ[task["TaskStatus"], "Finished" | "Removed"], 
				TaskAbort[task]; 
				TaskRemove[task]
			];
			cleanupfun[tempfile,file]
		]
	|>];
	$status
];

makeTempFile[file_,hash_]:=FileNameJoin[MapAt[FileBaseName[#] <> hash &, FileNameSplit[file], -1]]

urldownloadsafelock[url_, file_, reqas_,size_] := With[{res=Catch[With[{hash = Hash[{$SessionID, $KernelID, $ProcessID, RandomReal[]}, "MD5", "HexString"]},
	Block[{tempfile = makeTempFile[file,hash], lockfile = file <> ".lock", dlres, t0=SessionTime[]},
		If[FileExistsQ[lockfile],
			If[Get[lockfile] === hash,
				DeleteFile[lockfile];
				If[FileExistsQ[lockfile],Throw[$Failed, "urldownloadsafelock"]];
      			urldownloadsafelock[url, file, reqas,size]
      			,
      			Throw[$Failed, "urldownloadsafelock"]
      		]
     		,
     		Put[hash, lockfile];
     		urldownloadwithProgress[url,file,reqas,size,tempfile,renameHashCheck[lockfile, #1,#2,hash]&]
     		,
     		Throw[$Failed, "urldownloadsafelock"];
     	]
     ]], "urldownloadsafelock"]},
     If[FailureQ[res],
     	Message[ResourceData::dllock];Throw[$Failed]];
     res
]

renameHashCheck[lockfile_, tempfile_,file_,hash_]:=If[
	Quiet[Get[lockfile]] === hash,
		deleteFile[file];
 		If[FileExistsQ[file],Return[$Failed]];
		renameFileWithRetry[tempfile, file, AbsoluteTime[]];
 		deleteFile[lockfile];
 		file
 		,
     	deleteFile[tempfile];
     	$Failed
     ]


throttledCallback[progress_,callback_,{throttleOld_,currentOld_,totalOld_}]:=throttledCallbackCloud[progress,callback,{throttleOld,currentOld,totalOld}](*/;$CloudEvaluation
throttledCallback[progress_,callback_,{throttleOld_,currentOld_,totalOld_}]:=throttledcallback[progress,callback,{throttleOld,currentOld,totalOld}]*)

throttledcallback[progress_,callback_,{th_,currentOld_,totalOld_}]:=Module[{currentTemp,currentNew=currentOld,totalNew},
	{currentTemp, totalNew} = handleProgress[progress, totalOld];
	If[currentTemp=!=None,currentNew=currentTemp];
	callback[<|"progress" -> N[currentNew / totalNew], "current" -> MemoryString[currentNew], "total" -> MemoryString[totalNew]|>];
	{th,currentNew,totalNew}
]

throttledCallbackCloud[progress_,callback_,{0,currentOld_,totalOld_}]:=With[{res=throttledcallback[progress,callback,{0,currentOld,totalOld}]},
	If[ListQ[res],
		{1,res[[2]],res[[3]]},
		{1,currentOld,totalOld}
	]
]

throttledCallbackCloud[progress_,callback_,{th_,currentOld_,totalOld_}]:={Mod[th+1,$progressBarCloudUpdatePeriod],currentOld,totalOld}

cloudAuthenticationHeader[url_]:=None/;$CloudEvaluation&&(!cloudbaseConnected[tocloudbase[url]])
cloudAuthenticationHeader[url_]:=Quiet[CloudObject`Private`makeOAuthHeader[url, "GET"]]

renameFileWithRetry[source_, dest_, t0_]:=With[{res=RenameFile[source,dest]},
	If[FailureQ[res],
		Quiet[Pause[0.1];renameFileWithRetry[source,dest,t0]]
		,
		res
	]	
]/;TrueQ[AbsoluteTime[]-t0<1]

renameFileWithRetry[source_, dest_, _]:=RenameFile[source,dest]

resourcefiledownloadWithProgress[rtype_, id_, fmt_, co:HoldPattern[_CloudObject], elem_,dlinfo_,progressreset_]:=
With[{
	raw=fetchContentByteArray[ResourceSystemClient`ResourceDownload,Automatic,
		co, {"StatusCode", "BodyByteArray"},"CredentialsProvider" -> None],
	dir=resourceCopyDirectory[id,fmt,elem],
	lo=localObject[FileNameJoin[{resourceCopyDirectory[id,fmt,elem],"data"}]]},
	If[!AssociationQ[raw],
		$ProgressIndicatorContent=progressreset;
		Throw[$Failed]];
	createDirectory[dir];
    If[raw["StatusCode"]===200,
    	verifyDownloadHash[raw,dlinfo];
		If[$CacheResourceContent===False,
		    Throw[noCacheImportBytes[Lookup[raw,"BodyByteArray",Lookup[raw,"BodyBytesArray",$Failed]],fmt],"NoCacheResourceDownload"]
		    ,
	        resourcelocalrawexport[rtype,lo,Lookup[raw,"BodyByteArray",Lookup[raw,"BodyBytesArray",$Failed]],fmt];
	        Put[Association[
	        	{
	            "Location"->lo,
	            "DownloadDate"->DateObject[],
	            "Format"->fmt,
	            "Size"->bytecountQuantity[Length[Lookup[raw,"BodyByteArray",Lookup[raw,"BodyBytesArray",$Failed]]]]
	            }
	        ],
	        resourcecopyInfoFile[dir,fmt]
	        ];
	        {lo,fmt}
		]
        ,
        Throw[$Failed]
    ]
]

resourcecopyfiledownload[rtype_,id_,fmt_,co:HoldPattern[_CloudObject], elem_]:=Block[{
	dir=resourceCopyDirectory[id,fmt,elem],
	lo=localObject[FileNameJoin[{resourceCopyDirectory[id,fmt,elem],"data"}]], size},
	createDirectory[dir];
	size=fileByteCount[CopyFile[co, lo]];
    If[size>0,
    	If[$CacheResourceContent===False,    	
    		With[{content=importLocalObject[lo, fmt]},
    			Throw[content,"NoCacheResourceDownload"];
    			deleteFile[lo];
    			content
    		]
    		,
	        Put[Association[
	        	{
	            "Location"->lo,
	            "DownloadDate"->DateObject[],
	            "Format"->fmt,
	            "Size"->size
	            }
	        ],
	        resourcecopyInfoFile[dir,fmt]
	        ];
	        {lo,fmt}
	        ,
	        Throw[$Failed]
    	]
    ]
]


resourceimportFileDownload[rtype_,id_,_,co:HoldPattern[_CloudObject], elem_]:=With[{
	wdf=CloudImport[co],
	dir=resourceCopyDirectory[id,"MX",elem],
	lo=localObject[FileNameJoin[{resourceCopyDirectory[id,"MX",elem],"data"}]]},
	If[$CacheResourceContent===False,
	    Throw[wdf,"NoCacheResourceDownload"]
	    ,
		createDirectory[dir];
	    Export[lo,wdf,"MX"];
	    Put[Association[
	    	{
	        "Location"->lo,
	        "DownloadDate"->DateObject[],
	        "Format"->"MX",
	        "Size"->bytecountQuantity[ByteCount[wdf]]
	        }
	    ],
	    resourcecopyInfoFile[dir,"MX"]
	    ];
	    {lo,"MX"}
	]
]


$AutoCacheLimit=10^10;

readresourcelocal[format_,lo:HoldPattern[_LocalObject]]:=With[{res=readresourcelocal0[format,lo]},
	If[res===Null,
		$Failed,
		res
	]	
]

readresourcelocal[format_,location_]:=readresourcelocal0[format,location]

readresourcelocal0[format_, as_Association]:=readresourcelocal[format, as["Content"]]/;KeyExistsQ[as,"Content"]
readresourcelocal0[fmt_,dir_String]:=importFromLocalDir[fmt,dir]/;DirectoryQ[dir]
readresourcelocal0["WLNet",location_]:=Import[location,"WLNet"]
readresourcelocal0["Package",location_]:=Get[location]
readresourcelocal0["NB",lo_LocalObject]:=Get[lo]
readresourcelocal0["NB",file_]:=NotebookOpen[file]
readresourcelocal0["WDF",location_]:=Get[location]
readresourcelocal0[fmt:"MX"|"PNG"|"WLNet",location_]:=Import[location,fmt]
readresourcelocal0["Binary"|"WXF",location_]:=BinaryDeserialize[ByteArray[Import[location]]]
readresourcelocal0[Automatic|"Automatic",location_]:=importlocal[location]
readresourcelocal0[str_String,location_]:=Import[location,str]
readresourcelocal0[_,location_]:=readresourcelocal0[Automatic,location]

importFromLocalDir[fmt_,dir_String]:=importFromLocalDir[fmt,dir,FileNames["*",dir]]
importFromLocalDir[fmt_String,dir_,files_List]:=If[Quiet[MemberQ[files,FileNameJoin[{dir,"data."<>fmt}]]],
	Import[FileNameJoin[{dir,"data."<>fmt}]],
	If[Length[#]==1,readresourcelocal0[fmt,First[#]],Import[dir]]&[Select[files,FileBaseName[#]=="data"&]]
]

importFromLocalDir[_,dir_,files_List]:=If[Length[#]==1,readresourcelocal0[fmt,First[#]],Import[dir]]&[Select[files,FileBaseName[#]=="data"&]]

importFromLocalDir[___]:=$Failed


cacheResourceQ[info_Association]:=cacheresourceQ[Lookup[info,"Caching",Automatic], 
	Lookup[info,"ContentSize",Lookup[info,"ByteCount",Quantity[0, "Bytes"]]]]
cacheResourceQ[id_String,_]:=cacheResourceQ[getResourceInfo[id]]

cacheresourceQ[Automatic, _]:=False/;($CloudEvaluation&&($EvaluationCloudBase===$CloudBase))
cacheresourceQ[Automatic, size_Association]:=True/;Max[Select[QuantityMagnitude/@size,NumberQ]]<$AutoCacheLimit
cacheresourceQ[Automatic, size_]:=True/;QuantityMagnitude[size]<$AutoCacheLimit
cacheresourceQ[True, _]=True
cacheresourceQ[False, _]=True
cacheresourceQ[_, _]=False

resourceDataPostProcessFunction[{}]=Identity;
resourceDataPostProcessFunction[f:(_Function|_Composition|_RightComposition)]:=f
resourceDataPostProcessFunction[sym_Symbol]:=sym
resourceDataPostProcessFunction[___]:=Identity



ResourceSystemClient`$DeployResourceSubmissionContent=True;
completeResourceSubmissionWithElements[rtype_, id_, as0_]:=Block[{as, funcs,deployed},
	as=DeleteMissing[AssociationMap[validateParameter[rtype,#]&,as0]];
	If[KeyExistsQ[as,"ContentElementFunctions"],
		funcs=as["ContentElementFunctions"],
		funcs=getAllElementFunction[id,Lookup[as0,"ContentElements"]]
	];
	If[Keys[funcs]=!={},
		as["ContentElementFunctions"]=Compress[funcs]
	];
	If[TrueQ[ResourceSystemClient`$DeployResourceSubmissionContent],
		
		If[KeyExistsQ[as,"ContentElementLocations"],
			deployed=deploySubmissionContentLocations[rtype,id, as["ContentElementLocations"]];
			as["ContentElementLocations"]=Join[Lookup[as,"ContentElementLocations",Association[]],deployed]
		];
		If[KeyExistsQ[as,"Content"],
			deployed=deploySubmissionContent[rtype,id, as["Content"]];
			as["Content"]=KeyDrop[as["Content"],Keys[deployed]];
			as["ContentElementLocations"]=Join[Lookup[as,"ContentElementLocations",Association[]],deployed]
		]
	];	
	If[KeyExistsQ[as,"InformationElements"],
		as["InformationElements"]=secureInformationElements[as["InformationElements"]]	
	];
	If[!(KeyExistsQ[as,"Content"]||KeyExistsQ[as,"ContentElementLocations"]),Message[ResourceSubmit::noncont];Throw[$Failed]];
	as
]

deploySubmissionContent[rtype_,id_, content_Association]:=Association[KeyValueMap[deployLargeDataSubmissionContent[rtype,id,##]&,content]]

deployLargeDataSubmissionContent[rtype_,_, _, value_]:={}/;ByteCount[value]<deployResourceSubmissionContentSizeLimit[rtype]

deployLargeDataSubmissionContent[rtype_,id_, key_, value_]:=With[{co=submissionContentLocation[id, key,value]},
	key->If[FileExistsQ[co],
		co,
		$submitProgressContent=If[StringQ[key],
			"Exporting \""<>key<>"\" content \[Ellipsis]",
			"Exporting content \[Ellipsis]"
		];
		printTempOnce[$progressID,progressPanel[Dynamic[$submitProgressContent]]];
		deploydataSubmissionContent[value, co,determineSubmissionContentFormat[rtype,key, value]]
	]
]

deploydataSubmissionContent[value_, co_, "Binary"]:=CloudExport[BinarySerialize[value], "Binary",co,Permissions->{$ResourceSystemAdminUser->"Read"},MetaInformation->{"Format"->"Binary"}]
deploydataSubmissionContent[value_, co_, fmt_]:=CloudExport[value, fmt,co,Permissions->{$ResourceSystemAdminUser->"Read"},MetaInformation->{"Format"->fmt}]

determineSubmissionContentFormat[rtype_,_, _]:=defaultResourceTypeDownloadFormat[rtype]

deploySubmissionContentLocations[rtype_,id_, content_Association]:=With[{rules=KeyValueMap[deployLargeDataSubmissionFile[rtype,id,##]&,content]},
	Association[rules]
]

submissionContentLocation[id_,key_,value_]:=With[{hash=Hash[{"Expression",value},"Expression","HexString"], 
	basehash=StringTake[Hash[$submissionResourceBase/.Automatic->$ResourceSystemBase, "MD5", "HexString"], 5]},
	If[StringQ[hash],
		CloudObject[FileNameJoin[{"ResourceSubmissions",basehash,hash},OperatingSystem->"Unix"]]
		,
		CloudObject[]
	]	
]
submissionContentLocation[__]:=CloudObject[IconRules->{},Permissions->{$ResourceSystemAdminUser->"Read"}]

deployLargeDataSubmissionFile[_,_, _, HoldPattern[_CloudObject]] := {}
deployLargeDataSubmissionFile[_,_, _, Automatic|None] := {}
deployLargeDataSubmissionFile[rtype_,_, _, value_]:={}/;fileByteCount[value]<deployResourceSubmissionContentSizeLimit[rtype]

deployLargeDataSubmissionFile[rtype_,id_, key_, lo:HoldPattern[_LocalObject]]:=deploydataSubmissionFile[rtype,id,key,localObjectDataFile[lo]]
deployLargeDataSubmissionFile[rtype_,id_, key_, file:HoldPattern[_File]]:=deploydataSubmissionFile[rtype,id,key,First[file]]
deployLargeDataSubmissionFile[rtype_,id_, key_, file_]:=deploydataSubmissionFile[rtype,id,key,file]

deploydataSubmissionFile[rtype_,id_, key_, file_]:=(
	$submitProgressContent=If[StringQ[key],
		"Exporting \""<>key<>"\" content \[Ellipsis]",
		"Exporting content \[Ellipsis]"
	];
	printTempOnce[$progressID,progressPanel[Dynamic[$submitProgressContent]]];
	key->deploydatasubmissionFile[FileExtension[file],file,submissionHashCloudPath[rtype,id,key,file]]
	)

deploydatasubmissionFile["wl"|"m",file_, path_] := With[{co=submissionCloudObject[path]},
	If[StringQ[path]&&FileExistsQ[co],
		co,
		CloudExport[Import[file, "Byte"], {"Byte", "Package"}, co]
	]
]

deploydatasubmissionFile[_,file_, path_] := With[{co=submissionCloudObject[path,FileFormat[file]]},
	If[StringQ[path]&&FileExistsQ[co],
		co,
		CopyFile[file, co]
	]
]

submissionHashCloudPath[rtype_,id_,key_,file_]:=With[{hash=FileHash[file,"MD5",All,"HexString"],
	basehash=StringTake[Hash[$submissionResourceBase/.Automatic->$ResourceSystemBase, "MD5", "HexString"], 5]},
	If[StringQ[hash],
		FileNameJoin[{"ResourceSubmissions",basehash,hash},OperatingSystem->"Unix"]
		,
		Automatic
	]	
]

submissionCloudObject[path_String, fmt_String]:=CloudObject[path,IconRules->{},Permissions->{$ResourceSystemAdminUser->"Read"}, MetaInformation -> {"Format" -> fmt}]
submissionCloudObject[path_String,___]:=CloudObject[path,IconRules->{},Permissions->{$ResourceSystemAdminUser->"Read"}]
submissionCloudObject[Automatic,___]:=submissionCloudObject[]
submissionCloudObject[]:=submissionContentLocation[]

deployResourceSubmissionContentSizeLimit[_]:=10^5;

ResourceSystemClient`DeleteResourceSubmissionCloudObjects[]:=DeleteObject[CloudObject["ResourceSubmissions"]]

secureInformationElements[infoElements_]:=secureinformationElements/@infoElements

secureinformationElements[infoElem_]:=infoElem/;secureInformationElementQ[infoElem]

secureinformationElements[infoElem_]:=ResourceSystemClient`Private`CompressedInformationElement[Compress[infoElem]]

uncompressInformationElements[infoElements_Association]:=uncompressinformationElements/@infoElements

uncompressinformationElements[comp_ResourceSystemClient`Private`CompressedInformationElement]:=Uncompress[First[comp]]

uncompressinformationElements[expr_]:=expr

secureInformationElementQ[_String] = True;
secureInformationElementQ[l_List] := AllTrue[l, secureInformationElementQ]
secureInformationElementQ[_?AtomQ] := False
secureInformationElementQ[expr_] := 
 Head[expr] === 
  Head[Quiet[
    TimeConstrained[
     Interpreter[
       Restricted[
        "Expression", {Association, Hold, Rule, String, Automatic, 
         Integer, True, False, CloudObject, DateObject, Entity, URL, 
         Hyperlink, List, GeoPosition}, Automatic, None]][expr], 2]]]
         
         
End[] (* End Private Context *)

EndPackage[]