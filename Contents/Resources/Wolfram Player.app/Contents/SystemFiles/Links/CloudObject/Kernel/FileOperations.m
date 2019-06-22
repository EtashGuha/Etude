BeginPackage["CloudObject`"]


Begin["`Private`"]

Unprotect[CloudObject];

(* CopyFile *)

Unprotect[CopyFile];

Options[CopyFile] = DeleteDuplicates[Join[Options[CopyFile], {"MIMEType" -> Automatic}]];

CloudObject /: CopyFile[src_, obj:CloudObject[uri_, opts:OptionsPattern[CloudObject]], o:OptionsPattern[]] :=
    Module[{mimetype, meta, content, parameters, res, sharingList},
    	Catch[
        	mimetype = Replace[OptionValue["MIMEType"],
            	Automatic -> Switch[FileExtension[src, OperatingSystem -> "Unix"],
                	"js", formatToMimeType["text/javascript"],
                	"html", formatToMimeType["text/html"],
                	"css", formatToMimeType["text/css"],
                	_, formatToMimeType[FileFormat[src]]]];
        	meta = Join[handleMetaInfo[getOptionValue[{opts}, MetaInformation], CopyFile], {handleAppearanceRules[getOptionValue[{opts}, AppearanceRules], CopyFile],
        		handleAutoCopy[getOptionValue[{opts}, AutoCopy], CopyFile], handleSourceLink[getOptionValue[{opts}, SourceLink], False, obj, CopyFile]}];
        	{content, parameters} = If[TrueQ[$CloudEvaluation],
                 {AbsoluteFileName[src], {"requestBodyIsFilePath" -> "True"}},
                 {BinaryReadList[src], {}}];
        	res = writeObject[CloudObject[uri], content, mimetype,
        		getOptionValue[{opts}, Permissions],
        		getOptionValue[{opts}, IconRules],
        		Null, meta, parameters, CopyFile];
        	sharingList = handleSharingList[getOptionValue[{opts}, SharingList], CopyFile];
        	If[sharingList =!= {}, setSharees[res, sharingList]]; (* will be optimized in CLOUD-14132 *)
       		res,
       		(* catch tag: *)
       		BadOption
    	]
	]

CloudObject /: CopyFile[obj_CloudObject, target_, o:OptionsPattern[]] := Module[{tempfilename, mimetype, type},
    {tempfilename, type} = readObject[obj];
    mimetype = Replace[OptionValue["MIMEType"], Automatic->type];
    If[tempfilename === $Failed, Return[$Failed]];
    cleanup[tempfilename, CopyFile[tempfilename, target, o]]
]

CloudObject /: CopyFile[src_CloudObject, target:CloudObject[uri_, opts:OptionsPattern[CloudObject]], o:OptionsPattern[]] :=
   	Module[{mimetype, perms, iconRules, meta, res, sharingList},
   		Catch[
   			perms = getOptionValue[{opts}, Permissions];
   	   		iconRules = getOptionValue[{opts}, IconRules];
   	   		meta = Join[handleMetaInfo[getOptionValue[{opts}, MetaInformation], CopyFile], {handleAppearanceRules[getOptionValue[{opts}, AppearanceRules], CopyFile],
   	   			handleAutoCopy[getOptionValue[{opts}, AutoCopy], CopyFile], handleSourceLink[getOptionValue[{opts}, SourceLink], False, src, CopyFile]}]; 
       		res = If[getCloud[src] === getCloud[target],
       	   		(*TODO: Change the MIME type to Automatic once CLOUD-11740 is done.  copyContentFrom in execute should handle
       	   		copying the type*)
       	   		mimetype = Replace[OptionValue["MIMEType"], Automatic -> CloudObjectInformation[src, "MIMEType"]];
       	   		writeObject[CloudObject[uri], src, mimetype, perms, iconRules, Null, meta, {}, CopyFile],
       		(*else*)
           		Module[{tempfilename, content, type},
               		{tempfilename, type} = readObject[src];
               		mimetype = Replace[OptionValue["MIMEType"], Automatic->type];
               		If[tempfilename === $Failed, Return[$Failed]];
               		content = BinaryReadList[tempfilename];
               		cleanup[tempfilename, writeObject[CloudObject[uri], content, mimetype, perms, iconRules, Null, meta, {}, CopyFile]]
           		]];
       		sharingList = handleSharingList[getOptionValue[{opts}, SharingList], CopyFile];
       		If[sharingList =!= {}, setSharees[res, sharingList]];(* will be optimized in CLOUD-14132 *)
       		res,
       		(* catch tag: *)
       		BadOption
   		]
   	]

CloudObject /: CopyFile[failureObj:Failure["CloudObjectInvalidPathFailure", ___], target:CloudObject[uri_, opts:OptionsPattern[CloudObject]], o:OptionsPattern[]] := failureObj

CopyFile[failureObj:Failure["CloudObjectInvalidPathFailure", ___], target_, o:OptionsPattern[]] := failureObj

CopyFile[src_, failureObj:Failure["CloudObjectInvalidPathFailure", ___], o:OptionsPattern[]] := failureObj

Protect[CopyFile];

(* RenameFile *)

Unprotect[RenameFile];

CloudObject /: RenameFile[src_CloudObject, dest_CloudObject] :=
    Module[{uuid, cloud, path},
        {cloud, uuid} = Quiet[getCloudAndUUID[src]];
        If[cloud === $Failed || uuid === $Failed || uuid === None,
            Message[RenameFile::cloudnf, src];
            Return[$Failed]
        ];
        {cloud, path} = getCloudAndPath[dest];
        If[path === $Failed,
            Message[RenameFile::cldnm, dest];
            Return[$Failed]
        ];
        Replace[
            execute[cloud, "PUT", {"files", uuid, "path"}, Parameters -> {"path" -> path}],
                {
                {_String, _List} :> dest,
                HTTPError[409, ___] :> (Message[RenameFile::filex, dest]; $Failed),
                err_HTTPError :> (checkError[err, RenameFile]; $Failed),
                _ :> (Message[RenameFile::srverr]; $Failed)
            }
        ]
    ]

RenameFile[src_CloudObject, failureObj:Failure["CloudObjectInvalidPathFailure", ___]] := failureObj

RenameFile[failureObj:Failure["CloudObjectInvalidPathFailure", ___], dest_CloudObject] := failureObj

RenameFile[failureObjSrc:Failure["CloudObjectInvalidPathFailure", ___], failureObjDest:Failure["CloudObjectInvalidPathFailure", ___]] := failureObjSrc

Protect[RenameFile];

(* ParentDirectory *)

Unprotect[ParentDirectory];

ParentDirectory[obj_CloudObject] :=
    If[ obj === $CloudRootDirectory,
        $CloudRootDirectory,
        Module[{cloud, path},
            {cloud, path} = getCloudAndPath[obj];
            CloudObject[
                URLBuild[{cloud, $objRoot,
                    If[FileNameDepth[path] === 1, path, FileNameDrop[path, OperatingSystem -> "Unix"]]
                }]
            ]
        ]
    ]

ParentDirectory[failureObj:Failure["CloudObjectInvalidPathFailure", ___]] := failureObj

Protect[ParentDirectory];

(* RenameDirectory *)

Unprotect[RenameDirectory];

CloudObject /: RenameDirectory[src_CloudObject, dest_CloudObject] :=
    Module[{uuid, cloud, path},
        {cloud, uuid} = getCloudAndUUID[src];
        If[!(StringQ[cloud] && UUIDQ[uuid]),
            Message[RenameDirectory::cloudnf, src];
            Return[$Failed]
        ];
        {cloud, path} = getCloudAndPath[dest];
        If[path === $Failed,
            Message[RenameDirectory::cldnm, dest];
            Return[$Failed]
        ];
        Replace[execute[cloud, "PUT", {"files", uuid, "path"}, Parameters -> {"path" -> path}],
            {
                {_String, _List} :> dest,
                HTTPError[409, ___] :> (Message[RenameDirectory::filex, dest]; $Failed),
                err_HTTPError :> (checkError[err, RenameDirectory]; $Failed),
                _ :> (Message[RenameDirectory::srverr]; $Failed)
            }
        ]
    ]

RenameDirectory[src_CloudObject, failureObj:Failure["CloudObjectInvalidPathFailure", ___]] := failureObj

RenameDirectory[failureObj:Failure["CloudObjectInvalidPathFailure", ___], src_CloudObject] := failureObj

RenameDirectory[failureObjSrc:Failure["CloudObjectInvalidPathFailure", ___], failureObjDest:Failure["CloudObjectInvalidPathFailure", ___]] := failureObjSrc

getCloudAndPath[CloudObject[uri_String, ___]] := getCloudAndPath[uri]

getCloudAndPath[uri_String] :=
    Replace[
        parseURI[uri],
        {
            {cloud_, _, user_, path_List, ___} :>
                {cloud, user<>"/"<>FileNameJoin[path, OperatingSystem -> "Unix"]},
            {cloud_, uuid_String, _, None, _, extrapath_List, ___} :>
                {cloud,
                    uuid<>"/"<>FileNameJoin[extrapath, OperatingSystem -> "Unix"]},
            _ :> {$Failed, $Failed}
        }
    ]

Protect[RenameDirectory];

(* ReadList *)

CloudObject /: ReadList[co_CloudObject, rest___] :=
    Module[{tempfilename, mimetype},
        {tempfilename, mimetype} = readObject[co, ReadList];
        If[tempfilename === $Failed, Return[$Failed]];
        cleanup[tempfilename,
            Block[{$CharacterEncoding = "UTF-8"},
                ReadList[tempfilename, rest]
            ]
        ]
    ]

(* DeleteFile *)

Options[DeleteCloudObject] = {Asynchronous -> False};

DeleteCloudObject[co_CloudObject, OptionsPattern[]] :=
    responseCheck[execute[co, "DELETE", Asynchronous -> OptionValue[Asynchronous]], DeleteCloudObject]

Unprotect[DeleteFile];

CloudObject /: DeleteFile[co_CloudObject] :=
    Module[{mime, cloud, uuid, coinfo},
        coinfo = Quiet[Check[CloudObjectInformation[co, {"UUID", "MIMEType"}], $Failed]];
        If[FailureQ[coinfo], (* file not found *)
            Message[DeleteFile::nffil, co];
            Return[$Failed];
        ];

        uuid = coinfo["UUID"];
        mime = coinfo["MIMEType"];
        cloud = getCloud[co];
        Which[
            MemberQ[$autoRefreshedMimeTypes, mime], Catch[iCloudRemoveAutoRefreshed[{cloud, uuid}, DeleteFile], $tag],
            MemberQ[$taskMimeTypes, mime], Catch[iCloudRemoveScheduledTask[{cloud, uuid}, DeleteFile], $tag],
            True, deleteCloudObjectFile[cloud, uuid, co]
        ]
    ]

deleteCloudObjectFile[cloud_String, uuid_String, co_CloudObject] :=
    Replace[
        execute[cloud, "DELETE", {"files", uuid}, Parameters -> {"filter" -> "file"}],
        {
            {_String, _List} :> Null (* success *),
            HTTPError[404, ___] :> (Message[DeleteFile::nffil, co]; $Failed),
            HTTPError[412, ___] :> (Message[DeleteFile::fdir, co]; $Failed), (* attempted to delete directory *)
            other_ :> (Message[DeleteFile::srverr]; $Failed)
        }
    ]

DeleteFile[list : {___, _CloudObject, ___}] := (DeleteFile /@ list;)

DeleteFile[failureObj:Failure["CloudObjectInvalidPathFailure", ___]] := failureObj

Protect[DeleteFile];

(* DeleteDirectory *)

CloudObject /: DeleteDirectory[dir_CloudObject, OptionsPattern[]] :=
    Module[{recursive = TrueQ[OptionValue[DeleteContents]],
        cloud, uuid, params},
        {cloud, uuid} = Quiet[getCloudAndUUID[dir]];
        If[!UUIDQ[uuid], (* named directory not found *)
            Message[DeleteDirectory::nodir, dir];
            Return[$Failed];
        ];

        params = {"recursive" -> ToLowerCase@ToString[recursive], "filter" -> "directory"};
        Replace[
            execute[cloud, "DELETE", {"files", uuid}, Parameters -> params],
            {
                {_String, _List} :> Return[Null] (* success *),
                HTTPError[400, ___] :> If[recursive, Message[CloudObject::srverr]; $Failed, Null],
                HTTPError[404, ___] :> (Message[DeleteDirectory::nodir, dir]; Return[$Failed]),
                HTTPError[412, ___] :> (Message[DeleteDirectory::nodir, dir]; Return[$Failed]), (* attempted to delete file *)
                other_ :> (Message[CloudObject::srverr]; Return[$Failed])
            }
        ];
        (* assert: delete failed with status 400 and recursive was not true ->
            directory was not empty. *)
        (* if directory is a bundle type, does it have only the bundle file? *)
        (* TODO handle bundle type *)
        Message[DeleteDirectory::dirne, dir];
        $Failed
    ];

(* CreateDirectory *)

CloudObject /: CreateDirectory[co_CloudObject] :=
    responseCheck[
        Replace[
            execute[co, Automatic, UseUUID -> False, Type -> "inode/directory"],
            {
                HTTPError[400, ___] :> (Message[CreateDirectory::filex, co];
                Return[co])
            }
        ],
        CreateDirectory,
        co]

(* Handle CloudObject failure due to illegal path name *)
Unprotect[CopyDirectory, FileNameJoin]

CopyDirectory[src_, failureObjDest:Failure["CloudObjectInvalidPathFailure", ___]] := failureObjDest

CopyDirectory[failureObjSrc:Failure["CloudObjectInvalidPathFailure", ___], dest_] := failureObjSrc

CopyDirectory[src_CloudObject, failureObjDest:Failure["CloudObjectInvalidPathFailure", ___]] := failureObjDest

CopyDirectory[failureObjSrc:Failure["CloudObjectInvalidPathFailure", ___], dest_CloudObject] := failureObjSrc

CopyDirectory[failureObjSrc:Failure["CloudObjectInvalidPathFailure", ___], failureObjDest:Failure["CloudObjectInvalidPathFailure", ___]] := failureObjSrc

FileNameJoin[{failureObj:Failure["CloudObjectInvalidPathFailure", ___], path___}] := failureObj

Protect[CopyDirectory, FileNameJoin]

makeFailureHandling[func_, patt___] := (
    Unprotect[func];
    func[failureObj:Failure["CloudObjectInvalidPathFailure", ___], patt] := failureObj;
    Protect[func];
)

Scan[makeFailureHandling, {CreateDirectory, DeleteDirectory, FileExistsQ, DeleteObject, FileType, FileByteCount, FilePrint, FileDate, DirectoryQ, ReadList, FileHash}];

(* CopyDirectory *)
Unprotect[CopyDirectory]

CloudObject /: CopyDirectory[src_, obj:CloudObject[uri_, opts:OptionsPattern[CloudObject]]] := 
    Module[{files = FileNames["*", src], dir},
        If[DirectoryQ[obj],
            Message[CopyDirectory::filex, CloudObjectInformation[obj, "Name"]];
            $Failed,
            dir = CreateDirectory[obj];
            If[dir === $Failed,
                $Failed,
                Do[
                    If[FileType[file] === Directory,
                        CopyDirectory[file, FileNameJoin[{dir, FileNameTake[file]}]],
                        CopyFile[file, FileNameJoin[{dir, FileNameTake[file]}]]
                    ],
                    {file, files}
                ];
                dir
            ]
        ]
    ]
    
CloudObject /: CopyDirectory[src_CloudObject, target_] := 
    Module[{files = CloudObjects[src], dir},
        If[DirectoryQ[target],
            Message[CopyDirectory::filex, target];
            $Failed,
            dir = CreateDirectory[target];
            If[dir === $Failed,
                $Failed,
                Do[
                    If[FileType[file] === Directory,
                        CopyDirectory[file, FileNameJoin[{dir, CloudObjectInformation[file, "Name"]}]],
                        CopyFile[file, FileNameJoin[{dir, CloudObjectInformation[file, "Name"]}]]
                    ],
                    {file, files}
                ];
                dir
            ]
        ]
    ]
    
CloudObject /: CopyDirectory[src_CloudObject, obj:CloudObject[uri_, opts:OptionsPattern[CloudObject]]] := 
    Module[{files = CloudObjects[src], dir},
        If[DirectoryQ[obj],
            Message[CopyDirectory::filex, CloudObjectInformation[obj, "Name"]];
            $Failed,
            dir = CreateDirectory[obj];
            If[dir === $Failed,
                $Failed,
                Do[
                    If[FileType[file] === Directory,
                        CopyDirectory[file, FileNameJoin[{dir, CloudObjectInformation[file, "Name"]}]],
                        CopyFile[file, FileNameJoin[{dir, CloudObjectInformation[file, "Name"]}]]
                    ],
                    {file, files}
                ];
                dir
            ]
        ]
    ]
Protect[CopyDirectory]

(* FileExistsQ *)
Unprotect[FileExistsQ]

CloudObject /: FileExistsQ[CloudObject[uri_String, ___]] := Replace[
    parseURI[uri],
    {
    (* parts: {cloud, uuid, user, path, ext, extraPath, search} *)

    (* anonymous cloud object *)
        {cloud_, uuid_String?UUIDQ, _, None, _, {}, _} :>
            MatchQ[execute[cloud, "GET", {"files", uuid, {"path"}}],
                {_String, _List}],
    (* named cloud object *)
        {cloud_, None, userid_, path_, _, _, _} :>
            MatchQ[execute[cloud, "GET", {"files"},
                Parameters -> {"path" -> userid <> "/" <>
                    FileNameJoin[path, OperatingSystem -> "Unix"]}],
                {_String, bytes_List /; bytes =!= {}}],
    (* named cloud object inside unnamed directory *)
        {cloud_, diruuid_, userid_, None, _, extraPath_List, _} :>
            MatchQ[execute[cloud, "GET", {"files"},
                Parameters -> {"path" -> diruuid<>"/"<>
                    FileNameJoin[extraPath, OperatingSystem -> "Unix"]}],
                {_String, bytes_List /; bytes =!= {}}]
    }
]
Protect[FileExistsQ]

(* Cloud file name manipulation *)

Unprotect[FileNameJoin]

FileNameJoin[{CloudObject[uri_, rest___], path___}] := CloudObject[JoinURL[uri, path], rest]

Protect[FileNameJoin]

(* FileType *)

CloudObject /: FileType[co_CloudObject] :=
    With[{info = Quiet[cloudObjectInformation[co, FileType]]},
        If[Head[info] === System`CloudObjectInformationData,
            First[info]["FileType"],
            None (* FileType is quiet about non-existent files *)
        ]
    ]

(* DirectoryQ *)

CloudObject /: DirectoryQ[co_CloudObject] := FileType[co] === Directory

(* FileByteCount *)

CloudObject /: FileByteCount[co_CloudObject] := Replace[
    cloudObjectInformation[co, FileByteCount],
    {
        System`CloudObjectInformationData[info_Association] :> info["FileByteCount"],
        _ :> $Failed
    }
]

(* FilePrint *)

CloudObject /: FilePrint[co_CloudObject] :=
    With[{raw = Import[co, "Text"]},
        If[StringQ[raw],
            CellPrint[Cell[raw, "Print"]],
            (* else *)
            $Failed
        ]
    ]
    
(* FileFormat *)
    
CloudObject /: FileFormat[co_CloudObject] :=
    Replace[cloudObjectInformation[co, FileFormat, "Elements"->"MIMEType"], fileFormatLookup]

(* FileDate *)
    
CloudObject /: FileDate[obj_CloudObject] :=
    cloudObjectInformation[obj, FileDate, "Elements" -> "LastModified"]   
    
(* FileHash *)

CloudObject /: FileHash[obj_CloudObject] := FileHash[obj, "MD5"]

CloudObject /: FileHash[obj_CloudObject, "MD5"] := cloudObjectInformation[obj, FileHash, "Elements" -> "FileHashMD5"]
    
CloudObject /: FileHash[obj_CloudObject, codeType_] :=
    Module[ {tempfilename, type},
        {tempfilename, type} = readObject[obj];
        If[ tempfilename === $Failed,
            Return[$Failed]
        ];
        cleanup[tempfilename, FileHash[tempfilename, codeType]]
    ]
    
(* DeleteObject *)

CloudObject /: DeleteObject[obj_CloudObject] := deleteObjectOperation[obj, getCloud[obj]]

PermissionsGroup /: DeleteObject[grp_PermissionsGroup] := DeleteObject[CloudObject @@ grp]

PermissionsKey /: DeleteObject[key_PermissionsKey] := deletePermissionsKey[key, DeleteObject]
	
CloudObject`DeleteCloudObjects[objs:{__CloudObject}] :=
	With[{res = Map[deleteObjectOperation[#[[2]], #[[1]]]&, Normal[GroupBy[objs, getCloud[#]&]]]},
		If[MemberQ[res, $Failed], $Failed, Null]
	]

CloudObject`DeletePermissionGroups[groups:{__PermissionsGroup}] := CloudObject`DeleteCloudObjects[CloudObject @@@ groups]
	
CloudObject`DeletePermissionKeys[keys:{__PermissionsKey}] :=
	If[MemberQ[validatePermissionsKey /@ keys, $Failed],
		$Failed, 
		If[MemberQ[Map[deletePermissionsKey[#, DeleteObject]&, keys], $Failed], $Failed, Null]
	]
	
deleteObjectOperation[objs:(_CloudObject | {__CloudObject}), cloud_] :=
	Module[{elements, res},		
		elements = {"UUID", "FileType", "MIMEType"};
        res = Check[cloudObjectInformation[objs, DeleteObject, "Elements" -> elements], $Failed];
        Switch[res,
        	_Association, deleteObject[cloud, Lookup[res, elements]], (* single CloudObject *)
        	_List, Map[deleteObject[cloud, Lookup[#, elements]]&, res], (* list of CloudObjects *)
        	_, $Failed
        ]
	]

deleteObject[cloud_, {uuid_, fileType_, mimeType_}] /; MemberQ[$autoRefreshedMimeTypes, mimeType] :=
	Catch[iCloudRemoveAutoRefreshed[{cloud, uuid}, DeleteObject], $tag]
	
deleteObject[cloud_, {uuid_, fileType_, mimeType_}] /; MemberQ[$taskMimeTypes, mimeType] :=
	Catch[iCloudRemoveScheduledTask[{cloud, uuid}, DeleteObject], $tag]
	
deleteObject[cloud_, {uuid_, fileType:(File | Directory), mimeType_}] := deleteOperation[cloud, uuid, fileType]

deleteObject[cloud_, {uuid_, fileType_, mimeType_}] := Message[CloudObject::srverr]; $Failed
	
deleteOperation[cloud_, uuid_, filter_] := 
	With[{parameters = {"recursive" -> "true", "filter" -> If[filter === File, "file", "directory"]}},
		Replace[
        	execute[cloud, "DELETE", {"files", uuid}, Parameters -> parameters],
        	{
            	{_String, _List} :> Null (* success *),
            	other_ :> (Message[CloudObject::srverr]; $Failed)
        	}
    	]
	]
		
Protect[CloudObject];

getOptionValue[opts_List, optionName_Symbol, funcName_:CloudObject] := OptionValue[funcName, {opts}, optionName]

End[]

EndPackage[]
