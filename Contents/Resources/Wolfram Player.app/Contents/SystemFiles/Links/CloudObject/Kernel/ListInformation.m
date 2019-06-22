(* ::Package:: *)

BeginPackage["CloudObject`"]

System`CloudObjectInformation;
System`CloudObjectInformationData;
System`CloudObjects;

CloudObject`CloudObjectUUIDForm;

System`CloudObjectInformation::usage = "CloudObjectInformation[obj] gives information about a cloud object.
CloudObjectInformation[{obj, ...}] gives information about a list of objects.
CloudObjectInformation[obj, \"field\"] gives the information for a specific field.
CloudObjectInformation[obj, {field, ...}] limits the information to a selection of fields.
CloudObjectInformation[{obj, ...}, {field, ...}] gets a selection of information for a list of objects.
CloudObjectInformation[dir, \"DirectoryLevel\" -> 0] gives information about a directory. (This is the default.)
CloudObjectInformation[dir, \"DirectoryLevel\" -> 1] gives information about all objects in a given directory (but not in subdirectories).
CloudObjectInformation[dir, \"DirectoryLevel\" -> Infinity] gives information about all objects in a given directory subtree.
CloudObjectInformation[type, {field, ...}] gives information about all objects of a given MIME type.
CloudObjectInformation[type, field] gives a specific field of information about all objects of a given MIME type."

Begin["`Private`"]

(* CloudObjects *)

Options[CloudObjects] = {"Directory" -> Automatic, "Type" -> All, CloudBase -> Automatic, CloudObjectNameFormat -> Automatic}

queryTypeValue["CloudEvaluation"] := expressionMimeType["CloudEvaluation"];
queryTypeValue["Expression"|Expression] := expressionMimeType[Expression];
queryTypeValue["Notebook"|Notebook] := expressionMimeType[Notebook];
queryTypeValue["ExternalBundle"|ExternalBundle] := expressionMimeType[ExternalBundle];
queryTypeValue["Directory"|Directory] := expressionMimeType[Directory];
queryTypeValue[type_String] := formatToMimeType[type];
queryTypeValue[symbol_Symbol] := expressionMimeType[symbol];
queryTypeValue[All] = All;
queryTypeValue[Verbatim[Alternatives][types___]] := queryTypeValue[{types}];
queryTypeValue[list_List] := If[MemberQ[list, All], All, StringRiffle[Map[queryTypeValue, list], ","]];
queryTypeValue[_] = $Failed;

iCloudObjects[cloud_String, path_, opts:OptionsPattern[CloudObjects]] :=
    Module[{query = {}, typevalue, type, nameFormat},
        typevalue = OptionValue[CloudObjects, {opts}, "Type"];
        type = queryTypeValue[typevalue];
        If[type === $Failed,
            Message[CloudObjects::invtype, typevalue];
            type = All;
        ];
        nameFormat = Replace[OptionValue[CloudObjectNameFormat], Automatic -> $CloudObjectNameFormat];
        If[path === "", nameFormat = "UUID"];
        If[type =!= All, AppendTo[query, "mimeType" -> type]];
        If[path =!= All, AppendTo[query, "path" -> path]];
        AppendTo[query, "fields" -> If[nameFormat === "UUID", "uuid", "path,owner,uuid"]];
        Replace[responseToString[execute[cloud, "GET", {"files"}, Parameters -> query], CloudObjects],
        		result_String :> fileInfoToCloudObjects[cloud, result, nameFormat]
        	]
    ]
    
CloudObjects[All, opts:OptionsPattern[]] := 
	Block[{$CloudBase = handleCBase[OptionValue[CloudBase]]},
		iCloudObjects[$CloudBase, All, optsNoBase[{opts}]]]
		
CloudObjects[None, opts:OptionsPattern[]] := 
	Block[{$CloudBase = handleCBase[OptionValue[CloudBase]]},
		iCloudObjects[$CloudBase, "", optsNoBase[{opts}]]]
		
CloudObjects[obj_CloudObject, opts:OptionsPattern[]] :=
	Block[{$CloudBase = handleCBase[OptionValue[CloudBase]]},
    	Module[{cloud, path, name},
        	{cloud, path} = getCloudAndPathList[obj];
        	name = StringJoin[Riffle[Join[path, {"*"}], "/"]];
        	iCloudObjects[cloud, name, optsNoBase[{opts}]]
    	]
	]

CloudObjects[Automatic, opts:OptionsPattern[]] := 
	Block[{$CloudBase = handleCBase[OptionValue[CloudBase]]}, CloudObjects[CloudDirectory[], optsNoBase[{opts}]]]
	
CloudObjects[dir_String, opts:OptionsPattern[]] := 
	Block[{$CloudBase = handleCBase[OptionValue[CloudBase]]}, CloudObjects[CloudObject[dir], optsNoBase[{opts}]]]
	
CloudObjects[URL[url_String], opts:OptionsPattern[]] := CloudObjects[url, opts]

(* If no directory is given as positional argument, take the option value (with default Automatic). *)
CloudObjects[opts:OptionsPattern[]] := CloudObjects[OptionValue["Directory"], opts]

(* Expand an Association out into a list of rules, which get treated as options. *)
CloudObjects[before___, assoc_Association, after___] := CloudObjects[before, Normal[assoc], after]

CloudObjects[dir_, type:(_String|_Symbol|_Alternatives), opts:OptionsPattern[]] := CloudObjects[dir, "Type" -> type, opts]

CloudObjects[failureObj_Failure, args___] := failureObj

optsNoBase[opts_List] := Sequence @@ FilterRules[{opts}, Except[CloudBase]]	
	
(* List objects *)
CloudObjectsByType[contentType_String] :=
    Module[{response, uuids},
        response = responseToString @ execute[$CloudBase, "GET", {"files"},
            Parameters->{"mimeType" -> contentType}];
        If[!StringQ[response], Return[$Failed]];
        uuids = Map[FileNameTake[#, -1]&, StringSplit[response]];
        Map[cloudObjectFromUUID, uuids]
    ]
    
fileInfoToCloudObjects[cloud_, uuidsJSON_String, "UUID"] :=
	With[{uuids = importFromJSON[uuidsJSON]},
		If[ListQ[uuids],
			Map[cloudObjectFromUUID[cloud, #]&, uuids],
			{}
		]
	] 
  
fileInfoToCloudObjects[cloud_, info_String, nameFormat_] :=
	With[{result = importFromJSON[info]},
		If[ListQ[result],
			Map[cloudObjectFromPathInfo[cloud, #, nameFormat]&, result],
			{}]
	]

(**************************)
(* CloudObjectInformation *)

Options[CloudObjectInformation] = {"DirectoryLevel" -> 0}

(* in future, support a CloudObjectInformation[], 0-arg form that uses CloudDirectory[] *)

(* CloudObjectInformation[obj], 1-arg form *)

CloudObjectInformation[failureObj_Failure, arg___] := failureObj

CloudObjectInformation[failureObjs:{_Failure ..}, arg___] := failureObjs

CloudObjectInformation[failureObjs:{_Failure ..}] := failureObjs

CloudObjectInformation[obj_CloudObject, opts:OptionsPattern[]] := cloudObjectInformation[obj, CloudObjectInformation, opts]

CloudObjectInformation[obj_CloudObject, "UUID"] :=
    With[{result = Quiet[getCloudAndUUID[obj]]},
        If[MatchQ[result, {_String, _?CloudObject`UUIDQ}],
            Last[result],
        (* Else *)
            Message[CloudObjectInformation::cloudnf, obj];
            $Failed
        ]
    ]

CloudObjectInformation[obj_CloudObject, property_String, opts:OptionsPattern[]] :=
    cloudObjectInformation[obj, CloudObjectInformation, "Elements" -> property, opts]

CloudObjectInformation[obj_CloudObject, properties:{_String ..}, opts:OptionsPattern[]] :=
    cloudObjectInformation[obj, CloudObjectInformation, "Elements" -> properties, opts]

(* CloudObjectInformation[{obj,...}, ...], n-arg list form *)

CloudObjectInformation[{}, ___] := {}

CloudObjectInformation[objects:{(_CloudObject |_Failure) ..}] := cloudObjectInformation[objects, CloudObjectInformation]

CloudObjectInformation[objects:{(_CloudObject | _Failure) ..}, property_String] :=
    cloudObjectInformation[objects, CloudObjectInformation, "Elements" -> property]

CloudObjectInformation[objects:{(_CloudObject | _Failure) ..}, properties:{_String ..}] :=
    cloudObjectInformation[objects, CloudObjectInformation, "Elements" -> properties]

(* CloudObjectInformation[type, ...],  *)

CloudObjectInformation[type_String, property_String] := 
    cloudObjectInformation[type, CloudObjectInformation, "Elements" -> property]

CloudObjectInformation[type_String, properties:{_String ..}] := 
    cloudObjectInformation[type, CloudObjectInformation, "Elements" -> properties]

Options[cloudObjectInformation] = {"Elements" -> Automatic}

(*****************************************)
(* cloudObjectInformation helper function *)

Options[cloudObjectInformation] = Join[{"Elements" -> Automatic}, Options[CloudObjectInformation]]

cloudObjectInformation[obj_CloudObject, msghd_, opts:OptionsPattern[]] :=
    Catch[
        Module[{directoryLevel, elts, cloud, uuid, path, queryParameters, json, files, infoList, result},
            elts = OptionValue["Elements"];
            directoryLevel = handleDirectoryLevelOption[OptionValue["DirectoryLevel"], msghd];

            result = Replace[getCloudAndUUIDOrPath[obj], {
                {cloud_String, uuid_, pathelts_List} :> {cloud, uuid, StringRiffle[pathelts, "/"]},
                {cloud_String, uuid_, pathelts_} :> {cloud, uuid, pathelts},
                other_ :> handleObjectNotFound[msghd, obj]
            }];

            {cloud, uuid, path} = result;
            If[!StringQ[uuid] && !StringQ[path],
                handleObjectNotFound[msghd, obj]
            ];

            Check[
            	queryParameters = fileQueryAPIParameters[uuid, path, directoryLevel, elts], 
                Return[$Failed]
            ];

            json = Replace[execute[cloud, "GET", {"files"}, Parameters -> queryParameters], {
                {_String, content_List} :> FromCharacterCode[content],
                (* we get 400 for object by UUID, 404 for object by name *)
                HTTPError[400 | 404, ___] :> handleObjectNotFound[msghd, obj],
                HTTPError[403, ___] :> handleForbiddenObject[msghd, obj],
                other_ :> handleBadServerResponse[msghd]}
            ];
            
            files = expectForm[importFromJSON[json],
                _List,
                handleBadServerResponse[msghd]
            ];
            
            If[directoryLevel === 0 && files === {},
                handleObjectNotFound[msghd, obj]
            ];
            
            infoList = Map[objectInfo[#, "Elements" -> elts] &, files];

            If[directoryLevel === 0,
                First[infoList],
                infoList
            ]
	    ],
	    BadOption | BadServerResponse | ObjectNotFound | ForbiddenObject
    ]

handleBadServerResponse[msghd_] := (Message[msghd::srverr]; Throw[$Failed, BadServerResponse])

handleObjectNotFound[msghd_, obj_] := (Message[msghd::cloudnf, obj]; Throw[$Failed, ObjectNotFound])

handleForbiddenObject[msghd_, obj_] := (Message[msghd::notperm, obj]; Throw[$Failed, ForbiddenObject])

SetAttributes[expectForm, HoldAllComplete]
expectForm[expr_, pattern_, onfail_] := 
    Module[{result = expr},
    	If[MatchQ[result, pattern],
    		result,
    		onfail
    	]
    ]

handleDirectoryLevelOption[value:(0|1|Infinity), _] := value

handleDirectoryLevelOption[other_, msghd_] := (
    Message[msghd::dirlvl, other];
    Throw[$Failed, BadOption]
)

(* object is given with a uuid URL and info only for the object itself is requested *)
fileQueryAPIParameters[uuid_String, _, 0, elements_] :=
    {"uuid" -> uuid, "fields" -> fileQueryAPIFields[elements]}

(* object is given with a uuid URL and we need a wildcard query; here, the uuid becomes the path *)
fileQueryAPIParameters[uuid_String, _, dirlevel_, elements_] :=
    {"path" -> fileQueryAPIPath[uuid, dirlevel], "fields" -> fileQueryAPIFields[elements]}

(* object is given with a named URL *)
fileQueryAPIParameters[_, path_String, dirlevel_, elements_] := 
	{"path" -> fileQueryAPIPath[path, dirlevel], "fields" -> fileQueryAPIFields[elements]}

fileQueryAPIPath[path_String, dirlevel_] :=
    (* We don't use URLBuild here because it does URL encoding, which will happen later, and corrupts the * character. *)
    FileNameJoin[{path, Replace[dirlevel, {0 -> Nothing, 1 -> "*", Infinity -> "**", _ -> ""}]}, 
 	  OperatingSystem -> "Unix"]

fileQueryAPIFields[elements_] := commaSeparated[resolveInfoFields[elements]]

cloudObjectInformation[objects:{(_CloudObject | _Failure) ..}, msghd_, opts:OptionsPattern[]] :=
    Module[{args, bad, uuids, cloud, elements, fields, json, files, validCloudObjects, infoByUUID, 
        coInfoAssociation},

    	validCloudObjects = Cases[objects, _CloudObject];
		args = Map[Prepend[getCloudAndUUID[#], #]&, validCloudObjects]; (* this can be slow if objects are named, as they would be from CloudObjects[] *)

        (* test for objects that cannot be resolved *)
        bad = SelectFirst[args, ! MatchQ[#, {_, _String, _?CloudObject`UUIDQ}] &];

        If[Head[bad] =!= Missing,
            (* this indicates we were provided at least one named object that doesn't exist *)
            Message[msghd::cloudnf];
            Return[$Failed]
        ];

        uuids = Map[Last, args];
        cloud = args[[1, 2]]; (* assumes all objects are in the same cloud *)
        elements = OptionValue["Elements"];
        fields = resolveInfoFields[elements]; (* note we will additionally request uuid *)

        json = Replace[
            execute[
                cloud, "GET", {"files"}, 
                Parameters -> {"fields" -> commaSeparated[Union[{"uuid"}, fields]], (* fields also indicates v2 of the API, to return JSON *)
                "uuid" -> commaSeparated[uuids]}
            ],
            {
                HTTPError[404, ___] :> (
                    Message[msghd::cloudnf, objects]; 
                    Return[$Failed]),
                {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content]),
                other_ :> (
                    $lastInfoResult = other;
                    Message[msghd::srverr]; 
                    Return[$Failed]
                )
            }
        ];

        files = importFromJSON[json];
        If[!ListQ[files],
            Message[msghd::srverr];
            Return[$Failed]
        ];
        
        (* Turn the list of object data into a map from UUID to info, which can be used to answer
           whether or not the object was found (it was found if returned by the server). *)
        infoByUUID = Association[Map[Lookup[#, "uuid"] -> # &, files]];

        (* Make an association from CloudObject to either its info (if found) or $Failed (if not). *)
        coInfoAssociation = Map[
            Function[rec,
                With[{obj = First[rec], uuid = Last[rec]},
                    obj -> Replace[infoByUUID[uuid], {
                        _Missing :> (Message[CloudObjectInformation::cloudnf, obj]; $Failed),
                        info_ :> objectInfo[info, "Elements" -> elements]
                    }]
                ]
            ],
            args
        ];

        Lookup[coInfoAssociation, #, #] & /@ objects
    ]

cloudObjectInformation[type_String, msghd_, opts:OptionsPattern[]] := 
    Module[{elements, fields, json, info},

        elements = OptionValue["Elements"];
        fields = resolveInfoFields[elements];
        
        json = Replace[
            execute[
                $CloudBase, "GET", {"files"}, 
                Parameters -> {
                    "fields" -> commaSeparated[fields], (* indicate v2 of the API, to return JSON *)
                    "mimeType" -> formatToMimeType[type]
                }
            ],
            {
                HTTPError[404, ___] :> (
                    Message[msghd::cloudnf, objects]; 
                    Return[$Failed]),
                {_String, content_List} :>
                    ($lastInfoJSON = FromCharacterCode[content]),
                other_ :> (
                    $lastInfoResult = other;
                    Message[msghd::srverr]; 
                    Return[$Failed])
            }
        ];

        info = importFromJSON[json];
        If[!ListQ[info],
            Message[msghd::srverr];
            Return[$Failed]
        ];

        Map[objectInfo[#, "Elements" -> elements]&, info]
    ]

commaSeparated[elts_List] := StringJoin[Riffle[Cases[elts, _String], ","]]

resolveInfoFields[Automatic] := {"all"}

resolveInfoFields[field_String] := resolveInfoFields[{field}]

resolveInfoFields[fields_List] := 
    Map[Lookup[$jsonFields, #, handleUnknownProperty[#]]&, fields]

handleUnknownProperty[property_String] := 
(
    Message[CloudObjectInformation::noprop, property];
    $Failed
)

Options[objectInfo] = {"Elements" -> Automatic}

objectInfo[info_List, opts:OptionsPattern[]] := objectInfo[Association[info], opts]

objectInfo[info_Association, OptionsPattern[]] := 
    Module[{elements = OptionValue["Elements"], mimetype = Lookup[info, "mimeType", None], 
        displayName, infoData = <||>},
        displayName = Lookup[info, "displayName", info["name"]];
        
        Do[
            infoData[elt] = 
            <|
                "UUID" -> info["uuid"],
                "Path" -> info["path"] /. {Null -> None},
                "Name" -> displayName,
                "DisplayName" -> displayName,
                "OwnerWolframUUID" -> info["ownerUUID"],
                "OwnerWolframID" :> Lookup[info["owner"], "email", Missing["Unavailable"]],
                "MIMEType" -> mimetype,
                "MimeType" -> mimetype,
                "FileType" ->
                    If[mimetype === "inode/directory" || bundleMimeTypeQ[mimetype],
                        Directory,
                        File
                    ],
                "FileByteCount" :> FromDigits[info["fileSize"]],
                "Created" :> DateObject[info["created"]],
                "LastAccessed" :> DateObject[info["lastAccessed"]],
                "LastModified" :> DateObject[info["lastModified"]],
                "FileHashMD5" :> 
                	With[{hash = info["fileHashMD5"]}, Replace[hash, {x_String :> FromDigits[x, 16], Null -> None}]],
                "Permissions" :> fromServerPermissions[info["filePermission"]],
                "Active" -> info["active"]
            |>[elt],
            {elt, Switch[elements,
                Automatic, Keys[$jsonFields],
                _String, {elements},
                _, elements]
            }
        ];
 
        Switch[elements,
            Automatic, System`CloudObjectInformationData[infoData],
            _String, First[Values[infoData]],
            _List, infoData
        ]
    ]

(* $jsonFields is used to assist in field selection.
 For each CloudObjectInformation property on the left-hand side, 
 it says what json field on the right-hand side we should ask from the
 server from which to derive that property. This allows us to efficiently
 request only the info fields from the server that are needed to return
 the properties requested in WL.
 *)
$jsonFields = <|
    "UUID" -> "uuid",
    "Path" -> "path",
    "Name" -> "displayName",
    "DisplayName" -> "displayName",
    "OwnerWolframUUID" -> "ownerUUID",
    "OwnerWolframID" -> "owner",
    "MIMEType" -> "mimeType",
    "MimeType" -> "mimeType",
    "FileType" -> "mimeType",
    "FileByteCount" -> "fileSize",
    "FileHashMD5" -> "fileHashMD5",
    "Created" -> "created",
    "LastAccessed" -> "lastAccessed",
    "LastModified" -> "lastModified",
    "Permissions" -> "filePermission",
    "Active" -> "active"
|>

(* TODO: This should probably be exposed through CloudObjectInformation, and maybe as a sub value of CloudObject.
 Maybe also as Normal[obj]. *)
CloudObjectUUIDForm[obj : CloudObject[url_, opts___]] :=
    Module[{cloud, uuid},
        {cloud, uuid} = getCloudAndUUID[obj];
        If[!StringQ[cloud], Return[$Failed]];
        CloudObject[URLBuild[{cloud, $cloudObjectRootPath, uuid}], opts]
    ]

(* work around URLBuild bug https://bugs.wolfram.com/show?number=340917, that adds / without checking for / in arguments *)
$cloudObjectRootPath = StringReplace[CloudObject`Private`$CloudObjectsRoot, StartOfString ~~ "/" -> ""]

CloudObjectUUIDForm[group_PermissionsGroup] := PermissionsGroup @@ CloudObjectUUIDForm[CloudObject @@ group]

CloudObjectUUIDForm[failureObj_Failure] := failureObj

End[]

EndPackage[]
