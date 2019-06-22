(* ::Package:: *)

BeginPackage["CloudObject`"]

System`CloudGet;
System`CloudPut;
System`CloudSave;
System`IncludeDefinitions;

CloudObject`GetObjectMetadataFilename;

Begin["`Private`"]

Needs["Iconize`"]

Unprotect[CloudObject];

(* general read/write *)

SetAttributes[cleanup, {HoldRest, SequenceHold}];
cleanup[tempfilename_, expr_: Null] :=
	With[{result = expr},
		If[Head[tempfilename] === CloudObject,
			DeleteCloudObject[tempfilename, Asynchronous->True],
			DeleteFile[tempfilename]			
		];
		(* Unevaluated needed here since CompoundExpression doesn't hold sequences  *)
		Unevaluated[result]
	]

readObject::usage = "readObject[obj, head] reads the contents of a cloud object into a temporary file and returns {tempfile, type}.";

readObject[obj_CloudObject, head_Symbol : CloudObject] :=
    responseToFile[
        execute[obj, "ResponseFormat" -> If[TrueQ[$CloudEvaluation], "ByteString", "ByteList"]], 
        head
    ]

writeObject[obj_CloudObject, content_HTTPResponse, rest___] := 
    writeObject[obj, content["BodyBytes"], rest]
writeObject[obj_CloudObject, content_?StringQ, rest___] :=
    writeObject[obj, ToCharacterCode[content], rest] /; !$CloudEvaluation
writeObject[obj_CloudObject, content_, mimetype_,
        permissions_ : Automatic, iconRulesArg_ : None, iconExpr_ : Null, metaInformation_ : {},
        params_ : {}, head_Symbol : CloudObject] :=
    Module[{result, autoIcons, iconRules, encodedMetaInformation},
        If[permissions =!= Automatic && invalidPermissionsGroups[permissions],
            Return[$Failed]
        ];
        {autoIcons, iconRules} = normalizeIconRules[iconRulesArg];
        encodedMetaInformation = encodeMetaInformation[metaInformation];
        If[encodedMetaInformation === $Failed, Return[$Failed]];
        result = Replace[execute[obj, Automatic,
            UseUUID -> False, Body -> content, Type -> mimetype,
            Parameters -> Flatten@Join[params, {
                serverPermissionsForWriting[obj, permissions, mimetype, head],
                If[metaInformation === {}, {}, "properties" -> encodedMetaInformation],
                If[TrueQ[autoIcons], "icons" -> iconSizeString[iconRules], {}] 
            }]], {
        	{_, bytes_List} :> FromCharacterCode[bytes], 
        	HTTPError[400, {___, "errorCode" -> "non-empty-directory",___}, ___] :> (Message[head::nonempdir, obj]; $Failed),
        	other_ :> (checkError[other]; $Failed)}];
        If[result === $Failed, Return[$Failed]];
        If[!TrueQ[autoIcons] && iconRules =!= None,
            SetCloudIcons[obj, iconRules, Asynchronous->True, "Content" -> iconExpr,
                "Deployment" -> iconDeploymentType[mimetype, permissions]]
        ];
        obj
    ]

writeObject[failureObj_Failure, args___] := failureObj
    
serverPermissionsForWriting[obj_CloudObject, permissions_, mimetype_, head_]:=
	Module[{normalizePerms},
		Which[
			(* for "Private" permissions, we don't send any API parameter to the server since this is the default. *)
			(permissions === Automatic) && ($Permissions === "Private"), 
				{},
			permissions === Automatic,
				normalizePerms = Catch[escapeAndNormalizePermissions[$Permissions, mimetype, head], InvalidConstraintPatternValueTag | UnsupportedConstraintPatternTag];
				If[normalizePerms === $Failed, $Failed, "permissionsOnCreate" -> normalizePerms],
			True, 
				normalizePerms = Catch[escapeAndNormalizePermissions[permissions, mimetype, head], InvalidConstraintPatternValueTag | UnsupportedConstraintPatternTag];
				If[normalizePerms === $Failed, $Failed, "permissions" -> normalizePerms]
		]
	]			    

normalizeIconRules[rule_Rule] := normalizeIconRules[{rule}]

normalizeIconRules[Automatic] := {True, $automaticIconRules}

normalizeIconRules[rules:{Rule[_, Automatic]..}] := {True, rules}

normalizeIconRules[rules:{_Rule ..}] := {False, rules}

normalizeIconRules[expr : {} | <||> | None] := {False, None}

normalizeIconRules[expr_] := {False, Table[env -> expr, {env, $cloudIconNames}]}

iconSizeString[sizes_List] := 
	(iconSizeString[sizes] = StringJoin[Riffle[Map[First, sizes], ","]])

(*Put*)

Unprotect[CloudPut];

Options[CloudPut] = objectFunctionOptionsJoin[$objectCreationOptions, {CloudBase -> Automatic, IncludeDefinitions -> False}];
Options[iCloudPut] = Join[Options[CloudPut], {"Append" -> False}];

$SystemIs64Bit = SameQ[$SystemWordLength, 64];
(* Set it to False to avoid serializing expression to MX. *)
$AllowMXCloudPut = True;

iCloudPut[expr_, obj:CloudObject[uri_, objopts:OptionsPattern[CloudObject]], mimetype_String, head_Symbol : CloudPut, opts:OptionsPattern[]] :=
    Module[{content, objNew, optsNew, iconRules, metaInformation, permissions, params, defs, contentSize},
    	Catch[
    		optsNew = prepareOptionsForWriteObject[head, {opts, objopts}];
    		params = {"append" -> exportToJSON[TrueQ[OptionValue["Append"]]]};
    		{permissions, iconRules, metaInformation} = Lookup[optsNew, {Permissions, IconRules, MetaInformation}];
    		
        	If[TrueQ[OptionValue[IncludeDefinitions]],
        	(* save definitions *)
                defs = getDefinitionsList[Unevaluated[expr]];
            ,
        	(* do not save definitions *)
                defs = Language`DefinitionList[];
        	];

            contentSize = ByteCount[defs] + ByteCount[Unevaluated[expr]];
            content = If[TrueQ[$AllowMXCloudPut] && contentSize > 1048576 && mxVersionIsServerCompatibleQ[],
                exprToMXBytes[expr, defs],
                (* Else *)
                exprToByteArrayIncludingDefinitions[Unevaluated[expr], defs]
            ];

        	objNew = writeObject[CloudObject[uri, FilterRules[{objopts}, Except[CloudObjectNameFormat | CloudObjectURLType]]], 
        		content, mimetype, permissions, iconRules, Unevaluated[expr], metaInformation, params, head];
        	(* This additional trip to server will no longer be needed once the api returns richer information instead of just uuid *)
        	If[objNew === $Failed, $Failed,
        		CloudObject[objNew[[1]], CloudObjectNameFormat -> Lookup[optsNew, CloudObjectNameFormat], CloudObjectURLType -> Lookup[optsNew, CloudObjectURLType]]]
			,
     		BadOption | FailedToCloudConnect
     	]
    ]

$IncludedContexts = {}; (* Block override this with {"CloudObject"} to use CloudEvaluate from within CloudObject`Private` code *)
$ExcludeContexts = {"MailReceiver", "CloudSystem", "Forms", "Templating", "Interpreter", "CloudObject", "RLink"};

(*
    neutralContextBlock[expr] evalutes expr without any contexts on the context path,
    to ensure symbols are serialized including their context (except System` symbols).
*)
Attributes[neutralContextBlock] = {HoldFirst};
neutralContextBlock[expr_] := Block[{$ContextPath={"System`"}, $Context="System`"}, expr]

symbolNamesFromDefinitions[defs_Language`DefinitionList] := 
    Cases[defs, (HoldForm[sym_Symbol] -> valuesList_) :> Context[Unevaluated[sym]] <> SymbolName[Unevaluated[sym]]]
    
getDefinitionsList[expr_] :=
    With[
        {excl = Join[
            OptionValue[Language`ExtendedFullDefinition, "ExcludedContexts"],
            $ExcludeContexts
        ]},
        Language`ExtendedFullDefinition[expr, "ExcludedContexts" -> Complement[excl, $IncludedContexts]]
    ]

exprToMXBytes[expr_, defs_Language`DefinitionList] := 
    Block[{mxFileName = CreateTemporary[], CloudObject`Server`$LoadedObject = expr, content},
        (* get symbol names from definitions list and expression *)
        With[{names = Join[{"CloudObject`Server`$LoadedObject"}, symbolNamesFromDefinitions[defs]]},
            DumpSave[mxFileName, names]
        ];
        content = readMXFile[mxFileName];
        (* cleanup *)
        DeleteFile[mxFileName];
        content
    ]

If[$CloudEvaluation,
    readMXFile[fileName_] := ReadString[fileName],
    (* Else *)
    (* Version check for backwards compatibility. *)
    If[$VersionNumber >= 12.,
        readMXFile[fileName_] := ReadByteArray[fileName],
        readMXFile[fileName_] := BinaryReadList[fileName]
    ]
]

exprToStringIncludingDefinitions[expr_, defs_] := 
    Module[{defsString, exprLine},
        defsString = If[defs =!= Language`DefinitionList[],
            neutralContextBlock[With[{d = defs},
                (* Language`ExtendedFullDefinition[] can be used as the LHS of an assignment to restore
                 * all definitions. *)
                toPrintableString[Unevaluated[Language`ExtendedFullDefinition[] = d]]
            ]] <> ";\n\n",
        (* else *)
            ""
        ];
        exprLine = neutralContextBlock[toPrintableString[Unevaluated[expr]]];
        StringTrim[defsString <> exprLine] <> "\n"
    ]
    
exprToStringIncludingDefinitions[expr_] := 
    With[{defs = getDefinitionsList[Unevaluated[expr]]},
        exprToStringIncludingDefinitions[Unevaluated[expr], defs]
    ]
    
exprToStringNotIncludingDefinitions[expr_] :=
	neutralContextBlock[toPrintableString[Unevaluated[expr]]] <> "\n"     

toPrintableString[expr_] := 
    Module[{result = ToString[Unevaluated[expr], InputForm, CharacterEncoding -> "PrintableASCII"]},
        If[PrintableASCIIQ[result],
            result,
        (* Else re-encode with more conservative encoding that will be SyntaxQ *)
            ToString[Unevaluated[expr], InputForm, CharacterEncoding -> "ISO8859-1"]
        ]
    ]

exprToByteArrayIncludingDefinitions[expr_, defs_] := 
    stringToByteArray[exprToStringIncludingDefinitions[Unevaluated[expr], defs], "ISO8859-1"]

(* for backwards compatibility *)	
If[$VersionNumber >= 12.,
    stringToByteArray[expr_String, encoding_] := StringToByteArray[expr, encoding](* ByteArray for URLFetch is supported in M12.+ *),
(* Else simply get integer codes for each string character for pre 12. versions *)
    stringToByteArray[expr_String, encoding_] := ToCharacterCode[expr, encoding]
]

exprToStringBytesIncludingDefinitions[expr_, defs_] := 
    ToCharacterCode[exprToStringIncludingDefinitions[Unevaluated[expr], defs], "UTF-8"]
    
exprToStringBytesNotIncludingDefinitions[expr_] :=
	ToCharacterCode[exprToStringNotIncludingDefinitions[Unevaluated[expr]], "UTF-8"]
    
saveDefToIncludeDef[opts_List] := Replace[Flatten[opts], Rule[SaveDefinitions, value_] :> Rule[IncludeDefinitions, value], {1}]  

getCloudVersionNumber[] := getCloudVersionNumber[$CloudVersionNumber]

getCloudVersionNumber[version:None] := 
    With[{con = CloudConnect[]},
        If[con === $CloudUserID,
            getCloudVersionNumber[$CloudVersionNumber],
        (* Else not connected to cloud *)
            Throw[$Failed, FailedToCloudConnect]
        ]
    ]

getCloudVersionNumber[version_String] := getCloudVersionNumber[version] = 
     If[StringMatchQ[version, (DigitCharacter ..) | (DigitCharacter .. ~~ ("." ~~ DigitCharacter ..) ..)],
        Interpreter["Number"][First@StringCases[version, NumberString, 1]],
        Infinity
     ]

getCloudVersionNumber[version_] := Infinity

getCloudKernelVersionNumber[] := 
    If[NumberQ[$CloudKernelVersionNumber],
        $CloudKernelVersionNumber,
    (* Else *)
        $CloudKernelVersionNumber = If[$CloudConnected, CloudEvaluate[$VersionNumber], None]
    ]

cloudKernelVersionCompatibleQ[] := cloudKernelVersionCompatibleQ[] = 
    With[{versionNumber = getCloudKernelVersionNumber[]},
        Or[TrueQ[versionNumber >= $VersionNumber], versionNumber === 11.3]
    ]
    
mxVersionIsServerCompatibleQ[] := 
    AllTrue[{(getCloudVersionNumber[] >= 1.49), cloudKernelVersionCompatibleQ[], $SystemIs64Bit}, TrueQ]

Options[cloudPut] = Options[CloudPut];

cloudPut[expr_, opts : OptionsPattern[]] := 
	Block[ {$CloudBase = handleCBase[OptionValue[CloudBase]]},
            CloudPut[Unevaluated[expr], CloudObject[], Sequence @@ FilterRules[opts, Except[CloudBase]]]
        ]
        
cloudPut[expr_, uri_String, opts:OptionsPattern[]] :=
	 Block[ {$CloudBase = handleCBase[OptionValue[CloudBase]]},
            CloudPut[Unevaluated[expr], CloudObject[uri], Sequence @@ FilterRules[opts, Except[CloudBase]]]
        ]       

CloudPut[expr_, opts : OptionsPattern[]] :=
    cloudPut[Unevaluated[expr], saveDefToIncludeDef[{opts}]]

CloudPut[expr_, obj_CloudObject, opts:OptionsPattern[]] :=
	iCloudPut[Unevaluated[expr], obj, expressionMimeType["Expression"], CloudPut, saveDefToIncludeDef[{opts}]]
    
CloudPut[expr_, uri_String, opts:OptionsPattern[]] :=
    cloudPut[Unevaluated[expr], uri, saveDefToIncludeDef[{opts}]]
    
CloudPut[expr_, URL[dest_String], opts:OptionsPattern[]] := 
	CloudPut[Unevaluated[expr], dest, opts]    

CloudPut[expr_, obj_, opts:OptionsPattern[]]:=
    (Message[CloudPut::invcloudobj, obj];$Failed)

CloudPut[args___] := (ArgumentCountQ[CloudPut,Length[DeleteCases[{args},_Rule,Infinity]],1,2];Null/;False)

CloudPut[expr_, failureObj_Failure, opts:OptionsPattern[]] := failureObj

CloudObject /: Put[expr_, obj_CloudObject] := CloudPut[Unevaluated[expr], obj]

CloudObject /: PutAppend[expr_, obj_CloudObject] := iCloudPut[Unevaluated[expr], obj, expressionMimeType["Expression"], PutAppend, "Append" -> True]

SetAttributes[CloudPut, {ReadProtected}];
Protect[CloudPut];

(*Save*)

Unprotect[CloudSave];

Options[CloudSave] = objectFunctionOptionsJoin[$objectCreationOptions, {CloudBase -> Automatic}];
Attributes[CloudSave] = {HoldFirst};

CloudSave[expr_, obj:CloudObject[uri_, objopts:OptionsPattern[CloudObject]], opts:OptionsPattern[]] :=
	Module[{type, content, tempfilename, objNew, optsNew, permissions, iconRules, metaInformation},
		Catch[
			optsNew = prepareOptionsForWriteObject[CloudSave, {opts, objopts}];
    		{permissions, iconRules, metaInformation} = Lookup[optsNew, {Permissions, IconRules, MetaInformation}];
    		Block[{$CloudBase = handleCBase[OptionValue[CloudBase], CloudSave]},
        		If[FileExistsQ[obj],
            		{tempfilename, type} = readObject[obj, CloudSave];
            		If[tempfilename === $Failed, Return[$Failed]],
        		(* else *)
            		tempfilename = CreateTemporary[]
        		];
        		Save[tempfilename, Unevaluated[expr]];
	       		content = BinaryReadList[tempfilename];
        		objNew = writeObject[CloudObject[uri, FilterRules[{opts}, Except[CloudObjectNameFormat]]], content, expressionMimeType["Expression"],
            		permissions, iconRules, Unevaluated[expr], metaInformation, {}, CloudSave];
        		(* This additional trip to server will no longer be needed once the api returns richer information instead of just uuid *)
        		If[objNew === $Failed, $Failed, CloudObject[objNew[[1]], CloudObjectNameFormat -> Lookup[optsNew, CloudObjectNameFormat]]]
        	],
        	BadOption
        ]
	]

CloudSave[expr_, uri_String, opts:OptionsPattern[]] := 
	Block[ {$CloudBase = handleCBase[OptionValue[CloudBase]]}, CloudSave[expr, CloudObject[uri], opts]]

CloudSave[expr_, URL[uri_String], opts:OptionsPattern[]] := CloudSave[expr, uri, opts]

CloudSave[expr_, opts:OptionsPattern[]] := 
	Block[ {$CloudBase = handleCBase[OptionValue[CloudBase]]}, CloudSave[expr, CloudObject[], opts]]

CloudSave[args___] := (ArgumentCountQ[CloudSave,Length[DeleteCases[{args},_Rule,Infinity]],1,2];Null/;False)

CloudSave[expr_, failureObj_Failure, opts:OptionsPattern[]] := failureObj

CloudObject /: Save[obj_CloudObject, expr_] := CloudSave[expr, obj]

SetAttributes[CloudSave, {ReadProtected}];
Protect[CloudSave];

(*Get*)

Unprotect[CloudGet];

Options[CloudGet] = {CloudBase->Automatic};

bundleMimeTypeQ[mimetype_] :=
    StringQ[mimetype] &&
        StringMatchQ[mimetype, "application/vnd.wolfram.bundle" ~~ ___]

CloudGet[co_CloudObject, opts:OptionsPattern[]] :=
    Module[{tempfilename, mimetype},
        {tempfilename, mimetype} = readObject[co, CloudGet];
        Which[
            tempfilename === $Failed, $Failed,

            mimetype === "inode/directory", Message[Get::noopen, co]; $Failed,

            bundleMimeTypeQ[mimetype], CloudGet[FileNameJoin[{co, ".bundle"}]],

            True, cleanup[tempfilename, Block[{$CharacterEncoding = "UTF-8"},
                getMxOrContentFile[tempfilename]
            ]]
        ]
    ];

CloudGet[uri_String, opts:OptionsPattern[]] :=
    Block[{$CloudBase = handleCBase[OptionValue[CloudBase]]},
        CloudGet[CloudObject[uri]]
    ]

CloudGet[URL[uri_String], opts:OptionsPattern[]] := CloudGet[uri]

CloudObject /: Get[co_CloudObject] := CloudGet[co]

CloudGet[args___] := (ArgumentCountQ[CloudSave,Length[DeleteCases[{args},_Rule,Infinity]],1,1];Null/;False)

CloudGet[failureObj_Failure, opts:OptionsPattern[]] := failureObj

SetAttributes[CloudGet,{ReadProtected}];
Protect[CloudGet];

Protect[CloudObject];

getMxOrContentFile[file_] := 
    Block[{CloudObject`Server`$LoadedObject, res},
        res = Get[file];
        If[ValueQ[CloudObject`Server`$LoadedObject],
            CloudObject`Server`$LoadedObject,
        (* Else file is a content file *)
            res
        ]
    ]

(* From Jan, plus a tiny amount of error checking, and also allow CloudObject or UUID. *)
GetObjectMetadataFilename[obj_CloudObject, subpath___String] :=
    Replace[getCloudAndUUID[obj], {
                {_, uuid_} :> GetObjectMetadataFilename[uuid, subpath],
                _ :> $Failed
    }];

GetObjectMetadataFilename[uuid_String?UUIDQ, subpath___String] :=
    (* TODO (Jan?): does this need to use the $HomeDirectory of the CloudObject owner, not the caller?
       Or is that tautological? *)
    FileNameJoin[{$HomeDirectory, ".Objects", "metadata", StringTake[uuid, 3], uuid, subpath}];

GetObjectMetadataFilename[___] := $Failed;

GetObjectMetadataFilename[failureObj_Failure, arg___] := failureObj

(* It Throws with the BadOption tag if there is invalid option value. *)
prepareOptionsForWriteObject[msghd_, opts___?OptionQ] :=
    Module[ {optsNew, metaNew},
        optsNew = Association[Options[Replace[msghd, PutAppend -> iCloudPut]], opts]; (* PutAppend calls iCloudPut *)
        (* CloudPut/CloudExport might inheritate Appearance option from CloudDeploy, so we put AppearanceRules information first *)
        metaNew = Join[{handleAppearanceRules[optsNew[AppearanceRules], msghd]}, handleMetaInfo[optsNew[MetaInformation], msghd]];
        permissionsFormatValidation[optsNew[Permissions], msghd];
        Join[FilterRules[Normal[optsNew], {CloudObjectNameFormat, CloudObjectURLType, IconRules}], {MetaInformation -> metaNew, Permissions -> optsNew[Permissions]}]
    ]

End[]

EndPackage[]
