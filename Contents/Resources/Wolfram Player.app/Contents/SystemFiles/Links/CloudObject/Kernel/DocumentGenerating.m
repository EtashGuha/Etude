(* ::Package:: *)

(* Mathematica package *)
BeginPackage["CloudObject`"];

System`DocumentGenerator;
System`DocumentGeneratorInformation::usage = "DocumentGeneratorInformation[CloudObject] returns information about a generator.
DocumentGeneratorInformation[CloudObject, property] returns the value of the specified property.";
System`DocumentGeneratorInformationData;
System`DocumentGenerators::usage = "DocumentGenerators[] returns a list of the user's document generators, as cloud objects.";

(* Option symbols *)
System`DeliveryFunction;
System`GeneratorDescription;
System`EpilogFunction;
System`GeneratorHistoryLength;
System`GeneratorOutputType;

System`GeneratedDocumentBinding;

Begin["`Private`"];


Unprotect[DocumentGenerator, DocumentGenerators, DocumentGeneratorInformation, CloudObject];


Unprotect[DocumentGenerator, CloudObject];
SetAttributes[DocumentGenerator, {HoldAll, ReadProtected}];

simpleLog[args___] := Print[
    DateString[{"DateShort", " ", "Time"}],
    "\t",
    Apply[Sequence, Replace[{args}, s_StringForm :> ToString[s], {1}]]
];

$docGenMimeTypes = {"application/vnd.wolfram.bundle.document-generator"};
(* Slow! *)
documentGeneratorQ[co_CloudObject] :=
    With[{mime = Quiet[Check[CloudObjectInformation[co, "MIMEType"], $Failed]]},
    	MemberQ[$docGenMimeTypes, mime]
    ];
documentGeneratorQ[_] := False;

Options[DocumentGenerator] = Sort @ Join[{
        GeneratorDescription -> None,
        EpilogFunction -> None,
        GeneratorHistoryLength -> Infinity,
        DeliveryFunction -> None,

        GeneratorOutputType -> "CloudCDF",
        Permissions -> Automatic (* Applies to generated documents *)
    },
    Options[ScheduledTask]
];

driverTypeQ[x_] := MatchQ[x, Alternatives @@ {
    None,
    _File, Delayed[_File],
    _CloudObject, Delayed[_CloudObject],
    _Association (*, Delayed[_Association] *),
    _Function
}];

templateTypeQ[x_] := MatchQ[x, Alternatives @@ {
    _File, Delayed[_File],
    _CloudObject, Delayed[_CloudObject],
    _TemplateObject (* Delayed[_TemplateObject] *)
}];

epilogTypeQ[x_] := MatchQ[x, Alternatives @@ {
    None,
    _File, Delayed[_File],
    _CloudObject, Delayed[_CloudObject],
    _Function,
    _Missing (* Not supplied in options *)
}];

(* Upload local files from a desktop client, take care of as many cases as possible
 * on the server.
 *)
resolveDocGenResource[res_] := Check[
    Replace[res, {
        File[f_] :> CloudObject`Private`deleteable@CopyFile[FindFile[f], CloudObject[]],
        p:CloudObject`Private`preexisting[_] :> p,
        t:Except[_Delayed|_CloudObject|None] :> CloudObject`Private`deleteable@CloudPut[t, IncludeDefinitions -> True]
    }],
    Throw[$Failed, $tag]
];

toBeCopiedQ[_File] := True;
toBeCopiedQ[_CloudObject] := True;
toBeCopiedQ[CloudObject`Private`deleteable[_]] := True;
toBeCopiedQ[CloudObject`Private`preexisting[_]] := False;
toBeCopiedQ[_] := False;

$deliveryFormats = Join[ToUpperCase[$ExportFormats], ToLowerCase[$ExportFormats]];(*{"NB", "PDF", "CDF", Null, None};*)
$outputFormats = {"CloudCDF", "PDF", "CDF", "StaticPage"};

validateDocGenArg["options", {{}, r_}] := True
validateDocGenArg["options", {unknown_List, r_}] := (Message[DocumentGenerator::optx, First[unknown], r]; False)
(* No way to evaluate desktop file at runtime: *)
validateDocGenArg[a:"template"|"driver"|"epilog", Delayed[f_File]] := If[Not[TrueQ[$CloudEvaluation]],
    Message[DocumentGenerator::badarg, f, a]; False,
    True
]
validateDocGenArg[a:"template"|"driver"|"epilog", preexisting[p_]] := validateDocGenArg[a, p]
validateDocGenArg["template", _?templateTypeQ] := True
validateDocGenArg["driver", _?driverTypeQ] := True
validateDocGenArg["epilog", _?epilogTypeQ] := True
validateDocGenArg["notification", Automatic|All|None] := True
validateDocGenArg["notification", _String -> Automatic|All] := True
validateDocGenArg["notification", {___String}] := True
validateDocGenArg["notification", {({__String} -> (Automatic|All|None|_Function|_List))..}] := True
validateDocGenArg["notification", _Function] := True
validateDocGenArg["notification", a__] := (Message[DocumentGenerator::badarg, a, NotificationFunction]; False)
validateDocGenArg["delivery", Automatic|None] := True
validateDocGenArg["delivery", Alternatives @@ $deliveryFormats] := True
validateDocGenArg["delivery", {___String}] := True
validateDocGenArg["delivery", {__String} -> Alternatives @@ $deliveryFormats | Automatic| _Function] := True
validateDocGenArg["delivery", {({__String} -> Alternatives @@ $deliveryFormats | Automatic| _Function)..}] := True
validateDocGenArg["delivery", _String -> Alternatives @@ $deliveryFormats | Automatic] := True    (* non-email channel delivery *)
validateDocGenArg["delivery", _Function] := True
validateDocGenArg["delivery", a__] := (Message[DocumentGenerator::badarg, a, DeliveryFunction]; False)
validateDocGenArg["outputformat", Alternatives @@ $outputFormats] := True
validateDocGenArg["outputformat", f_] := (Message[DocumentGenerator::badarg, f, GeneratorOutputType]; False)
validateDocGenArg["historylength", _Integer?(# >= 1 &)] := True
validateDocGenArg["historylength", Infinity|DirectedInfinity] := True
validateDocGenArg["historylength", h_] := (Message[DocumentGenerator::badarg, h, GeneratorHistoryLength]; False)
(* Better time zone validation in ST path, this is almost pointless. *)
validateDocGenArg["timezone", _String] := True
validateDocGenArg["timezone", _?NumericQ] := True
validateDocGenArg["timezone", Automatic] := True
validateDocGenArg["timezone", HoldPattern[Entity["TimeZone", _String]]] := True
validateDocGenArg["timezone", t_] := (Message[DocumentGenerator::badarg, t, TimeZone]; False)
validateDocGenArg["autoremove", True|False] := True
validateDocGenArg["autoremove", a_] := (Message[DocumentGenerator::badarg, a, AutoRemove]; False)
validateDocGenArg[__] := False

constructTemplate[f_File] := {"notebook", Import[f, "String"]}
constructTemplate[obj_CloudObject] := {"uuid", CloudObjectInformation[obj, "UUID"]}
constructTemplate[nb_Notebook] := {"notebook", ToString[nb, InputForm]}
constructTemplate[obj_TemplateObject] := {"co", ToString[obj, InputForm]}
constructTemplate[d_Delayed] := {"co", d}

constructDriver[f_File] := {"notebook", Import[f, "String"]}
constructDriver[obj_CloudObject] := {"uuid", CloudObjectInformation[obj, "UUID"]}
constructDriver[nb_Notebook] := {"notebook", ToString[nb, InputForm]}
constructDriver[assoc_Association] := {"co", ToString[assoc, InputForm]}
constructDriver[f_Function] := {"co", ToString[f, InputForm]}
constructDriver[d_Delayed] := {"co", ToString[d, InputForm]}
constructDriver[_] := {Null, "None"}

SetAttributes[constructEpilog, {HoldAll}];
constructEpilog[f_File] := {"notebook", Import[f, "String"]}
constructEpilog[obj_CloudObject] := {"uuid", CloudObjectInformation[obj, "UUID"]}
constructEpilog[nb_Notebook] := {"notebook", ToString[nb, InputForm]}
constructEpilog[d_Delayed] := {Null, ""}
constructEpilog[f_Function] := {Null, ""}
constructEpilog[_] := {Null, "None"}


generateReportJson[r:DocumentGenerator[template_, driver_, __], {name:(_String|Null), uuid:(_String|Null)}, {start_, stdSched_, end_}, o:{___?OptionQ}]:=
Module[{opts, type = "dg", strexpr, result, startTimestamp, endTimestamp, templateType, driverType, tempFormatted, driverFormatted},
    (* Format NotificationFunction *)
    If[TrueQ[KeyMemberQ[o, NotificationFunction]], 
        opts = Normal @ MapAt[verboseNotificationFunctionformat,
                Association @@ o,
                Key[NotificationFunction]
            ],
        opts = o;
    ];

    If[TrueQ[KeyMemberQ[opts, DeliveryFunction]], 
        opts = Normal @ MapAt[
                Replace[#,
                    {Null | None -> {},
                    	Automatic -> {{$CloudUserID} -> Automatic},
                    	str_String :> {{$CloudUserID} -> str},
                    	f_Function :> {{$CloudUserID} -> f},
                    	{u__String} :> {{u} -> Automatic},
                    	Rule[s_String, cond : Automatic | All] :> {Rule[{$CloudUserID}, {s, cond}]}
                    }] &,
                Association @@ opts,
                Key[DeliveryFunction]
            ]
    ];

    (* Clean up options *)
    opts = formatTaskOptions[opts, DocumentGenerator];

    {templateType, tempFormatted} = constructTemplate[template];
    {driverType, driverFormatted} = constructDriver[driver];

    strexpr = exportToJSON[{"DocumentGenerator", Sequence @@ Map[ToString[#, InputForm]&, DeleteCases[opts, Rule["RunImmediately"|"update", _]]]}, "Compact"->True];

    (* Format start and end times *)
    startTimestamp = Replace[start, {d_DateObject :> 1000*UnixTime[d]}];
    endTimestamp = Replace[end, {d_DateObject :> 1000*UnixTime[d]}];

    result = {
        "type" -> type,
        "userId" -> $CloudUserUUID,
        "uuid" -> uuid,
        "jobType"-> type,
        "visible" -> True,
        "startTimestamp" -> startTimestamp,
        "endTimestamp" -> endTimestamp,
        "timeZone" -> Lookup[opts, TimeZone],
        "count" -> If[Length[stdSched]>1, ToString[stdSched[[-1]]], "Infinity"],
        "repeatCount" -> ToString[stdSched[[-1]]],
        "interval" -> If[TrueQ[Head[stdSched[[1]]] == String], "", stdSched[[1]]],
        "notificatees" -> exportToJSON[denormalizeNotificationFunction[Lookup[opts, NotificationFunction]], "Compact" -> True],
        "cron" -> If[StringQ[stdSched[[1]]], stdSched[[1]], ""],
        "runImmediately" -> Lookup[opts, "RunImmediately", False],
        "update" -> Lookup[opts, "update", False],
        "name" -> name,
        "expression" -> strexpr,
        "templateType" -> templateType,
        "template" -> tempFormatted,
        "driverType" -> driverType,
        "driver" -> driverFormatted
    }
];


(* No driver *)
DocumentGenerator /: CloudDeploy[DocumentGenerator[t_?templateTypeQ, sched_, o:OptionsPattern[]], args___] :=
    CloudDeploy[DocumentGenerator[t, None, sched, o], args];

DocumentGenerator /: CloudDeploy[r:DocumentGenerator[t_?templateTypeQ, d_?driverTypeQ, sched_, o:OptionsPattern[]],
    co_CloudObject, oD:OptionsPattern[]] :=
    Catch[iCloudDeployDocumentGenerator[DocumentGenerator[t, d, sched, o], co, oD], $tag]

iCloudDeployDocumentGenerator[
    r:DocumentGenerator[templateRaw_?templateTypeQ, driverRaw_?driverTypeQ, sched_, o:OptionsPattern[]],
    co:CloudObject[uri_String, ___],
    oD:OptionsPattern[]
] := Module[
    {cloud, uuid, name,
    	runImmediately, cronned,
    	opts = Flatten[{o}],
    	params, rJson,
    	templateMed = templateRaw, driverMed = driverRaw,
    endpoint
    },
    {cloud, uuid, name} = getCloudUUIDAndPathForTask[co];

    {runImmediately, cronned} = Which[
        nowQ[sched],
        {True, resolveTimespec[None]},

        True,
        {False, resolveTimespec[ReleaseHold[sched]]}
    ];

    If[MatchQ[cronned, $Failed],
        Message[DocumentGenerator::sched, sched];
        Throw[$Failed, $tag]
    ];

    cronned = ReplaceAll[cronned, {None -> Null}];

    opts = DeleteDuplicatesBy[Flatten[Join[opts, Options[DocumentGenerator]]], First];
    If[And @@ # === False, Throw[$Failed, $tag]] & @ MapThread[
        validateDocGenArg[#1, #2] &,
        {
            {"options", "template", "driver", "delivery", "epilog", "historylength", "notification", "outputformat",
                "timezone", "autoremove"},
            {{FilterRules[opts, Except[Options[DocumentGenerator]]], r}, templateMed, driverMed,
                Lookup[opts, DeliveryFunction], Lookup[opts, EpilogFunction],
                Lookup[opts, GeneratorHistoryLength], Lookup[opts, NotificationFunction], Lookup[opts, GeneratorOutputType],
                Lookup[opts, TimeZone], Lookup[opts, AutoRemove]}
        }
    ];
    If[FileExistsQ[co],
        opts = Append[opts, "update" -> True];
        If[uuid == Null, uuid = getUUID[cloud, FileNameJoin[{"user-"<>$CloudUserUUID, name}]]]
    ];
    rJson = exportToJSON[generateReportJson[r, {name, uuid}, cronned, opts], "Compact" -> True];

    params = $lastJSON = ExportString[{"report" -> rJson}, "JSON", "Compact" -> True];

    endpoint = {"reports"};
    With[{mh = DocumentGenerator},
        rJson = Replace[
        execute[$CloudBase, "POST", endpoint,
        Body -> ToCharacterCode[params, "UTF-8"], 
        Type -> "application/vnd.wolfram.bundle.document-generator"], {
            {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content]),
            HTTPError[400, ___] :> (Message[mh::argu]; Throw[$Failed, $tag]),
            HTTPError[403, ___] :> (Message[DocumentGenerator::restr]; Throw[$Failed, $tag]),
            HTTPError[404, ___] :> (Message[DocumentGenerator::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag]),
            other_ :> (Message[mh::srverr]; Message[mh::crea]; Throw[$Failed, $tag])
        }];
    ];

    co
]

iCloudDeployDocumentGenerator[DocumentGenerator[args___, o:OptionsPattern[]], ___] :=
    (ArgumentCountQ[DocumentGenerator, Length[Hold[args]], 3, 3]; $Failed)

Options[iCloudRunDocumentGenerator] = {
    GeneratedDocumentBinding -> Automatic
};

iCloudRunDocumentGenerator[obj_CloudObject, mh_:RunScheduledTask, o:OptionsPattern[]] :=
    (iCloudRunDocumentGenerator[safeCloudAndUUIDFetch[obj, mh], mh, o]; obj)
iCloudRunDocumentGenerator[{cloud_String, uuid_String}, mh_:RunScheduledTask, o:OptionsPattern[]] := Module[
    {json, params},

    params = Replace[OptionValue[GeneratedDocumentBinding], {
        Automatic|Null|None -> {},
        a_Association :> Normal[a],
        x:Except[{___Rule}] :> (Message[DocumentGenerator::badarg, x, GeneratedDocumentBinding]; Throw[$Failed, $tag])
    }];

    json = Replace[execute[cloud, "POST", {"reports", uuid, "execute"}, Parameters -> params], {
        HTTPError[400, content_, ___] :> ( (* object inactive *)
            mess = ToExpression[Lookup[importFromJSON[content], "error"], InputForm, Hold];
            ReleaseHold[Replace[mess, slug_ :> Message[slug, cloudObjectFromUUID[uuid]], 1]];
            Throw[$Failed, $tag]
        ),
        HTTPError[404, ___] :> (Message[ScheduledTask::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag]),
        {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content]),
        other_ :> (Message[mh::srverr, obj]; Throw[$Failed, $tag])
    }];
    uuid
];

(* RemoveScheduledTask *)
iCloudRemoveDocumentGenerator[obj_CloudObject, mh_:RemoveScheduledTask, o:OptionsPattern[]] := (iCloudRemoveDocumentGenerator[safeCloudAndUUIDFetch[obj, mh], mh, o]; obj)
iCloudRemoveDocumentGenerator[{cloud_String, uuid_String}, mh_:RemoveScheduledTask, o:OptionsPattern[]] := Module[
    {json},
    json = Replace[execute[cloud, "DELETE", {"reports", uuid}], {
        HTTPError[404, ___] :> (Message[ScheduledTask::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag]).
        {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content]).
        other_ :> (Message[mh::srverr]; Message[ScheduledTask::norm, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
    }];
    uuid
];

(* iCloudRemoveDocumentGenerator[co_, OptionsPattern[]] := (Message[DocumentGenerator::norm]; $Failed) *)


(* DocumentGeneratorInformation *)
SetAttributes[DocumentGeneratorInformation, {ReadProtected}];

DocumentGeneratorInformation::noprop = "`1` is not a property returned by DocumentGeneratorInformation.";

(* Convert DeliveryFunction from flat list *)
gatherDeliveryFunction[n:{{__}...}] := SortBy[
    Sort[Lookup[#, "email"]] -> ToExpression[Lookup[#[[1, -1]], "format"]] & /@ GatherBy[n, Lookup["format"]],
    Last
];


(* Output is a flat list digestible by the web ui. *)
denormalizeDeliveryFunction[dfRaw_] := Module[
    {df = Replace[dfRaw, {
        Null|None -> {},
        Automatic -> {{$CloudUserID} -> Automatic},
        s_String -> {{$CloudUserID} -> s},
        {u__String} :> {{u} -> Automatic},
        Rule[{u__String}, f_] :> {{u} -> f},
        f_Function :> {{$CloudUserID} -> f},
        (* non-mail channels *)
        Rule[s_String, form_] :> {Rule[{$CloudUserID}, {s, form}]}
    }]},

    (* Naked addresses etc. *)
    df = Replace[
        df,
        {
            (addr_ -> form_) :> Flatten[{addr}] -> Replace[form, {None -> Null, s_ :> ToString[InputForm[s]]}]
        },
        {1}
    ];

    With[{pairs = DeleteDuplicates @ Flatten[(Thread[List @@ #1, List, 1] &) /@ df, 1]},
        {"email" -> First@#, "format" -> Last@#} & /@ pairs
    ]
]


$docGenNormalizationRules = {
    Rule[tag:"CreationDate"|"LastModificationDate"|"LastRunDate", 0|Null] :> Rule[tag, None],
    Rule[tag:"CreationDate"|"LastModificationDate"|"LastRunDate", t_?NumericQ] :> Rule[tag, FromUnixTime[Round[t/1000]]],
    Rule["CurrentDocument", uuid_String] :> {
        Rule["CurrentDocumentUUID", uuid],
        Rule["CurrentDocument", cloudObjectFromUUID[uuid]]
    },
    Rule[DeliveryFunction, mo_] :> Rule[DeliveryFunction, gatherDeliveryFunction[mo]],
    Rule[de:"Driver"|EpilogFunction, Null] :> Rule[de, None],
    Rule["Driver", uuid_String] :> {
        Rule["DriverUUID", uuid],
        Rule["Driver", cloudObjectFromUUID[uuid]]
    },
    Rule["DriverSrc", uuid_String] :> {
        Rule["DriverSrcUUID", uuid],
        Rule["DriverSrc", cloudObjectFromUUID[uuid]]
    },
    Rule[EpilogFunction, uuid_String] :> {
        Rule[EpilogFunction, uuid],
        Rule[EpilogFunction, cloudObjectFromUUID[uuid]]
    },
    Rule["EpilogFunctionSrc", uuid_String] :> {
        Rule["EpilogFunctionSrc", uuid],
        Rule["EpilogFunctionSrc", cloudObjectFromUUID[uuid]]
    },
    Rule["Name", Null] -> Rule["Name", None],
    Rule[GeneratorDescription, Null] -> Rule[GeneratorDescription, None],
    Rule["GeneratedDocumentHistory", uuid_String] :> {
        Rule["GeneratedDocumentHistoryUUID", uuid],
        Rule["GeneratedDocumentHistory", cloudObjectFromUUID[uuid]]
    },
    Rule[GeneratorHistoryLength, -1] -> Rule[GeneratorHistoryLength, Infinity],
    Rule["Log", uuid_String] :> {
        Rule["LogUUID", uuid],
        Rule["Log", cloudObjectFromUUID[uuid]]
    },
    Rule["CurrentOutput", uuid_String] :> {
        Rule["CurrentOutputUUID", uuid],
        Rule["CurrentOutput", cloudObjectFromUUID[uuid]]
    },
    Rule[Permissions, s_String] :> Rule[Permissions, ToExpression[s]],
    Rule["Task", expr_] :> Rule["Task", 
        formatTaskInformation[KeyMap[Lookup[$taskMetaToWLKeyMap, #, "Expression"]&, KeySelect[expr, MemberQ[$taskMetaKeys, #]&]]]],
    Rule["Template", uuid_String] :> {
        Rule["TemplateUUID", uuid],
        Rule["Template", cloudObjectFromUUID[uuid]]
    },
    Rule["TemplateSrc", uuid_String] :> {
        Rule["TemplateSrcUUID", uuid],
        Rule["TemplateSrc", cloudObjectFromUUID[uuid]]
    },
    Rule["UUID", uuid_String] :> {
        Rule["UUID", uuid],
        Rule["Directory", cloudObjectFromUUID[uuid]]
    }
};


$docGenDenormalizationRules = {
    (* Make sure the time is not an integer here, to work around bug 289879. *)
    Rule["archiveLength", Infinity|DirectedInfinity] -> Rule["archiveLength", -1],
    Rule[tag:"creationDate"|"lastModificationDate"|"lastRunDate", t:(_DateObject|_?NumericQ)] :>
        Rule[tag, ToString@Round[1000*ToUnixTime[t]]],
    Rule["outputPermissions", expr_] :> Rule["outputPermissions", ToString[InputForm[expr]]],
    Rule["recipients", df_] :> Rule["recipients", denormalizeDeliveryFunction[df]],
    Rule[lhs_, None] :> Rule[lhs, Null]
};


$docGenMetaToWLKeyMap = Association[
    "archiveId" -> "GeneratedDocumentHistory",
    "archiveLength" -> GeneratorHistoryLength,
    "archivePath" -> "GeneratedDocumentHistoryPath",
    "active" -> "Active",
    "creationDate" -> "CreationDate",
    "copyDriver" -> "CopyDriver",
    "copyEpilog" -> "CopyEpilog",
    "copyTemplate" -> "CopyTemplate",
    "currentId" -> "CurrentDocument",
    "currentPath" -> "CurrentDocumentPath",
    "driverId" -> "DriverSrcUUID",
    "driverPath" -> "DriverPath",
    "epilogId" -> "EpilogSrcUUID",
    "epilogPath" -> "EpilogPath",
    "expression" -> "Expression",
    "lastModificationDate" -> "LastModificationDate",
    "lastRunDate" -> "LastRunDate",
    "logId" -> "Log",
    "logPath" -> "LogPath",
    "operationPath" -> "OperationPath",
    "outputFormat" -> GeneratorOutputType,
    "outputId" -> "CurrentOutput",
    "outputPath" -> "CurrentOutputPath",
    "outputPermissions" -> Permissions,
    "owner" -> "Owner",
    "recipients" -> DeliveryFunction,
    "reportDescription" -> GeneratorDescription,
    "reportHistory" -> "ReportHistory",
    "reportName" -> "Name",
    "reportPath" -> "DirectoryPath",
    "schedule" -> "Task",
    "taskId" -> "TaskUUID",
    "templateId" -> "TemplateSrcUUID",
    "templatePath" -> "TemplatePath",
    "uuid" -> "UUID",
    "workingDriverId" -> "Driver",
    "workingEpilogId" -> EpilogFunction,
    "workingTemplateId" -> "Template"
];

$WLToDocGenMetaKeyMap = Association[Reverse /@ Normal[$docGenMetaToWLKeyMap]];

$presentableDocGenInfoKeys = Key /@ List[
    "CreationDate",
    "CurrentOutput",
    DeliveryFunction,
    "Directory",
    "Driver",
    "Expression",
    EpilogFunction,
    GeneratorDescription,
    "GeneratedDocumentHistory",
    GeneratorHistoryLength,
    GeneratorOutputType,
    "LastModificationDate",
    (* "LastRunDate", *)
    "Log",
    "Name",
    "Owner",
    Permissions,
    "Task",
    "Template",
    "UUID"
];


$outgoingDocGenMetaKeys = Key /@ List[
    "archiveLength",
    "copyDriver",
    "copyEpilog",
    "copyTemplate",
    "deliveryFunction",
    "driverId",
    "epilogId",
    "outputFormat",
    "outputPermissions",
    "recipients",
    "reportDescription",
    "reportName",
    "templateId",
    "uuid"
];


docGenMetaToWL[raw_List] := Module[
    {med, well},
    (* Replace json keys with WL symbols/strings *)
    med = DeleteCases[
        Replace[raw, Rule[lhs_, rhs_] :> Rule[$docGenMetaToWLKeyMap[lhs], rhs], 1],
        Rule[Missing[__], _]
    ];

    well = Association @@ Flatten[Replace[med, $docGenNormalizationRules, 1], 1]
]


WLToDocGenMeta[System`DocumentGeneratorInformationData[a_Association]] := WLToDocGenMeta[Normal[a]];
WLToDocGenMeta[a_Association] := WLToDocGenMeta[Normal[a]];
WLToDocGenMeta[raw:OptionsPattern[]] := Module[
    {med, well},

    (* Replace WL symbols and strings with json keys *)
    med = DeleteDuplicates[DeleteCases[
        Replace[raw, Rule[lhs_, rhs_] :> Rule[$WLToDocGenMetaKeyMap[lhs], rhs], 1],
        Rule[Missing[__], _]
    ], First[#1] === First[#2] &];

    well = Replace[med, $docGenDenormalizationRules, 1];
    well
]


unpresentifyDocGenMeta[raw_] := With[{med = WLToDocGenMeta[raw]},
    DeleteCases[
        Normal[Apply[Association, med][[$outgoingDocGenMetaKeys]]],
        Rule[_, Missing[__]]
    ]
];


presentifyDocGenMeta[med_Association] := Module[
    {well = med},
    If[KeyExistsQ[well, "Task"],
        well["Task"] = System`ScheduledTaskInformationData[(*presentifyTaskMeta[*)well["Task"](*]*)];
    ];
    well[[$presentableDocGenInfoKeys]]
]

DocumentGeneratorInformation[obj_CloudObject] := Replace[
    Catch[iCloudDocumentGeneratorInformation[obj], $tag], {
    assoc_Association :> Dataset[(*presentifyDocGenMeta @ *)assoc]
}]

DocumentGeneratorInformation[obj_CloudObject, "Task"] := 
Module[{uuid, cloud, taskID},
	{cloud, uuid} = safeCloudAndUUIDFetch[obj, DocumentGeneratorInformation];
	taskID = Lookup[Lookup[Options[cloudObjectFromUUID[uuid], MetaInformation->"__Task"],MetaInformation],"__Task"];
	
	ScheduledTaskInformation[cloudObjectFromUUID[taskID]]
]

DocumentGeneratorInformation[obj_CloudObject, property_] := Replace[
    Catch[iCloudDocumentGeneratorInformation[obj], $tag], {
    assoc_Association :> If[KeyExistsQ[assoc, property],
        (*presentifyDocGenMeta[assoc]*)assoc[property],
        (* else *)
        Message[DocumentGeneratorInformation::noprop, property];
        $Failed
    ]
}]

iCloudDocumentGeneratorInformation[obj_CloudObject, mh_:DocumentGeneratorInformation] :=
    iCloudDocumentGeneratorInformation[safeCloudAndUUIDFetch[obj, mh], mh]
iCloudDocumentGeneratorInformation[{cloud_String, uuid_String}, mh_:DocumentGeneratorInformation] := Module[
    {taskID, assoc, expr, opts, name, current, report = cloudObjectFromUUID[uuid]},

    taskID = Lookup[Lookup[Options[report, MetaInformation->"__Task"], MetaInformation], "__Task"];
    assoc = Normal[ScheduledTaskInformation[cloudObjectFromUUID[taskID]]];
    expr = Lookup[assoc, "Expression"];
    opts = List@@expr[[4;;]];
    name = CloudObjectInformation[report, "Name"];
    current = FileNameJoin[{report, "current.nb"}];
    current = Quiet[Catch[CloudObject[current, CloudObjectNameFormat->"UUID"], None]];
    
    <|
    	       "Name" -> name,
    	       "Task" -> Dataset[assoc],
    	       "Template" -> formatDGElement[expr[[1]]],
    	       "Driver" -> formatDGElement[expr[[2]]],
    	       DeliveryFunction -> Lookup[opts, DeliveryFunction],
    	       EpilogFunction -> Lookup[opts, EpilogFunction],
    	       GeneratorDescription -> Lookup[opts, GeneratorDescription],
    	       GeneratorHistoryLength -> Lookup[opts, GeneratorHistoryLength],
    	       GeneratorOutputType -> Lookup[opts, GeneratorOutputType],
    	       Permissions -> Lookup[opts, Permissions],
    	       "Owner" -> Lookup[assoc, "Owner"],
    	       "CurrentOutput" -> If[FailureQ[current], None, current]
    	   |>
];

DocumentGeneratorInformation[failureObj_Failure, arg___] := failureObj

formatDGElement[uuid_?UUIDQ] := cloudObjectFromUUID[uuid]
formatDGElement[e_] := e

SetAttributes[DocumentGenerators, {ReadProtected}];
DocumentGenerators[o:OptionsPattern[]] := Module[
    {res = Catch[iCloudDocumentGenerators[o], $tag]},
    res
];

iCloudDocumentGenerators[o:OptionsPattern[]] := Module[
    {raw, json, msghd = DocumentGenerators},

    With[{mh = msghd},
        json = Replace[execute[$CloudBase, "GET", {"reports"}], {
            {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content])
            , other_ :> (Message[mh::srverr]; Throw[$Failed, $tag])
        }];

        Check[raw = Lookup[importFromJSON[json], "reports", {Missing[]}],
            Message[mh::srverr];
            Throw[$Failed, $tag]
        ];
    ];

    If[Length[raw] == 0, Return[{}]];

    cloudObjectFromUUID /@ raw
]


Protect[DocumentGenerator, DocumentGenerators, DocumentGeneratorInformation, CloudObject];

End[]

EndPackage[]
