(* ::Package:: *)

(* Mathematica package *)
BeginPackage["CloudObject`"];

System`ScheduledTask;
System`EvaluateScheduledTask;
System`NextScheduledTaskTime;
System`ScheduledTaskActiveQ;
System`ScheduledTaskInformation::usage = "ScheduledTaskInformation[CloudObject] returns information about a task.
ScheduledTaskInformation[CloudObject, property] returns the value of the specified property.";
System`ScheduledTaskInformationData;
System`ContinuousTask;
System`AbortScheduledTask;
System`AutoRefreshed;
System`TaskAbort;
Tasks`TaskEvaluate;
System`TaskExecute;

(* Option symbols *)
System`NotificationFunction;
System`IncludeGeneratorTasks;
System`RestartInterval;

Begin["`Private`"];

Unprotect[ScheduledTask, ContinuousTask, CloudObject];
SetAttributes[ScheduledTask, {HoldAll, ReadProtected}];

(* MySQL-specific; must match CalendarUtilities *)
$neverRunDate = 253402300799000;
$taskMimeTypes = {"application/vnd.wolfram.expression.task"};
$restrictedScheduler := CloudEvaluate[CloudSystem`Scheduling`$restricted];

Options[ScheduledTask] = Options[ContinuousTask] = Sort@{
    NotificationFunction -> Automatic,
    TimeZone -> Automatic,
    AutoRemove -> False
};

ScheduledTask /: CloudDeploy[task_ScheduledTask, co_CloudObject, OptionsPattern[]] := With[
    {res = Catch[iCloudDeployScheduledTask[task, co], $tag]}, 
    res
]

SetAttributes[ContinuousTask, {HoldAll, ReadProtected}];
Options[ContinuousTask] = {
    NotificationFunction -> Automatic,
    RestartInterval -> Automatic,
    TimeZone -> Automatic
};

(* Continuous tasks must be pausable, so don't use None for timespec. *)
ContinuousTask /: CloudDeploy[ContinuousTask[expr_, o:OptionsPattern[]], co_CloudObject, oCD:OptionsPattern[]] := 
    continuousTaskStaging[ContinuousTask[expr, 365*24*3600, o], co, oCD]

ContinuousTask /: CloudDeploy[ct:ContinuousTask[expr_, end:Except[HoldPattern[_Quantity]], o:OptionsPattern[]], co_CloudObject, oCD:OptionsPattern[]] := 
    continuousTaskStaging[ContinuousTask[expr, {Now, 365*24*3600, DateObject[end]}, o], co, oCD]

ContinuousTask /: CloudDeploy[ct:ContinuousTask[expr_, tspan:HoldPattern[_Quantity], o:OptionsPattern[]], co_CloudObject, oCD:OptionsPattern[]] := 
    continuousTaskStaging[ContinuousTask[expr, {Now, 365*24*3600, Now + tspan}, o], co, oCD]

continuousTaskStaging[ContinuousTask[expr_, tspec_, o:OptionsPattern[]], co_CloudObject, oCD:OptionsPattern[]] := With[
    {options = Join[Flatten[{o}], {"Continuous" -> True}, Options[ContinuousTask]]}, 
    Catch[iCloudDeployScheduledTask[ScheduledTask[expr, tspec, options], co, oCD], $tag]
];

iCloudDeployScheduledTask[st:ScheduledTask[expr_, sched_, o:OptionsPattern[]], obj:CloudObject[uri_String, ___], i0:OptionsPattern[]] :=
Module[{cloud, uuid, name, continuous = TrueQ[Lookup[Flatten[{o}], "Continuous", False]],
    runImmediately = False, cronned, params, opts = Flatten[{o}], endpoint, taskJson, rJson},

    {cloud, uuid, name} = getCloudUUIDAndPathForTask[obj];

    {runImmediately, cronned} = Which[
        continuous,
        {True, resolveTimespec[ReleaseHold[sched]]},
        
        nowQ[sched],
        {True, resolveTimespec[None]},
        
        True,
        {False, resolveTimespec[ReleaseHold[sched]]}
    ];
    
    If[FailureQ[cronned],
        Message[ScheduledTask::sched, sched];
        Throw[$Failed, $tag]
    ];

    cronned = ReplaceAll[cronned, {None -> Null}];

    taskJson = generateTaskJson[st, {name, uuid}, cronned, Join[opts, {"RunImmediately" -> runImmediately}]];
    If[taskJson === $Failed,
        Throw[$Failed, $tag]
    ];
    params = {"task" -> taskJson};
    (* Print@params; *)

    endpoint = If[scheduledTaskQ[obj], {"tasks", Last@safeCloudAndUUIDFetch[obj, ScheduledTask]}, {"tasks"}];
    With[{mh = ScheduledTask},
        rJson = Replace[
        	execute[cloud, "POST", endpoint, 
            Body -> ToCharacterCode[exportToJSON[params], "UTF-8"]
        	], {
            {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content]),
            HTTPError[400, content_, ___] :> (With[{json = importFromJSON[content]},
                handleTaskErrorMessage[Lookup[json, "error"], mh];
                Throw[$Failed, $tag]];),
            HTTPError[403, content_, ___] :> (
                    ToExpression[Lookup[importFromJSON[content], "error", "ScheduledTask::restr"], InputForm, Message];
                    Throw[$Failed, $tag]
                )
            , other_ :> (Message[mh::srverr]; Message[mh::crea]; Throw[$Failed, $tag])
        }];
    ];

  obj
]

getCloudUUIDAndPathForTask[obj_CloudObject] :=
Module[{cloud, uuid, name},
	{cloud, uuid, name} = Replace[getCloudAndUUIDOrPath[obj], None -> Null, {1}];
    name = Replace[name, {
        n : {"user-" ~~ $CloudUserUUID, __} :> FileNameJoin[Rest[n], OperatingSystem -> "Unix"],
        n : {$UserURLBase, __} :> FileNameJoin[Rest[n], OperatingSystem -> "Unix"],
        n_List :> FileNameJoin[n, OperatingSystem -> "Unix"]
    }];
    {cloud, uuid, name}
]

taskOptFormat[o___, head_] := StringJoin[Sequence @@ Riffle[ToString[#, InputForm] & /@ DeleteDuplicatesBy[Flatten[Join[{o}, Options[head]], 1],First], ", "]]

replaceDefaults[AutoRemove -> x_] := Rule[AutoRemove, Replace[x, {True -> True, _ :> False}]]
replaceDefaults[NotificationFunction -> notif_] := Rule[NotificationFunction, Replace[notif, {Automatic -> {{$CloudUserID} -> Automatic}}]]
replaceDefaults[Rule[TimeZone, tz_]] := Rule[TimeZone, Replace[tz, {Automatic -> $TimeZone}]]
replaceDefaults[rule_] := rule

resolveTaskExpr[st:ScheduledTask[expr_, sched_, opts___]] := Replace[Unevaluated[expr], {
	(* Local file, cloud; client supplies path, validated on server *)
    f_File :> {"file", StringTemplate["ScheduledTask[File[`expr`], `sched`, `opts`]"][
        <|"expr" -> ToString[AbsoluteFileName[First[f]], InputForm],
        "sched" -> ToString[sched, InputForm],
        "opts" -> taskOptFormat[opts, ScheduledTask]|>]} /; TrueQ[$CloudEvaluation],
    (* Local file, desktop; client supplies uuid *)
    f_File :> With[{o = Check[CopyFile[f, CloudObject[], "MIMEType" -> "application/vnd.wolfram.wl"], Throw[$Failed, $tag]]},
    	Message[ScheduledTask::copied, First[f], o];
    	{"file", StringTemplate["ScheduledTask[`expr`, `sched`, `opts`]"][
        <|"expr" -> ToString[Last@safeCloudAndUUIDFetch[o], InputForm],
        "sched" -> ToString[sched, InputForm],
        "opts" -> taskOptFormat[opts, ScheduledTask]|>]}
    ] /; Not[TrueQ[$CloudEvaluation]],
    (* Existing cloud object; client supplies uuid *)
    o_CloudObject :> {"file", StringTemplate["ScheduledTask[`expr`, `sched`, `opts`]"][
        <|"expr" -> ToString[Last@safeCloudAndUUIDFetch[o], InputForm],
        "sched" -> ToString[sched, InputForm],
        "opts" -> taskOptFormat[opts, ScheduledTask]|>]},
    (* Arbitrary expression *)
	e_ :> {"co", StringTemplate["ScheduledTask[`expr`, `sched`, `opts`]"][
		<|"expr" -> exprToStringIncludingDefinitions[Unevaluated[expr]],
		"sched" -> ToString[sched, InputForm],
		"opts" -> taskOptFormat[opts, ScheduledTask]|>]}
}];

resolveTaskExpr[con:ContinuousTask[expr_, end_, opts__]] := {"cont", StringTemplate["ContinuousTask[`expr`, `end`, `opts`]"][
  <|"expr" -> exprToStringIncludingDefinitions[Unevaluated[expr]],
    "end" -> ToString[end, InputForm],
    "opts" -> taskOptFormat[opts, ContinuousTask]|>
]};

resolveTaskExpr[ar:AutoRefreshed[expr_, sched_, fmt_,  opts___]] := {"ar", StringTemplate["AutoRefreshed[`expr`, `sched`, `format`, `opts`]"][
  <|"expr" -> exprToStringIncludingDefinitions[Unevaluated[expr]],
  "sched" -> ToString[sched, InputForm],
  "format" -> ToString[fmt, InputForm],
  "opts" -> taskOptFormat[opts, AutoRefreshed]|>
]};

resolveTaskExpr[dg:DocumentGenerator[template_, driver_, sched_, opts___]] := {"dg", StringTemplate["DocumentGenerator[`template`, `driver`, `sched`, `opts`]"][
  <|"template" -> ToString[template, InputForm],
  "driver" -> ToString[driver, InputForm],
  "sched" -> ToString[sched, InputForm],
  "opts" -> taskOptFormat[opts, DocumentGenerator]|>
]};

exprDefinitionsToString[expr_] :=
    Module[{defs, defsString},
        (* This fn is used by the package itself, so make sure the package context
         * is not excluded. *)
        defs = With[
            {excl = Join[
                OptionValue[Language`ExtendedFullDefinition, "ExcludedContexts"],
                {"MailReceiver", "CloudSystem", "Forms", "Templating", "Interpreter", "CloudObject"}
            ]},
            Language`ExtendedFullDefinition[expr, "ExcludedContexts" -> Complement[excl, $IncludedContexts]]
        ];
        defsString = If[defs =!= Language`DefinitionList[],
            neutralContextBlock[With[{d = defs},
                (* Language`ExtendedFullDefinition[] can be used as the LHS of an assignment to restore
                 * all definitions. *)
                ToString[Unevaluated[Language`ExtendedFullDefinition[] = d], InputForm,
                    CharacterEncoding -> "PrintableASCII"]
            ]] <> ";\n\n",
        (* else *)
            ""
        ];
        StringTrim[defsString] <> "\n\n"
    ]

(* This experimental and undocumented ScheduledTask[File[...]] form is deprecated and may be removed in the future. *)
generateTaskJson[ScheduledTask[f_File, rest__], more__] :=
With[{package = CloudExport[Import[f, "String"], "String"]},
	generateTaskJson[ScheduledTask[CloudGet[package], rest], more]
]

(* This experimental and undocumented ScheduledTask[CloudObject[...]] form is deprecated and may be removed in the future. *)
generateTaskJson[ScheduledTask[co_CloudObject, rest__], more__] := generateTaskJson[ScheduledTask[CloudGet[co], rest], more]

generateTaskJson[st:ScheduledTask[expr_, __], {name:(_String|Null), uuid:(_String|Null)}, {start_, stdSched_, end_}, 
    o:{___?OptionQ}] :=
Module[{opts, cont, type = "co", strexpr, result, startTimestamp, endTimestamp},

    cont = Lookup[o, "Continuous", False];

    If[TrueQ[KeyMemberQ[o, NotificationFunction]], 
        opts = Normal @ MapAt[verboseNotificationFunctionformat,
                Association @@ o,
                Key[NotificationFunction]
            ],
        (* else *)
        opts = o;
    ];

    opts = formatTaskOptions[opts, ScheduledTask];
    
    strexpr = exportToJSON[{Replace[exprDefinitionsToString[Unevaluated[expr]], "\n\n" -> ""],
        If[cont, "ContinuousTask", "ScheduledTask"],
        neutralContextBlock[ToString[Unevaluated[expr], InputForm, CharacterEncoding -> "PrintableASCII"]],
        Sequence @@ Map[ToString[#, InputForm]&, DeleteCases[opts, Rule["RunImmediately", _]]]}, "Compact" -> True];

    startTimestamp = Replace[start, {d_DateObject :> 1000*UnixTime[d]}];
    endTimestamp = Replace[end, {d_DateObject :> 1000*UnixTime[d]}];

    result = {
        "type" -> If[cont, "cont", type],
        "userId" -> $CloudUserUUID,
        "uuid" -> uuid,
        "jobType" -> If[cont, "cont", type],
        "continuous" -> cont,
        "visible" -> Lookup[opts, "Visible", True],
        "startTimestamp" -> startTimestamp,
        "endTimestamp" -> endTimestamp,
        "timeZone" -> Lookup[opts, TimeZone],
        "count" -> If[Length[stdSched]>1, ToString[stdSched[[-1]]], "Infinity"],
        "repeatCount" -> ToString[stdSched[[-1]]],
        "interval" -> If[TrueQ[Head[stdSched[[1]]] == String], "", stdSched[[1]]],
        "notificatees" -> exportToJSON[denormalizeNotificationFunction[Lookup[opts, NotificationFunction]], "Compact" -> True],
        "cron" -> If[StringQ[stdSched[[1]]], stdSched[[1]], ""],
        "runImmediately" -> Lookup[opts, "RunImmediately", False],
        "name" -> name,
        "expression" -> strexpr
    }
];

generateTaskJson[ar:AutoRefreshed[expr_, _, fmt_, ___], {name:(_String|Null), uuid:(_String|Null)}, {start_, stdSched_, end_},
  o:{___?OptionQ}] :=
    Module[{opts, cont, type, strexpr, result,startTimestamp, endTimestamp},

      cont = Lookup[o, "Continuous", False];

      If[TrueQ[KeyMemberQ[o, NotificationFunction]],
        opts = Normal @ MapAt[verboseNotificationFunctionformat,
          Association @@ o,
          Key[NotificationFunction]
        ],
        (* else *)
        opts = o;
      ];

      opts = formatTaskOptions[opts, AutoRefreshed];

      {type, strexpr} = {"ar", StringTrim[exportToJSON[{exprDefinitionsToString[Unevaluated[expr]], "AutoRefreshed",
        neutralContextBlock[ToString[Unevaluated[expr], InputForm, CharacterEncoding -> "PrintableASCII"]],
        StringJoin["\"",fmt,"\""], Sequence @@ Map[ToString[#, InputForm]&, DeleteCases[opts, Rule["RunImmediately"|"Visible", _]]]}, "Compact"->True]]};

      startTimestamp = Replace[start, {d_DateObject :> 1000*UnixTime[d]}];
      endTimestamp = Replace[end, {d_DateObject :> 1000*UnixTime[d]}];

      result = {
        "type" -> type,
        "userId" -> $CloudUserUUID,
        "uuid" -> uuid,
        "jobType" -> type,
        "continuous" -> cont,
        "visible" -> Lookup[opts, "Visible", True],
        "startTimestamp" -> startTimestamp,
        "endTimestamp" -> endTimestamp,
        "timeZone" -> Lookup[opts, TimeZone],
        "count" -> If[Length[stdSched]>1, ToString[stdSched[[-1]]], "Infinity"],
        "repeatCount" -> ToString[stdSched[[-1]]],
        "interval" -> If[TrueQ[Head[stdSched[[1]]] == String], "", stdSched[[1]]],
        "notificatees" -> exportToJSON[denormalizeNotificationFunction[Lookup[opts, NotificationFunction]], "Compact" -> True],
        "cron" -> If[StringQ[stdSched[[1]]], stdSched[[1]], ""],
        "runImmediately" -> False,
        "name" -> name,
        "expression" -> strexpr
      }
    ];
  
verboseNotificationFunctionformat[nf_] :=
Replace[nf, {
    Null | None -> {},
    All -> {{$CloudUserID} -> All},
    Automatic -> {{$CloudUserID} -> Automatic},
    f_Function :> {{$CloudUserID} -> f},
    {u__String} :> {{u} -> Automatic},
    (*non-mail channels*)
    Rule[s_String, cond : Automatic | All] :> {Rule[{$CloudUserID}, {s, cond}]}
}];

formatTaskOptions[opts_, head_] :=
Replace[DeleteDuplicatesBy[Join[opts, Options[head]], First], {
    Rule[TimeZone, Automatic] :> Rule[TimeZone, $TimeZone],
    Rule[TimeZone, id_String] :> With[{tz = Entity["TimeZone", id]},
        Rule[TimeZone, QuantityMagnitude[tz["OffsetFromUTC"], "Hours"]]
    ],
    Rule[TimeZone, tz:Entity["TimeZone", _String]] :> Rule[TimeZone, QuantityMagnitude[tz["OffsetFromUTC"], "Hours"]],
    Rule[TimeZone, tz_?NumberQ] :> Rule[TimeZone, tz],
    Rule[TimeZone, _] :> Rule[TimeZone, $TimeZone]
}, 1];

iCloudDeployScheduledTask[ScheduledTask[args___], ___] := 
    (ArgumentCountQ[ScheduledTask, Length[DeleteCases[{args}, _Rule|_RuleDelayed, Infinity]], 2, 2]; $Failed)
iCloudDeployScheduledTask[___] := $Failed


toSchedule[n_String] := Module[{cron},
    cron = StringSplit[n];
    If[Length@cron < 3 || Length@cron > 7, Return[$Failed]];
    cron = Replace[
        cron,
        {
            (*{s_, m_, h_, dom_, m_, dow_, y_}:>{s, m, h, dom, m, dow, y},
            {s_, m_, h_, dom_, m_, dow_}:>{s, m, h, dom, m, dow, "*"},
            {h_, dom_, m_, dow_, y_}:>{"*", "*", h, dom, m, dow, y},
            {h_, dom_, m_, dow_}:>{"*", "*", h, dom, m, dow, "*"}*)
            
            
            {s_, m_, h_, dom_, mo_, dow_, y_}:>{s, m, h, dom, mo, dow, y}, (* quartz expression *)
            {m_, h_, dom_, mo_, dow_, y_}:>{"*", m, h, dom, mo, dow, y}, (* classic cron with optional year *)
            {m_, h_, dom_, mo_, dow_}:>{"*", m, h, dom, mo, dow, "*"}, (* classic cron *)
            {h_, dom_, mo_, dow_}:>{"*", "*", h, dom, mo, dow, "*"}
        
        
        }
    ];
    StringJoin[Riffle[ToUpperCase@cron, " "]]
]

toSchedule[___] := $Failed

current[spec_] := DateString[DateList[], spec];
(* We can remove this and instead use dowToCron *)
currentDOW[] := With[{date = DateList[]}, Which[
    DayMatchQ[date, Sunday], "1",
    DayMatchQ[date, Monday], "2",
    DayMatchQ[date, Tuesday], "3",
    DayMatchQ[date, Wednesday], "4",
    DayMatchQ[date, Thursday], "5",
    DayMatchQ[date, Friday], "6",
    DayMatchQ[date, Saturday], "7",
    True, "*"
]];

(* this really needs to get fixed... the first section of each cron expression*)
$TimeSpecs = {
   "Hourly" :> StringJoin["* ", current[{"MinuteShort"}]," * * * ? *"],
   "Daily" :> StringJoin["* ", current[{"MinuteShort", " ", "HourShort"}], " * * ? *"],
   "Weekly" :> StringJoin["* ", current[{"MinuteShort", " ", "HourShort"}], " ? * ", currentDOW[], " *"],
   "Monthly" :> StringJoin["* ", current[{"MinuteShort"," ", "HourShort", " ", "DayShort"}], " * ? *"],
   "Yearly" :> StringJoin["* ", current[{"MinuteShort", " ", "HourShort", " ", "DayShort", " ", "MonthShort"}], " ? *"]
};

$AvailableTimeSpecs = First /@ $TimeSpecs;

resolveTimespec[expr_] := With[{cronned = timespec2Cron[expr]},
	If[MemberQ[cronned, $Failed, Infinity], $Failed, cronned]
];
(* Dispatch *)
timespec2Cron[ts:{_DateObject, _DateObject|_Integer}] := (Message[ScheduledTask::ambig, ts]; $Failed);

timespec2Cron[spec_] := timespec2Cron[{Null, spec, Null}];
timespec2Cron[{start:_DateObject|None|Null, spec_}] := timespec2Cron[{start, spec, Null}];
timespec2Cron[{spec_, end:_DateObject|None|Null}] := timespec2Cron[{Null, spec, end}];
timespec2Cron[{start:_DateObject|None|Null, spec:Except[_List], end:_DateObject|None|Null}] := {start, {handleSpec[spec], Infinity}, end};
timespec2Cron[{start:_DateObject|None|Null, {spec_}, end:_DateObject|None|Null}] := {start, {handleSpec[spec], 1}, end};
timespec2Cron[{start:_DateObject|None|Null, {spec_, rc:_Integer?Positive|_DirectedInfinity}, end:_DateObject|None|Null}] := {start, {handleSpec[spec], rc}, end};
timespec2Cron[__] := $Failed;

(* CRON Output *)
handleSpec[spec_String] /; MemberQ[$AvailableTimeSpecs, spec] := spec /. $TimeSpecs;
handleSpec[spec_DateObject] := With[{cronned = DateObjectToCronSpecification[spec]}, 
    handleSpec[cronned]
]
handleSpec[spec_String] := With[{cronned = toSchedule[spec]}, 
    If[StringQ[cronned], 
        If[$restrictedScheduler && !StringMatchQ[cronned, "* * "~~__],
            Message[ScheduledTask::rstsch];
            cronned,
            cronned
        ],
        $Failed
    ]
];

(* Interval Output *)
handleSpec[HoldPattern[q_Quantity]] := Module[{res},
    If[CompatibleUnitQ[q, "Seconds"], 
        res = QuantityMagnitude[UnitConvert[q, "Seconds"]];
        If[$restrictedScheduler && res < 3600,
            Message[ScheduledTask::rstsch];
            3600,
            res
        ],
        $Failed
    ]
];
handleSpec[n_Integer?Positive] := If[$restrictedScheduler && n < 3600, Message[ScheduledTask::rstsch]; 3600, n];

(* None/dummy *)
handleSpec[None|Null] := Null;

(* fallthrough *)
handleSpec[__] := $Failed;


(* Now? *)
SetAttributes[nowQ, HoldAll];
nowQ[Now] := True;
nowQ[{Now}] := True;
nowQ[{Now, count_}] := MatchQ[ReleaseHold[count], _Integer|Infinity|DirectedInfinity];
nowQ[{start_, Now, end_}] := MatchQ[ReleaseHold[start], _DateObject] && MatchQ[ReleaseHold[end], _DateObject];
nowQ[{start_DateObject, Now}] := MatchQ[ReleaseHold[start], _DateObject];
nowQ[{Now, end_DateObject}] := MatchQ[ReleaseHold[end], _DateObject];
nowQ[___] := False;

handleTaskErrorMessage[error_String, mh_] := If[
    StringMatchQ[error, "ContinuousTask" | "ScheduledTask" ~~ "::" ~~ WordCharacter..],
    ToExpression[error, InputForm, Message],
    (* else *)
    Message[mh::argu]
];

handleTaskErrorMessage[_, mh_] := Message[mh::argu]

(*
    StopScheduledTask (pre-11.2)
*)

CloudObject /: StopScheduledTask[co_CloudObject, OptionsPattern[]] /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := Module[
    {task = Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation, {}], "__Task", {}]},
    If[StringQ[task],
        Catch[iCloudStopScheduledTask[{$CloudBase, task}], $tag]; co,
        Message[ScheduledTask::tasknf, co]; $Failed
    ]
];

CloudObject /: StopScheduledTask[task_CloudObject, OptionsPattern[]] := With[
    {res = Catch[iCloudStopScheduledTask[task], $tag]},
    res
];

iCloudStopScheduledTask[obj_CloudObject, mh_:StopScheduledTask] := (iCloudStopScheduledTask[safeCloudAndUUIDFetch[obj, mh], mh]; obj);
iCloudStopScheduledTask[{cloud_String, uuid_String}, mh_:StopScheduledTask] := Module[
    {json},
    json = Replace[execute[cloud, "POST", {"tasks", uuid, "pause"}], {
        HTTPError[404, ___] :> (Message[ScheduledTask::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
        , {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content])
        , other_ :> (Message[mh::srverr, obj]; Message[ScheduledTask::nostop, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
    }];
    uuid
];

(*iCloudStopScheduledTask[st_,OptionsPattern[]] := (Message[ScheduledTask::nostop,st];$Failed)*)

(*
    11.2+. The same code with replaced StopScheduledTask --> TaskSuspend
*)

CloudObject /: TaskSuspend[co_CloudObject, OptionsPattern[]] /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := Module[
    {task = Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation, {}], "__Task", {}]},
    If[StringQ[task],
        Catch[iCloudStopScheduledTask[{$CloudBase, task}], $tag]; co,
        Message[ScheduledTask::tasknf, co]; $Failed
    ]
];

CloudObject /: TaskSuspend[task_CloudObject, OptionsPattern[]] := With[
    {res = Catch[iCloudStopScheduledTask[task, TaskSuspend], $tag]},
    res
]


(*
    StartScheduledTask (pre-11.2)
*)

CloudObject /: StartScheduledTask[co_CloudObject, o:OptionsPattern[]]  /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := 
Module[{task = Lookup[Lookup[Options[co, MetaInformation->"__Task"], MetaInformation], "__Task"]},
    Catch[iCloudStartScheduledTask[{$CloudBase, task}], $tag];
    co
];

CloudObject /: StartScheduledTask[task_CloudObject, OptionsPattern[]] := With[
    {res = Catch[iCloudStartScheduledTask[task], $tag]}, res
];

iCloudStartScheduledTask[obj_CloudObject, mh_:StartScheduledTask] := (iCloudStartScheduledTask[safeCloudAndUUIDFetch[obj, mh], mh]; obj);
iCloudStartScheduledTask[{cloud_String, uuid_String}, mh_:StartScheduledTask] := Module[
    {json},
    json = Replace[execute[cloud, "POST", {"tasks", uuid, "resume"}], {
        HTTPError[404, ___] :> (Message[ScheduledTask::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
        , {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content])
        , other_ :> (Message[mh::srverr, cloudObjectFromUUID[uuid]]; Message[ScheduledTask::nostart]; Throw[$Failed, $tag])
    }];
    uuid
];

(*iCloudResumeScheduledTask[st_, OptionsPattern[]] := handleSchedulingResponse[$Failed]*)

(*
    11.2+. The same code with replaced StartScheduledTask --> TaskResume
*)

CloudObject /: TaskResume[co_CloudObject, o:OptionsPattern[]]  /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := 
Module[{task = Lookup[Lookup[Options[co, MetaInformation->"__Task"], MetaInformation], "__Task"]},
    Catch[iCloudStartScheduledTask[{$CloudBase, task}], $tag];
    co
];

CloudObject /: TaskResume[task_CloudObject, OptionsPattern[]] := With[
    {res = Catch[iCloudStartScheduledTask[task, TaskResume], $tag]}, res
]


(*
 * Equivalent to "Run now" in web interface.
 *)
CloudObject /: RunScheduledTask[co_CloudObject, o:OptionsPattern[]] /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := Module[
    {task = Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation, {}], "__Task", {}]},
    If[StringQ[task],
        Catch[iCloudRunScheduledTask[{$CloudBase, task}, RunScheduledTask], $tag]; co,
        Message[ScheduledTask::tasknf, co]; $Failed
    ]
];

CloudObject /: RunScheduledTask[co_CloudObject, OptionsPattern[]] := With[
    {res = Catch[iCloudRunScheduledTask[co], $tag]},
    res
];

iCloudRunScheduledTask[obj_CloudObject, mh_:RunScheduledTask] := (iCloudRunScheduledTask[safeCloudAndUUIDFetch[obj, mh], mh]; obj);
iCloudRunScheduledTask[{cloud_String, uuid_String}, mh_:RunScheduledTask] := Module[
    {json, mess},
    json = Replace[execute[cloud, "POST", {"tasks", uuid, "execute"}], {
        HTTPError[400, content_, ___] :> ( (* object inactive *)
            mess = ToExpression[Lookup[importFromJSON[content], "error"], InputForm, Hold];
            ReleaseHold[Replace[mess, slug_ :> Message[slug, cloudObjectFromUUID[uuid]], 1]];
            Throw[$Failed, $tag]
        )
        , HTTPError[404, ___] :> (Message[ScheduledTask::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
        , {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content])
        , other_ :> (Message[mh::srverr, obj]; Throw[$Failed, $tag])
    }];
    uuid
];

(*iCloudRunScheduledTask[st_,OptionsPattern[]] := handleSchedulingResponse[$Failed]*)


(*
    11.2+. The same code with replaced RunScheduledTask --> TaskExecute
*)

CloudObject /: TaskExecute[co_CloudObject, o:OptionsPattern[]] /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := Module[
    {task = Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation, {}], "__Task", {}]},
    If[StringQ[task],
        Catch[iCloudRunScheduledTask[{$CloudBase, task}, TaskExecute], $tag]; co,
        Message[ScheduledTask::tasknf, co]; $Failed
    ]
];

CloudObject /: TaskExecute[co_CloudObject, OptionsPattern[]] := With[
    {res = Catch[iCloudRunScheduledTask[co, TaskExecute], $tag]},
    res
]


(*
    RemoveScheduledTask
*)

CloudObject /: RemoveScheduledTask[co_CloudObject, o:OptionsPattern[]] /; Quiet[documentGeneratorQ[co]] := Module[
    {res = Catch[iCloudRemoveDocumentGenerator[co, RemoveScheduledTask, o], $tag]},
    res
];

$autoRefreshedMimeTypes = {"application/vnd.wolfram.bundle.autorefreshed"};
(* Slow! *)
autoRefreshedQ[co_CloudObject] := 
    With[{mime = Quiet[Check[CloudObjectInformation[co, "MIMEType"], $Failed]]},
    	MemberQ[$autoRefreshedMimeTypes, mime]
    ];
autoRefreshedQ[_] := False;

CloudObject /: RemoveScheduledTask[ar_CloudObject, o:OptionsPattern[]] /; Quiet[autoRefreshedQ[ar]] := With[
    {res = Catch[iCloudRemoveAutoRefreshed[ar, RemoveScheduledTask, o], $tag]},
    res
];

iCloudRemoveAutoRefreshed[obj_CloudObject, mh_:RemoveScheduledTask, o:OptionsPattern[]] := (iCloudRemoveAutoRefreshed[safeCloudAndUUIDFetch[obj, mh], mh, o]; obj);
iCloudRemoveAutoRefreshed[{cloud_String, uuid_String}, mh_:RemoveScheduledTask, o:OptionsPattern[]] := Module[
    {json},
    json = Replace[execute[cloud, "DELETE", {"tasks", uuid, "autorefreshed"}], {
        HTTPError[404, ___] :> (Message[ScheduledTask::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag]),
        {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content]),
        other_ :> (Message[mh::srverr]; Message[ScheduledTask::norm, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
    }];
    uuid
];

scheduledTaskQ[co_CloudObject] := 
    With[{mime = Quiet[Check[CloudObjectInformation[co, "MIMEType"], $Failed]]},
    	MemberQ[$taskMimeTypes, mime]
    ];
scheduledTaskQ[_] := False;

CloudObject /: RemoveScheduledTask[task_CloudObject, o:OptionsPattern[]] := With[
    {res = Catch[iCloudRemoveScheduledTask[task, RemoveScheduledTask, o], $tag]},
    res
];

iCloudRemoveScheduledTask[obj_CloudObject, mh_:RemoveScheduledTask, o:OptionsPattern[]] := (iCloudRemoveScheduledTask[safeCloudAndUUIDFetch[obj, mh], mh, o]; obj);
iCloudRemoveScheduledTask[{cloud_String, uuid_String}, mh_:RemoveScheduledTask, o:OptionsPattern[]] := Module[
    {json},
    json = Replace[execute[cloud, "DELETE", {"tasks", uuid}], {
        HTTPError[404, ___] :> (Message[ScheduledTask::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag]),
        {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content]),
        other_ :> (Message[mh::srverr]; Message[ScheduledTask::norm, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
    }];
    uuid
];

iCloudRemoveScheduledTask[st_, OptionsPattern[]] := (Message[ScheduledTask::norm, st]; $Failed);


(*
    11.2+. The same code with replaced RemoveScheduledTask --> TaskRemove
*)


CloudObject /: TaskRemove[co_CloudObject, o:OptionsPattern[]] /; Quiet[documentGeneratorQ[co]] := Module[
    {res = Catch[iCloudRemoveDocumentGenerator[co, TaskRemove, o], $tag]},
    res
]


CloudObject /: TaskRemove[ar_CloudObject, o:OptionsPattern[]] /; Quiet[autoRefreshedQ[ar]] := With[
    {res = Catch[iCloudRemoveAutoRefreshed[ar, TaskRemove, o], $tag]},
    res
]


CloudObject /: TaskRemove[task_CloudObject, o:OptionsPattern[]] := With[
    {res = Catch[iCloudRemoveScheduledTask[task, TaskRemove, o], $tag]},
    res
]


(*
    EvaluateScheduledTask (pre-11.2)
*)
Unprotect[EvaluateScheduledTask];

CloudObject /: EvaluateScheduledTask[co_CloudObject, o:OptionsPattern[]] /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := Module[
    {task = Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation, {}], "__Task", {}]},
    If[StringQ[task],
        Catch[EvaluateScheduledTask[cloudObjectFromUUID[task]], $tag]; co,
        Message[ScheduledTask::tasknf, co]; $Failed
    ]
];

CloudObject /: EvaluateScheduledTask[co_CloudObject] := Module[
	{expr = CloudGet[co]},
	Replace[expr, {
		ScheduledTask[File[path_], ___] :> (Get[path]),
		ScheduledTask[obj_CloudObject, ___] :> CloudGet[obj],
		ScheduledTask[str_String, ___] :> If[UUIDQ[str], Get[cloudObjectFromUUID[str]], str],
        ScheduledTask[code_, ___] :> (code),
        DocumentGenerator[template_, driver_, __] :> GenerateDocument[CloudGet[cloudObjectFromUUID[template]], CloudGet[cloudObjectFromUUID[driver]]]
        }
    ]
];

(*
   11.2+ Repeat the code for EvaluateScheduledTask replacing it with TaskEvaluate
*)

Unprotect[TaskEvaluate];

CloudObject /: TaskEvaluate[co_CloudObject, o:OptionsPattern[]] /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := Module[
    {task = Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation, {}], "__Task", {}]},
    If[StringQ[task],
        Catch[TaskEvaluate[cloudObjectFromUUID[task]], $tag]; co,
        Message[ScheduledTask::tasknf, co]; $Failed
    ]
];

CloudObject /: TaskEvaluate[co_CloudObject] := Module[
    {expr, i},
    
    Check[
        i = ScheduledTaskInformation[co],
        Return[$Failed]
    ];
    i = First[i];
    
    Switch[i["TaskType"],
        "file", Replace[i["Expression"], {s_String :> Get[cloudObjectFromUUID[s]]}],
        _, 
        (* Pre-1.22 tasks store their code in the cloud object and have execution expressions like
         *      "TaskEvaluate[CloudObject[\"http://www.wolframcloud.\com/objects/user-74e17eb9-8669-4795-b270-032b6ad916af/task\\"]]"
         * In 1.22+ this results in recursion if evaluated naively, so check for expressions of this form. 
         *)
        expr = Replace[i["Expression"], x_String :> ToExpression[x, InputForm, Hold]];
        Last @ List @ ReleaseHold @ Replace[expr, {
        	Hold[TaskEvaluate[co2_CloudObject]] :> Replace[
            CloudGet[co2],
            ScheduledTask[code_, ___] :> (code)
        ] /; CloudObjectInformation[co, "UUID"] === CloudObjectInformation[co2, "UUID"]
        }]
    ]
]



(*
 * Hybrid task listing
 *)
Unprotect[ScheduledTasks];
Unprotect[Tasks];
SetAttributes[ScheduledTasks, {ReadProtected}];
SetAttributes[Tasks, {ReadProtected}];

$cloudScheduledTasksFlag = True;
ScheduledTasks[] /; TrueQ[And[$CloudConnected, $cloudScheduledTasksFlag]] :=
    Block[{$cloudScheduledTasksFlag = False, cloudTasks = Catch[iCloudTasks[], $tag]},
        If[FailureQ[cloudTasks]||!ListQ[cloudTasks],
            Tasks[],
            Join[Tasks[], cloudTasks]
        ]
    ];
    
Tasks[] /; TrueQ[And[$CloudConnected, $cloudScheduledTasksFlag]] :=
    Block[{$cloudScheduledTasksFlag = False, cloudTasks = Catch[iCloudTasks[], $tag]},
        If[FailureQ[cloudTasks]||!ListQ[cloudTasks],
            Tasks[],
            Join[Tasks[], cloudTasks]
        ]
    ];

iCloudTasks[] := Module[
    {raw, med, json, msghd = ScheduledTasks},

    With[{mh = msghd},
        json = Replace[execute[$CloudBase, "GET", {"tasks"}, Parameters -> {"fields"->"uuid"}], {
            {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content])
            , other_ :> (Message[mh::srverr]; Throw[$Failed, $tag])
        }];

        Check[raw = Lookup[importFromJSON[json], "tasks", {Missing[]}],
            Message[mh::srverr];
            Throw[$Failed, $tag]
        ];
    ];
    tasks = If[StringQ[raw] && StringMatchQ[raw, "["~~___~~"]"], importFromJSON[raw], raw];
    If[!ListQ[tasks],
        Message[mh::srverr];
        Return[$Failed]
    ];

    Map[
        cloudObjectFromUUID[Lookup[#, "uuid"]] &,
        tasks
    ]
];


(* ScheduledTaskInformation *)
Unprotect[ScheduledTaskInformation, ScheduledTaskInformationData];
SetAttributes[ScheduledTaskInformation, {ReadProtected}];

ScheduledTaskInformation::noprop = "`1` is not a property returned by ScheduledTaskInformation.";


gatherNotificationFunction[n:{{__}...}] := SortBy[
    Sort[Lookup[#, "email"]] -> ToExpression[Lookup[#[[1]], "condition"]] & /@ GatherBy[n, Lookup["condition"]],
    Last
];


(* Output is a flat list digestible by the web ui. *)
denormalizeNotificationFunction[notifRaw_] := Module[
    {notif = Replace[notifRaw, {
        Null | None -> {},
        All -> {{$CloudUserID} -> All},
        Automatic -> {{$CloudUserID} -> Automatic},
        f_Function :> {{$CloudUserID} -> f},
        {u__String} :> {{u} -> Automatic},
        (* non-mail channels *)
        Rule[s_String, cond : Automatic | All] :> {Rule[{$CloudUserID}, {s, cond}]}
    }]},
    
    notif = Replace[
        notif,
        (addr_ -> cond_) :> Flatten[{addr}] -> Replace[cond, {None -> Null, ns:Except[_String] :> ToString[InputForm[ns]]}],
        {1}
    ];
    
    With[{pairs = DeleteDuplicates @ Flatten[(Thread[List @@ #1, List, 1] &) /@ notif, 1]},
        {"email" -> First@#, "condition" -> Last@#} & /@ pairs
    ]
];


$taskInfoNormalizationRules = {
    Rule[tag:"CreationDate"|"StartDate"|"EndDate"|"LastRunDate"|"NextRunDate", t_] /; t >= $neverRunDate :> 
        Rule[tag, None],
    Rule[tag:"CreationDate"|"StartDate"|"EndDate"|"LastRunDate"|"NextRunDate", Null|0] :> 
        Rule[tag, None],
    Rule[tag:"CreationDate"|"StartDate"|"EndDate"|"LastRunDate"|"NextRunDate", t_?NumericQ] :> 
        Rule[tag, FromUnixTime[Round[t/1000]]],
    Rule["Log", uuid_String] :> {Rule["Log", cloudObjectFromUUID[uuid]], Rule["LogUUID", uuid]},
    Rule[NotificationFunction, notif_] :> Rule[NotificationFunction, gatherNotificationFunction[notif]],
    Rule["Name", Null] -> Rule["Name", None],
    Rule[RestartInterval, r_] :> Rule[RestartInterval, ToExpression[r]],
    Rule["RepeatInterval", t_?NumericQ] :> Rule["RepeatInterval", Round[t/1000]],
    Rule["RepeatCount", -1|Null] :> Rule["RepeatCount", Infinity]
};


$taskInfoDenormalizationRules = {
    (* Make sure the time is not an integer here, to work around bug 289879. *)
    Rule[tag:"creationTimestamp"|"startTimestamp"|"endTimestamp"|"lastTimestamp"|"nextTimestamp", t:(_DateObject|_?NumericQ)] :> 
        Rule[tag, ToString@Round[1000*UnixTime[t]]],
    Rule[tag:"creationTimestamp"|"startTimestamp"|"endTimestamp"|"lastTimestamp"|"nextTimestamp", t:None] :> 
        Rule[tag, Null],
    Rule["name", None] -> Rule["name", Null],
    Rule["notificatees", notif_] :> Rule["notificatees", denormalizeNotificationFunction[notif]],
    Rule["restartInterval", r_] :> Rule["restartInterval", ToString[InputForm[r]]],
    Rule["repeatInterval", t_?NumericQ] :> Rule["repeatInterval", Round[1000*t]],
    Rule["repeatCount", Infinity|DirectedInfinity] -> Rule["repeatCount", -1]
};


$taskMetaToWLKeyMap = Association[
    "active" -> "Active", 
    "autoRemove" -> AutoRemove, 
    "completed" -> "Completed", 
    "continuous" -> "Continuous", 
    "cronSchedule" -> "CronSchedule", 
    "endTimestamp" -> "EndDate",
    "executions" -> "Executions",
    "failures" -> "Failures", 
    "lastTimestamp" -> "LastRunDate", 
    "log" -> "Log", 
    "name" -> "Name", 
    "nextTimestamp" -> "NextRunDate", 
    "notificatees" -> NotificationFunction,
    "owner" -> "Owner", 
    "paused" -> "Paused",
    "quartzTablePrefix" -> "QuartzTablePrefix", 
    "remainingTriggers" -> "RemainingTriggers",
    "repeatCount" -> "RepeatCount", 
    "repeatInterval" -> "RepeatInterval", 
    "restartInterval" -> RestartInterval, 
    "startTimestamp" -> "StartDate", 
    "creationTimestamp" -> "CreationDate", 
    "status" -> "Status", 
    "taskData" -> "Expression", 
    "taskType" -> "TaskType", 
    "timeZone" -> TimeZone,
    "uuid" -> "UUID",
    "visible" -> "Visible"
];

$WLToTaskMetaKeyMap = Association[Reverse /@ Normal[$taskMetaToWLKeyMap]];

taskMetaToWL[raw_List, properties_List] := Module[
    {med, well},
        (* Replace json keys with WL symbols/strings *)
        med = DeleteCases[
            Replace[raw, Rule[lhs_, rhs_] :> Rule[$taskMetaToWLKeyMap[lhs], rhs], 1],
            Rule[Missing[__], _]
        ];
        well = Association @@ Flatten[Replace[med, $taskInfoNormalizationRules, 1], 1];
        well[TimeZone] = Quiet@Replace[well["TimeZoneFullName"], {
            name_String :> Entity["TimeZone", name] /; MemberQ[EntityList["TimeZone"], Entity["TimeZone", name]],
            _ -> well[TimeZone]
        }];
        If[well["Continuous"],
            well["EndDate"] = Lookup[
                Lookup[Options[cloudObjectFromUUID[well["UUID"]], MetaInformation -> "__ContinuousEndDate"], MetaInformation],
                "__ContinuousEndDate"
            ];
            well["Completed"] = TrueQ[Now > well["EndDate"]];
            well["RepeatInterval"] = Null;
        ];
        Join[Association@@Map[Rule[#, None]&, properties], KeySort @ well]
];

$defaultTaskProperties = {
    "Active",
    AutoRemove,
    "Completed",
    "Continuous",
    "CronSchedule",
    "EndDate",
    "Executions",
    "Expression",
    "Failures",
    "LastRunDate",
    "Log",
    "Name",
    "NextRunDate",
    NotificationFunction,
    "Paused",
    "RepeatCount",
    "RepeatInterval",
    RestartInterval,
    "StartDate",
    "Status",
    "TaskType",
    TimeZone,
    "UUID"
};

ScheduledTaskInformation[objs:{__CloudObject}] := Replace[
	Catch[iCloudScheduledTaskInformation[objs, $defaultTaskProperties, ScheduledTaskInformation], $tag], {
		res:{_Association} :> Dataset[res[[1]]],
		res:{__Association} :> Dataset[res]
	}];
	
ScheduledTaskInformation[objs:{__CloudObject}, properties_List] := Replace[
    Catch[iCloudScheduledTaskInformation[objs, properties, ScheduledTaskInformation], $tag], {
        res:{_Association} :> Dataset[res[[1]]],
        res:{__Association} :> Dataset[res]
    }];
    
ScheduledTaskInformation[objs:{__CloudObject}, property_] := ScheduledTaskInformation[objs, {property}]

ScheduledTaskInformation[obj_CloudObject] := Replace[
    Catch[iCloudScheduledTaskInformation[{obj}, $defaultTaskProperties, ScheduledTaskInformation], $tag], {
        res:{_Association} :> Dataset[res[[1]]],
        res:{__Association} :> Dataset[res]
}];

ScheduledTaskInformation[obj_CloudObject, properties_List] := Replace[
    Catch[iCloudScheduledTaskInformation[{obj}, properties, ScheduledTaskInformation], $tag], {
        res:{_Association} :> Dataset[res[[1]]],
        res:{__Association} :> Dataset[res]
    }];

ScheduledTaskInformation[obj_CloudObject, property_] := Normal[ScheduledTaskInformation[{obj}, {property}]][property]

iCloudScheduledTaskInformation[objs:{CloudObject__}, properties_List, msghd_:ScheduledTaskInformation] :=
    Module[{fields, uuids, cloud = $CloudBase, json, raw, tasks},
        fields = formatOutgoingFields[properties];
        tasks = Map[getTask, objs];
        uuids = Map[safeCloudAndUUIDFetch[#, msghd]&, tasks][[All, 2]];
        json = Replace[
            execute[cloud, "GET", {"tasks"}, Parameters -> {"uuids"->commaSeparated[uuids], "fields"->commaSeparated[fields]}],
            {
                HTTPError[404, ___] :> (Message[ScheduledTask::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag]),
                    {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content]),
                HTTPError[410, ___] :> (mess = ToExpression[Lookup[importFromJSON[content], "error"], InputForm, Hold];
                ReleaseHold[Replace[mess, slug_ :> Message[slug, cloudObjectFromUUID[uuid]], 1]];
            )
            , other_ :> (Message[msghd::srverr]; Throw[$Failed, $tag])
            }
        ];
        
        Check[raw = importFromJSON[Lookup[importFromJSON[json], "tasks"]],
            Message[msghd::srverr];
            Throw[$Failed, $tag]
        ];
        
        Map[
	        	KeySelect[
	            If[StringMatchQ[Lookup[#, "taskType"], "script"],
	                taskMetaToWL[#, properties],
	                formatTaskInformation[#, properties]
	            ],
	            (MemberQ[properties, #]&)
	        ]&,
	        raw
        ]
    ]

formatOutgoingFields[properties_] := Union@@Append[Map[formatOutgoingField, properties], {"taskType"}]

formatOutgoingField["Active"]={"active","status"};
formatOutgoingField[AutoRemove]={"autoRemove","taskData"};
formatOutgoingField["Completed"]={"completed","status"};
formatOutgoingField["Continuous"]={"continuous","taskData"};
formatOutgoingField["CronSchedule"]={"cronSchedule"};
formatOutgoingField["EndDate"]={"endTimestamp"};
formatOutgoingField["Executions"]={"executions"};
formatOutgoingField["Expression"]={"taskData"};
formatOutgoingField["Failures"]={"failures"};
formatOutgoingField["LastRunDate"]={"lastTimestamp"};
formatOutgoingField["Log"]={"log","uuid"};
formatOutgoingField["Name"]={"name","uuid"};
formatOutgoingField["NextRunDate"]={"nextTimestamp"};
formatOutgoingField[NotificationFunction]={"notificatees","taskData"};
formatOutgoingField["Paused"]={"paused"};
formatOutgoingField["QuartzTablePrefix"]={"quartzTablePrefix"};
formatOutgoingField["RepeatCount"]={"repeatCount"};
formatOutgoingField["RepeatInterval"]={"repeatInterval"};
formatOutgoingField[RestartInterval]={"restartInterval","taskData"};
formatOutgoingField["StartDate"]={"startTimestamp"};
formatOutgoingField["Status"]={"status"};
formatOutgoingField["TaskType"]={"taskType"};
formatOutgoingField[TimeZone]={"timeZone","taskData"};
formatOutgoingField["UUID"]={"uuid"};
formatOutgoingField[other_] := (Message[ScheduledTaskInformation::noprop, other]; {})

getTask[co_CloudObject] /; Quiet[documentGeneratorQ[co]] := cloudObjectFromUUID[Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation], "__Task"]]
getTask[co_CloudObject] /; Quiet[autoRefreshedQ[co]] := cloudObjectFromUUID[Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation], "__Task"]]
getTask[co_] := co

formatTaskInformation[data_, properties_] :=
    Module[{assoc, formatKeys, computedValues},
        assoc = Association@@data;
        formatKeys = KeyMap[formatIncomingField, assoc];
        computedValues = Map[computeTaskValue[#, formatKeys]&, properties];
        Join[Association@@Map[Rule[#, None]&, properties], Association@@computedValues]
    ];

formatIncomingField["status"]="Status";
formatIncomingField["autoRemove"]=AutoRemove;
formatIncomingField["taskData"]="Expression";
formatIncomingField["continuous"]="Continuous";
formatIncomingField["cronSchedule"]="CronSchedule";
formatIncomingField["endTimestamp"]="EndDate";
formatIncomingField["executions"]="Executions";
formatIncomingField["failures"]="Failures";
formatIncomingField["lastTimestamp"]="LastRunDate";
formatIncomingField["log"]="Log";
formatIncomingField["uuid"]="UUID";
formatIncomingField["name"]="Name";
formatIncomingField["nextTimestamp"]="NextRunDate";
formatIncomingField["notificatees"]=NotificationFunction;
formatIncomingField["paused"]="Paused";
formatIncomingField["quartzTablePrefix"]="QuartzTablePrefix";
formatIncomingField["repeatCount"]="RepeatCount";
formatIncomingField["repeatInterval"]="RepeatInterval";
formatIncomingField["restartInterval"]=RestartInterval;
formatIncomingField["startTimestamp"]="StartDate";
formatIncomingField["taskType"]="TaskType";
formatIncomingField["timeZone"]=TimeZone;

computeTaskValue["Active", data_] := Rule["Active", !StringMatchQ[Lookup[data, "Status", "paused"], "paused"]];
computeTaskValue[AutoRemove, data_] := Rule[AutoRemove,
	With[{opts = Cases[getFormattedTaskExpression[data["Expression"]], _Rule|_RuleDelayed]}, Lookup[opts, AutoRemove, Lookup[data, AutoRemove]]]];
computeTaskValue["Completed", data_] := Rule["Completed", StringMatchQ[Lookup[data, "Status", "missing"], "completed"]];
computeTaskValue["Continuous", data_] := Rule["Continuous", Or[TrueQ[Lookup[data, "continuous"]], StringMatchQ[data["Expression"], ___~~"ContinuousTask"~~__]]];
computeTaskValue["CronSchedule", data_] := Rule["CronSchedule", Lookup[data, "CronSchedule", None]];
computeTaskValue["EndDate", data_] := Rule["EndData",
	With[{end = Lookup[data, "EndDate"]}, If[MissingQ[end], None, FromUnixTime[data["EndDate"]/1000]]]];
computeTaskValue["Executions", data_] := Rule["Executions", Lookup[data, "Executions", 0]];
computeTaskValue["Expression", data_] := Rule["Expression", getFormattedTaskExpression[data["Expression"]]];
computeTaskValue["Failures", data_] := Rule["Failures", Lookup[data, "Failures", 0]];
computeTaskValue["LastRunDate", data_] := Rule["LastRunDate",
	With[{last = Lookup[data, "LastRunDate"]}, If[MissingQ[last], None, FromUnixTime[data["LastRunDate"]/1000]]]];
computeTaskValue["Log", data_] := Rule["Log", CloudObject[StringJoin["Base/Logs/",data["UUID"],".log"]]];
computeTaskValue["Name", data_] := Rule["Name", 
    With[{name = data["Name"]}, Which[UUIDQ[FileNameTake[name]], data["UUID"], StringContainsQ[name, ".Objects"], FileNameTake[name, -2], True, name]]];
computeTaskValue["NextRunDate", data_] := Rule["NextRunDate",
	With[{next = Lookup[data, "NextRunDate"]}, If[MissingQ[next], None, FromUnixTime[data["NextRunDate"]/1000]]]];
computeTaskValue[NotificationFunction, data_] := Rule[NotificationFunction,
	With[{opts = Cases[getFormattedTaskExpression[data["Expression"]], _Rule|_RuleDelayed]}, Lookup[opts, NotificationFunction, Lookup[data, "notificatees"]]]];
computeTaskValue["Paused", data_] := Rule["Paused", TrueQ[Lookup[data, "Paused"]]];
computeTaskValue["QuartzTablePrefix", data_] := Rule["QuartzTablePrefix", data["QuartzTablePrefix"]];
computeTaskValue["RepeatCount", data_] := Rule["RepeatCount", Replace[data["RepeatCount"], {-1 -> Infinity}]];
computeTaskValue["RepeatInterval", data_] := Rule["RepeatInterval", Lookup[data, "RepeatInterval", 0]/1000];
computeTaskValue[RestartInterval, data_] := Rule[RestartInterval,
	With[{opts = Cases[getFormattedTaskExpression[data["Expression"]], _Rule|_RuleDelayed]}, Lookup[opts, RestartInterval, Lookup[data, RestartInterval, Quantity[1, "Hours"]]]]];
computeTaskValue["StartDate", data_] := Rule["StartDate",
	With[{start = Lookup[data, "StartDate"]}, If[MissingQ[start], None, FromUnixTime[data["StartDate"]/1000]]]];
computeTaskValue["Status", data_] := Rule["Status", formatTaskStatus[data["Status"]]];
computeTaskValue["TaskType", data_] := Rule["TaskType", data["TaskType"]];
computeTaskValue[TimeZone, data_] := Rule[TimeZone,
	With[{opts = Cases[getFormattedTaskExpression[data["Expression"]], _Rule|_RuleDelayed]}, Lookup[opts, TimeZone, Lookup[data, TimeZone, 0]]]];
computeTaskValue["UUID", data_] := Rule["UUID", data["UUID"]];
computeTaskValue[_, _] := Nothing[]

getFormattedTaskExpression[expr_] := getFormattedTaskExpression[expr] = ToExpression[expr]

formatTaskStatus[None] := None
formatTaskStatus[status_String] :=
    formatTaskStatus[status] = StringReplace[ToLowerCase[status], f_ ~~ rest___ :> StringJoin[{ToUpperCase[f], ToLowerCase[rest]}]]
formatTaskStatus[s_] := s

CloudObject /: ScheduledTaskActiveQ[co_CloudObject, o:OptionsPattern[]] /; Quiet[documentGeneratorQ[co]] := With[
    {i = Catch[iCloudDocumentGeneratorInformation[co], $tag]},
    If[MatchQ[i, _Association],
        (*i["Active"] && *)Not[i["Task"]["Paused"]] && Not[i["Task"]["Completed"]],
        (* else *)
        i
    ] 
];

CloudObject /: ScheduledTaskActiveQ[co_CloudObject] := With[
    {i = Catch[Normal[ScheduledTaskInformation[co, {"Paused", "Completed"}]], $tag]},
    If[MatchQ[i, _Association],
        (*i["Active"] && *)Not[i["Paused"]] && Not[i["Completed"]],
        (* else *)
        i
    ]
];


CloudObject /: NextScheduledTaskTime[co_CloudObject, o:OptionsPattern[]] /; Quiet[documentGeneratorQ[co]] := With[
    {i = Catch[iCloudDocumentGeneratorInformation[co], $tag]},
    If[MatchQ[i, _Association],
        Lookup[Normal[Lookup[i, "Task"]], "NextRunDate", Message[DocumentGenerator::nonext, co]; $Failed],
        (* else *)
        i
    ] 
]


CloudObject /: NextScheduledTaskTime[task_CloudObject] := With[
    {i = Catch[ScheduledTaskInformation[task, "NextRunDate"], $tag]},
    If[MatchQ[i, _Association],
        Lookup[i, "NextRunDate", Message[ScheduledTask::nonext, co]; $Failed],
        (* else *)
        i
    ] 
];


(*
    AbortScheduledTask (pre-11.2)
*)

Unprotect[AbortScheduledTask];
SetAttributes[AbortScheduledTask, {ReadProtected}];

CloudObject /: AbortScheduledTask[co_CloudObject] /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := Module[
    {task = Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation, {}], "__Task", {}]},
    If[StringQ[task],
        Catch[TaskAbort[cloudObjectFromUUID[task]], $tag]; co,
        Message[ScheduledTask::tasknf, co]; $Failed
    ]
];

CloudObject /: AbortScheduledTask[task_CloudObject] := With[
    {res = Catch[iCloudAbortScheduledTask[task], $tag]},
    res
];

iCloudAbortScheduledTask[obj_CloudObject, mh_:ScheduledTask] := (iCloudAbortScheduledTask[safeCloudAndUUIDFetch[obj, mh], mh]; obj)
iCloudAbortScheduledTask[{cloud_String, uuid_String}, mh_:ScheduledTask] := Module[
    {raw, json},

    json = Replace[execute[cloud, "POST", {"tasks", uuid, "abort"}], {
        HTTPError[404, ___] :> (Message[mh::tasknf, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
        , HTTPError[409, ___] :> (Message[mh::norun, cloudObjectFromUUID[uuid]]; Return[obj])
        , {_String, content_List} :> ($lastInfoJSON = FromCharacterCode[content])
        , other_ :> (Message[mh::srverr, cloudObjectFromUUID[uuid]]; Throw[$Failed, $tag])
    }];

    Check[raw = Lookup[importFromJSON[json], "status"],
        Message[mh::srverr];
        Throw[$Failed, $tag]
    ];
    
    uuid
];


(*
    11.2+ AbortScheduledTask --> TaskAbort
*)

CloudObject /: TaskAbort[co_CloudObject] /; Quiet[Or[documentGeneratorQ[co], autoRefreshedQ[co]]] := Module[
    {task = Lookup[Lookup[Options[co, MetaInformation -> "__Task"], MetaInformation, {}], "__Task", {}]},
    If[StringQ[task],
        Catch[TaskAbort[cloudObjectFromUUID[task]], $tag]; co,
        Message[ScheduledTask::tasknf, co]; $Failed
    ]
];

CloudObject /: TaskAbort[task_CloudObject] := With[
    {res = Catch[iCloudAbortScheduledTask[task], $tag]},
    res
]


(* AutoRefreshed *)

Unprotect[AutoRefreshed];
SetAttributes[AutoRefreshed, {HoldFirst, ReadProtected}];
Options[AutoRefreshed] = {
};

AutoRefreshed /: CloudDeploy[
    ct:AutoRefreshed[
        expr_, 
        tspec:Except[_Rule|_RuleDelayed]:3600, 
        fmt:Except[_Rule|_RuleDelayed]:"WL", 
        o:OptionsPattern[AutoRefreshed]
    ], 
    co_CloudObject, 
    oCD:OptionsPattern[CloudDeploy]
    ] := 
    Catch[autoRefreshedStaging[AutoRefreshed[expr, tspec, fmt, o], co, oCD], $tag];


(*
 * This function hijacks the InstantAPIServer internals to provide the documented support
 * for export and response forms.
 *) 
SetAttributes[exploy, HoldFirst];
Options[exploy] = Options[CloudDeploy];

exploy[expr_, fmt_, dest:CloudObject[uri_, destOpts:OptionsPattern[CloudObject]], o:OptionsPattern[]] := 
    Block[
        {CloudObject`$EvaluationParameters = <||>},
        (* This will basically use Delayed to render. *)
        Replace[
            Map[GenerateHTTPResponse[AutoRefreshed[expr, None, fmt]], {"Body", "ContentType"}],
            {body_, contentType_} :> 
                CloudObject`Private`writeObject[
                    dest, 
                    ToCharacterCode[body], 
                    contentType,
                    OptionValue[Permissions],
                    OptionValue[CloudObject, {destOpts}, IconRules],
                    Unevaluated[expr],
                    OptionValue[CloudObject, {destOpts}, MetaInformation],
                    {},
                    AutoRefreshed
                ]    
        ]
    ];

autoRefreshedStaging[AutoRefreshed[expr_, sched_, fmt_, o:OptionsPattern[]], co_CloudObject, oCD:OptionsPattern[]] :=
    Module[{cloud, uuid, name, runImmediately = False, cronned, continuous = TrueQ[Lookup[Flatten[{o}], "Continuous", False]],
        	permissions, iconRules, metaInformation, taskExpr, params, endpoint, updating},

        {cloud, uuid, name} = getCloudUUIDAndPathForTask[co];

        If[autoRefreshedQ[co],
	        updating = True;
	    ];

        {runImmediately, cronned} = Which[
            continuous,
            {True, resolveTimespec[ReleaseHold[sched]]},
            nowQ[sched],
            {True, resolveTimespec[None]},
            True,
            {False, resolveTimespec[ReleaseHold[sched]]}
        ];

        If[MatchQ[cronned, $Failed],
            Message[AutoRefreshed::sched, sched];
            Throw[$Failed, $tag]
        ];

        permissions = Lookup[Flatten[{oCD}], Permissions];
        iconRules = Lookup[Flatten[{oCD}], IconRules];
        metaInformation = Lookup[Flatten[{oCD}], MetaInformation];

        taskExpr = AutoRefreshed[expr, cronned, fmt, Join[Flatten[{o}], Options[ScheduledTask]]];

        params = {"bundle" -> ($lastJSON = generateTaskJson[taskExpr, {name, uuid}, cronned, Join[Flatten[{o}],
                  {"Visible" -> False, "RunImmediately" -> runImmediately}, Options[ScheduledTask]]])};

        endpoint = If[TrueQ[updating], {"tasks", "autorefreshed", Last@safeCloudAndUUIDFetch[co, AutoRefreshed]}, {"tasks", "autorefreshed"}];
        Replace[
            execute[cloud, "POST", endpoint,
                Body -> ToCharacterCode[exportToJSON[params], "UTF-8"],
                Type -> "application/vnd.wolfram.bundle.autorefreshed"
            ], {
                {_String, b_List} :> ($lastInfoJSON = FromCharacterCode[b]),
                HTTPError[400, ___] :> (Message[AutoRefreshed::argu]; Throw[$Failed, $tag]),
                HTTPError[403, b_, ___] :> (
                    ToExpression[Lookup[ImportString[b, "JSON"], "error", "ScheduledTask::restr"], InputForm, Message];
                    Throw[$Failed, $tag]
                ),
                other_ :> (Message[CloudObject::srverr]; Message[cloudObject::crea]; Throw[$Failed, $tag])
            }
        ];

        co
    ]

(* etc. *)
GetNameFromURI[uri_String] := With[{split = StringSplit[uri, "/"]},
    If[Length[split] < 2, Message[ScheduledTask::nouri, uri]; Throw[$Failed, $tag], Last[split]]
];

GetUUID[obj_CloudObject] := Module[{res}, If[MatchQ[res = getCloudAndUUID[obj], {_, id_String}], Last[res], Throw[$Failed, $tag]]];
GetUUID[obj_String] := GetUUID[CloudObject[obj]];
GetUUID[___] := Throw[$Failed, $tag];

safeCloudAndUUIDFetch[CloudObject`Private`deleteable[obj_CloudObject], mh_:ScheduledTask] := safeCloudAndUUIDFetch[obj, mh];
safeCloudAndUUIDFetch[CloudObject`Private`preexisting[obj_CloudObject], mh_:ScheduledTask] := safeCloudAndUUIDFetch[obj, mh];
safeCloudAndUUIDFetch[obj_CloudObject, mh_:ScheduledTask] := Replace[getCloudAndUUID[obj], {
    {_, None} :> (Message[mh::cloudnf, obj]; Throw[$Failed, $tag])
}];
safeCloudAndUUIDFetch[None, mh_] := {$CloudBase, Null};
safeCloudAndUUIDFetch[___] := Throw[$Failed, $tag];


Protect[ScheduledTask, CloudObject, ScheduledTasks, Tasks, EvaluateScheduledTask, ScheduledTaskInformation, ScheduledTaskInformationData,
    ContinuousTask, AbortScheduledTask, AutoRefreshed, Tasks`TaskEvaluate, TaskExecute, TaskAbort];

$Flag = False;


(* begin helper functions for timespec2cron[DateObject] *)
DatePatternQ[list_List] := MatchQ[list, {_?DatePatternElementQ ..}];

$DaysOfTheWeek = {Sunday, Monday, Tuesday, Wednesday, Thursday, 
   Friday, Saturday};
DatePatternElementQ[_?NumberQ] := True;
DatePatternElementQ[Verbatim[Blank[]]] := True;
DatePatternElementQ[day_Symbol] := MemberQ[$DaysOfTheWeek, day];
DatePatternElementQ[
  Verbatim[Alternatives][_?DatePatternElementQ ..]] := True;
DatePatternElementQ[___] := False;

QuartzValueQ[year_Integer, {1}] := TrueQ[1970 <= year <= 2099];
QuartzValueQ[month_Integer, {2}] := TrueQ[1 <= month <= 12];
QuartzValueQ[dayofmonth_Integer, {3}] := TrueQ[1 <= dayofmonth <= 31];
QuartzValueQ[hour_Integer, {4}] := TrueQ[0 <= hour <= 23];
QuartzValueQ[minute_Integer, {5}] := TrueQ[0 <= minute <= 59];
QuartzValueQ[seconds_Integer, {6}] := 
 TrueQ[0 <= seconds <= 59];(*probably need numberQ*)

QuartzValueQ[___] := False;


Clear[ElementToCron];
ElementToCron[element_?DatePatternElementQ, n : Except[{3}]] := 
  Which[QuartzValueQ[element, n], ToString[Ceiling[element]], 
   MatchQ[element, Verbatim[Blank[]]], "*", 
   MatchQ[element, 
    Verbatim[Alternatives][
     Repeated[PatternTest[Blank[], QuartzValueQ[#, n] &]]]], 
   StringJoin[
    Riffle[ToString[Ceiling[#]] & /@ (List @@ element), ","]], True, 
   Throw[$Failed, $tag]];

(*DOM[] is a wrapper for "day of the month" and DOW[] is for "day of \
the week"*)

ElementToCron[day_?DatePatternElementQ, n : {3}] := 
 Which[QuartzValueQ[day, n], DOM[ToString[Ceiling[day]]], 
  MatchQ[day, Verbatim[Blank[]]], DOM["*"], 
  MemberQ[$DaysOfTheWeek, Verbatim[day]], 
  DOW[ToString[First[Flatten[Position[$DaysOfTheWeek, day]]]]], 
  MatchQ[day, 
   Verbatim[Alternatives][
    Repeated[PatternTest[Blank[], MemberQ[$DaysOfTheWeek, #] &]]]], 
  DOW[StringJoin[
    Riffle[ToString[
        First[Flatten[Position[$DaysOfTheWeek, #]]]] & /@ (List @@ 
        day), ","]]], 
  MatchQ[day, 
   Verbatim[Alternatives][
    Repeated[PatternTest[Blank[], QuartzValueQ[#, n] &]]]], 
  DOM[StringJoin[
    Riffle[ToString[Ceiling[#]] & /@ (List @@ day), ","]]], True, 
  Throw[$Failed, $tag]];

ElementToCron[___] := Throw[$Failed, $tag];

currentTime[spec_] := DateString[DateList[], spec];

(* should probably get minutes and seconds in here as well 
Changed the name here
*)
$current := {currentTime["MonthShort"], currentTime["DayShort"], 
    currentTime["HourShort"], currentTime["MinuteShort"], 
    currentTime["SecondShort"]};

Clear[PadAppropriately];
PadAppropriately[list_List] := 
 Join[list, Take[$current, {Length[list], -1}]];


Clear[OrderForDOM, OrderForDOW];
OrderForDOM[list_List] := 
 Insert[Reverse[PadAppropriately[list]], "?", 6];

OrderForDOW[list_List] := 
 Insert[Part[PadAppropriately[list], {6, 5, 4, 2, 3, 1}], "?", 4];

DateObjectToCronSpecification[HoldPattern[dObj_DateObject?DateObjectQ], target_: $TimeZone] := 
With[{d = DateAndTime`DateObjectToDateList[dObj, target]}, 
    StringJoin[
        Riffle[Join[
            Reverse[ToString[Ceiling[#]] & /@ Rest[d]], {"?"}, {ToString[
                First[d]]}],
            " "
        ]
    ]
];

DateObjectToCronSpecification[HoldPattern[dObj : DateObject[_List, TimeObject[time_List, ___?OptionQ], ___?OptionQ]?DateObjectQ], target_: $TimeZone] :=
With[{d = DateAndTime`DateObjectToDateList[dObj, target]}, 
    StringJoin[
        Riffle[Join[
            ToString[Ceiling[#]] & /@ 
                Reverse[Rest[d]], {"?"}, {ToString[First[d]]}],
            " "
        ]
    ]
];

DateObjectToCronSpecification[DateObject[l_?DatePatternQ]] := 
    If[0 < Length[l] < 7, 
        Catch[With[{cron = MapIndexed[ElementToCron, l]}, 
            StringJoin[
                Riffle[If[FreeQ[cron, DOW], OrderForDOM[cron], 
                    OrderForDOW[cron]] /. {DOM[d_] :> d, DOW[d_] :> d},
                    " "
                ]
            ]],
            $tag],
        $Failed
    ];

DateObjectToCronSpecification[HoldPattern[dObj_DateObject?DateObjectQ]] := DateObjectToCronSpecification[dObj, $TimeZone]

DateObjectToCronSpecification[___] := $Failed;


End[]

EndPackage[]
