(* :Name: HPC Library File *)

(* :Title: Integration of gridMathematica into Resource Management Systems *)

(* :Context: ClusterIntegration`HPC` *)

(* :Author: Charles Pooh *)

(* :Summary:

    This package provides functionalities for integrating gridMathematica
    into Microsoft High Performance Computing Server.

*)

(* :Copyright: 1986 - 2012 Wolfram Research, Inc. *)

(* :Sources: *)

(* :Package Version: 2.0 *)

(* :Mathematica Version: 9.0 *)

(* :History: *)

(* :Keywords: None *)

(* :Warnings: None *)

(* :Limitations: *)

(* :Discussion: *)

(* :Requirements: *)

(* :Examples: None *)


(*****************************************************************************)


BeginPackage["ClusterIntegration`HPC`", {"ClusterIntegration`", "CURLLink`"}]

If[$OperatingSystem === "Windows", Needs["NETLink`"]]


(* usage *)

HPCNewComputeKernels::usage = "Internal function. Use NewKernels to launch \
remote kernels."

HPCLoginScript::usage = "Internal functions. Use Kernel Configuration menu \
to configure kernels."

HPCMathBatchJob::usage = "Internal functions. Use the Batch Submission Job \
palette to submit batch jobs."


Begin["`Private`"]


(* options *)

$HPCOptions :=
    {
        "EnginePath" -> Automatic,
        "KernelOptions" -> "-subkernel -mathlink -LinkProtocol TCPIP -LinkMode Connect -LinkName `linkname`",
        "KernelProgram" -> FileNameJoin[{$InstallationDirectory, First[$CommandLine]}],
        "NetworkInterface" -> "",
        "ToQueue" -> False,
        "Username" -> "",
        "Password" -> "",
        "JobOptions" -> {},
        "TaskOptions" -> {}
    }


Options[HPC] := Sort[$HPCOptions]


(* HTTP parameters *)


`$HTTPHeaders = {"api-version" -> "2011-11-01", "Content-Type" -> "application/xml; chartset=utf-8"}
`$HTTPBody = "<ArrayOfProperty xmlns=\"http://schemas.microsoft.com/HPCS2008R2/common\">\n`body`\n</ArrayOfProperty>"


(* ************************************************************************* **

                           HPCNewComputeKernels


   Comments:

   ToDo:

** ************************************************************************* *)


HPCNewComputeKernels[headnode_String, kcount:{_Integer, _Integer}, opts___?OptionQ] :=
    Block[{res, res1, job, nopts, scheduler, tasks, method, version},

        nopts   = iFilterRules[opts, Options[HPC]];
        method  = getConnection[headnode, nopts];

        DebugPrint[" Launching remote kernels ...",
                   "\n Cluster type: HPC",
                   "\n Cluster headnode: ", headnode,
                   "\n Cluster connection: ", method,
                   "\n Number of kernels: ", kcount,
                   "\n Cluster info: ", TableForm[Options[HPC]],
                   "\n User options: ", TableForm[Flatten[{opts}]]
        ];

        res = CheckAbort[

            Catch[

                scheduler = connectScheduler[method, headnode, nopts];
                If[!FreeQ[scheduler, $Failed], Throw[$Failed]];
                DebugPrint["-- Connected to scheduler."];

                version = getVersion[method, scheduler, nopts];
                If[!FreeQ[version, $Failed], Throw[$Failed]];
                DebugPrint["   version: " <> ToString[version]];

                job = createJob[method, scheduler, version, kcount, nopts];
                If[!FreeQ[job, $Failed], Throw[$Failed]];
                DebugPrint["-- Job created."];
                DebugPrint["   jobID: " <> If[method === ".NET", ToString[job@Id], job]];

                tasks = createTasks[method, scheduler, version, job, kcount, nopts];
                If[!FreeQ[tasks, $Failed], Throw[$Failed]];
                DebugPrint["-- Tasks added."];

                res1 = submitJob[method, scheduler, job, nopts];
                If[!FreeQ[res1, $Failed], Throw[$Failed]];
                DebugPrint["-- Job submitted."];

                res1 = waitingResources[method, scheduler, job, nopts];
                If[!FreeQ[res1, $Failed], Throw[$Failed]];
                DebugPrint["-- Job running."];

            ],

            cancelJob[scheduler, job, tasks];
            DebugPrint["Aborted - Job cancelled."];
            Abort[]

        ];

        AbortProtect[
            If[res === $Failed, cancelJob[scheduler, job, tasks]]
        ];

        (
          If[method === ".NET",
               res = job@Id;
              res1 = Through[tasks[[All, 1]][TaskId]];
               ReleaseNETObject[job];
               ReleaseNETObject /@ tasks[[All, 1]]
            ,
               res = job;
               res1 = tasks[[All, 1]]
          ];

          {tasks[[All, 2]], headnode, nopts, 0, res, res1, scheduler}

        ) /; FreeQ[res, $Failed]

    ]


HPCNewComputeKernels[___] := $Failed


(* :connectCluster: *)

connectScheduler[".NET", headnode_, opts_] :=
    Block[{res, scheduler, dir},

        dir = getEnginePath[".NET", OptionValue[opts, "EnginePath"]];
        (
          res = Needs["NETLink`"];
          (
            res = InstallNET[];
            (
              res = iLoadNETAssembly["Microsoft.Hpc.Scheduler", dir];
              (
                scheduler = NETNew["Microsoft.Hpc.Scheduler.Scheduler"];
                (
                   res = scheduler@Connect[headnode];
                   scheduler /; FreeQ[res, $Failed]

                ) /; Head[scheduler] === Symbol

              ) /; VectorQ[Flatten[{res}], (Head[#] === NETAssembly) &]

            ) /; Head[res] === LinkObject

          ) /; (res =!= $Failed)

        ) /; StringQ[dir]

    ]


connectScheduler["HTTP", headnode_, opts_] :=
    Block[{res, scheduler, uri},

        uri = iStringJoin[headnode, "WindowsHPC", "Clusters"];
        (
            res = Needs["CURLLink`"];
            (
               res = URLFetch[uri, "Headers" -> $HTTPHeaders, "VerifyPeer" -> False,
                              "Username" -> OptionValue[opts, "Username"], "Password" -> OptionValue[opts, "Password"]];
               (
                 scheduler = StringCases[res, "<Property><Name>Name</Name><Value>" ~~ x__ ~~ "</Value></Property>" :> x];
                 (
                   res = iStringJoin[headnode, "WindowsHPC", First[scheduler]];
                   res /; (res =!= $Failed)

                 ) /; ListQ[scheduler] && (scheduler =!= {})

               ) /; StringQ[res]

            ) /; (res =!= $Failed)

        ) /; StringQ[uri]

    ]

connectScheduler[___] :=
    (Message[HPC::load, "HPC"]; $Failed)


(* :createJob: *)

createJob[".NET", 2, scheduler_, kcount_, opts_] :=
    Block[{res, job, name},

        job = scheduler@CreateJob[];
        (
          name = StringJoin["Wolfram gridMathematica - sessionID:",
                            ToString[$SessionID], "-", ToString[$ModuleNumber]];

          res = iSetOptions[job, "JobOptions" /. opts /. {"JobOptions" -> {}},
                              "Name" -> name,
                              "MaximumNumberOfProcessors" -> Last[kcount],
                              "MinimumNumberOfProcessors" -> Last[kcount]];
          (
            res = scheduler@AddJob[job];
            job /; (res === Null)

          ) /; (res =!= $Failed)

        ) /; FreeQ[job, $Failed]

    ]

createJob[".NET", scheduler_, version_, kcount_, opts_] :=
    Block[{res, job, name},

        job = scheduler@CreateJob[];
        (
          name = StringJoin["Wolfram gridMathematica - sessionID:",
                            ToString[$SessionID], "-", ToString[$ModuleNumber]];

          res = iSetOptions[job, "JobOptions" /. opts /. {"JobOptions" -> {}},
                              "Name" -> name,
                              "MaximumNumberOfCores" -> Last[kcount],
                              "MinimumNumberOfCores" -> Last[kcount]];
          (
            res = scheduler@AddJob[job];
            job /; (res === Null)

          ) /; (res =!= $Failed)

        ) /; FreeQ[job, $Failed]

    ]

createJob["HTTP", scheduler_,  version_, kcount_, opts_] :=
    Block[{res, uri, body, job, name},

        uri = iStringJoin[scheduler, "Jobs"];
        (
          name = StringJoin["Wolfram gridMathematica - sessionID:",
                            ToString[$SessionID], "-", ToString[$ModuleNumber]];

          body = imakeBody["JobOptions" /. opts /. {"JobOptions" -> {}},
                           "Name" -> name,
                           "MinCores" -> Last[kcount],
                           "MaxCores" -> Last[kcount]];
          (
            res = URLFetch[uri, "Method" -> "POST", "Headers" -> $HTTPHeaders, "BodyData" -> body, "VerifyPeer" -> False,
                           "Username" -> OptionValue[opts, "Username"], "Password" -> OptionValue[opts, "Password"]];
            (
               job = StringCases[res, "<int xmlns=\"http://schemas.microsoft.com/2003/10/Serialization/\">" ~~ (x : DigitCharacter ..) ~~ "</int>" :> x];
               First[job] /; (ListQ[job] && (job =!= {}))

            ) /; StringQ[res]

          ) /; (body =!= $Failed)

        ) /; FreeQ[job, $Failed]

    ]

createJob[___] := $Failed


(* :addTasks: *)

createTasks[method_, scheduler_, version_, job_, kcount_, opts_] :=
    Block[{res},
        res = Table[addTasks[method, scheduler, version, job, opts], {Last[kcount]}];
        res /; FreeQ[res, $Failed]
    ]

addTasks[".NET", scheduler_, 2, job_, opts_] :=
    Block[{res, cmdMK, optsMK, interfaceML, link, task,
           commandLine, name},

        optsMK = OptionValue[opts, "KernelOptions"];
        interfaceML = OptionValue[opts, "NetworkInterface"];

        cmdMK = "KernelProgram" /. opts;
        (
          cmdMK = "\"" <> cmdMK <> "\"";

          If[interfaceML === "",
             link = LinkCreate[
                LinkProtocol -> "TCPIP",
                LinkMode -> Listen
               ];
             commandLine = StringJoin[cmdMK, " ",
                StringReplace[optsMK, "`linkname`" -> link[[1]]]]
          ,
             link = LinkCreate[
                LinkProtocol -> "TCPIP",
                LinkHost -> interfaceML,
                LinkMode -> Listen
               ];
             commandLine = StringJoin[cmdMK, " ",
                StringReplace[optsMK <> " -LinkHost `linkhost`",
                    {
                      "`linkname`" -> link[[1]],
                      "`linkhost`" -> interfaceML
                    }
                ]]
          ];

          (
            task = job@CreateTask[];
            (
                res = iSetOptions[task, OptionValue[opts, "TaskOptions"],
                                 "Name" -> "MathKernel",
                                 "CommandLine" -> commandLine,
                                 "MaximumNumberOfProcessors" -> 1,
                                 "MinimumNumberOfProcessors" -> 1];
                (
                res = job@AddTask[task];
                {task, link} /; (res === Null)

              ) /; (res =!= $Failed)

            ) /; FreeQ[task, $Failed]

          ) /; Head[link] === LinkObject

        ) /; StringQ[cmdMK]

    ]

addTasks[".NET", scheduler_, version_, job_, opts_] :=
    Block[{res, cmdMK, optsMK, interfaceML, link, task,
           commandLine, name},

        optsMK = OptionValue[opts, "KernelOptions"];
        interfaceML = OptionValue[opts, "NetworkInterface"];

        cmdMK = "KernelProgram" /. opts;
        (
          cmdMK = "\"" <> cmdMK <> "\"";

          If[interfaceML === "",
             link = LinkCreate[
                LinkProtocol -> "TCPIP",
                LinkMode -> Listen
               ];
             commandLine = StringJoin[cmdMK, " ",
                StringReplace[optsMK, "`linkname`" -> link[[1]]]]
          ,
             link = LinkCreate[
                LinkProtocol -> "TCPIP",
                LinkHost -> interfaceML,
                LinkMode -> Listen
               ];
             commandLine = StringJoin[cmdMK, " ",
                StringReplace[optsMK <> " -LinkHost `linkhost`",
                    {
                      "`linkname`" -> link[[1]],
                      "`linkhost`" -> interfaceML
                    }
                ]]
          ];

          (
            task = job@CreateTask[];
            (
                res = iSetOptions[task, OptionValue[opts, "TaskOptions"],
                                 "Name" -> "MathKernel",
                                 "CommandLine" -> commandLine,
                                 "MaximumNumberOfCores" -> 1,
                                 "MinimumNumberOfCores" -> 1];
                (
                res = job@AddTask[task];
                {task, link} /; (res === Null)

              ) /; (res =!= $Failed)

            ) /; FreeQ[task, $Failed]

          ) /; Head[link] === LinkObject

        ) /; StringQ[cmdMK]

    ]

addTasks["HTTP", scheduler_, version_, job_, opts_] :=
    Block[{res, cmdMK, optsMK, interfaceML, link, task, uri,
           commandLine, name},

        optsMK = OptionValue[opts, "KernelOptions"];
        interfaceML = OptionValue[opts, "NetworkInterface"];

        cmdMK = "KernelProgram" /. opts;
        (
          cmdMK = "\"" <> cmdMK <> "\"";

          If[interfaceML === "",
             link = LinkCreate[
                LinkProtocol -> "TCPIP",
                LinkMode -> Listen
               ];
             commandLine = StringJoin[cmdMK, " ",
                StringReplace[optsMK, "`linkname`" -> link[[1]]]]
          ,
             link = LinkCreate[
                LinkProtocol -> "TCPIP",
                LinkHost -> interfaceML,
                LinkMode -> Listen
               ];
             commandLine = StringJoin[cmdMK, " ",
                StringReplace[optsMK <> " -LinkHost `linkhost`",
                    {
                      "`linkname`" -> link[[1]],
                      "`linkhost`" -> interfaceML
                    }
                ]]
          ];

          (
            uri = iStringJoin[scheduler, "Job", job, "Tasks"];
            (
                body = imakeBody[OptionValue[opts, "TaskOptions"],
                                "Name" -> "MathKernel",
                                "CommandLine" -> commandLine,
                                "MinCores" -> 1,
                                "MaxCores" -> 1];
                (
                res = URLFetch[uri, "Method" -> "POST", "Headers" -> $HTTPHeaders, "BodyData" -> body, "VerifyPeer" -> False,
                               "Username" -> OptionValue[opts, "Username"], "Password" -> OptionValue[opts, "Password"]];
                (
                   task = StringCases[res, "<int xmlns=\"http://schemas.microsoft.com/2003/10/Serialization/\">" ~~ (x : DigitCharacter ..) ~~ "</int>" :> x];
                   {First[task], link} /; (ListQ[task] && (task =!= {}))

                ) /; StringQ[res]

              ) /; (res =!= $Failed)

            ) /; FreeQ[task, $Failed]

          ) /; Head[link] === LinkObject

        ) /; StringQ[cmdMK]

    ]

addTasks[___] := $Failed


(* :submitJob: *)

submitJob[".NET", scheduler_, job_, opts_] :=
    Block[{res, username, password, window},

        username = OptionValue[opts, "Username"];
        password = OptionValue[opts, "Password"];

        window = NETNew["System.IntPtr"];
        scheduler@SetInterfaceMode[False, window];
        (
          res = scheduler@SubmitJob[job, username, password];
          res /; (res === Null)

        ) /; ((username === Null) || StringQ[username]) &&
             ((password === Null) || StringQ[password])

    ]

submitJob["HTTP", scheduler_, job_, opts_] :=
    Block[{res, uri, body},

        uri = iStringJoin[scheduler, "Job", job, "Submit"];
        (
          body = imakeBody["FailOnTaskFailure" -> "true"];
           (
            res = URLFetch[uri, "StatusCode", "Method" -> "POST", "Headers" -> $HTTPHeaders, "BodyData" -> body,
                           "VerifyPeer" -> False, "Username" -> OptionValue[opts, "Username"], "Password" -> OptionValue[opts, "Password"]];

            Null /; (res === 200)

          ) /; StringQ[body]

        ) /; StringQ[uri]

    ]

submitJob[___] := $Failed


(* :waitingResources: *)

$PauseTime = 0.5

waitingResources[method_, scheduler_, job_, opts_] :=
    Block[{res, notRunningQ, queueQ, status},

        notRunningQ = True;
        queueQ = TrueQ[OptionValue[opts, "ToQueue"]];
        pausetime = 2;

        While[queueQ && notRunningQ,
            status = getJobStatus[method, scheduler, job, opts];
            If[!FreeQ[status, $Failed] || status === "Running",
               res = status; notRunningQ = False,
               Pause[pausetime]
            ]
        ];

        res /; FreeQ[res, $Failed]

    ]

waitingResources[___] := $Failed


(* :getJobStatus: *)

getJobStatus[".NET", scheduler_, job_, opts_] :=
    Block[{res},
          job@Refresh[];
          res = job@State;
          res = res@ToString[];
          res /; Head[res] === String
    ]


getJobStatus["HTTP", scheduler_, job_, opts_] :=
    Block[{res, uri},

        uri = iStringJoin[scheduler, "Job", job];
        (
           res = URLFetch[uri, "Parameters" -> {"Properties" -> "State"}, "VerifyPeer" -> False,
                          "Username" -> OptionValue[opts, "Username"], "Password" -> OptionValue[opts, "Password"]];
           (
              res = StringCases[res, __ ~~ "<Property><Name>State</Name><Value>" ~~ x__ ~~ "</Value></Property>" :> x];
              First[res] /; (ListQ[res] && (res =!= {}))

           ) /; StringQ[res]

        ) /; StringQ[uri]

    ]


getJobStatus[___] := $Failed


(* :cancelJob: *)

cancelJob[scheduler_, job_, tasks_] :=
    Quiet[
        scheduler@CancelJob[job@Id, "Failed to start. Cancelled by Wolfram Mathematica."];
        ReleaseNETObject[scheduler];
        ReleaseNETObject[job];
        If[ListQ[tasks],
           ReleaseNETObject /@ tasks[[All, 1]];
           LinkClose /@ tasks[[All, 2]]
        ];
    ]

cancelJob[___] := Null


(* :iFilterRules: *)

iFilterRules[a___] :=
    Block[{res, opts, c},

        opts = DeleteDuplicates[Flatten[{a, $HPCOptions}], (First[#1] === First[#2]) &];

        res = Complement[opts[[All, 1]], $HPCOptions[[All, 1]]];
        If[res === {},
            opts
           ,
            Message[OptionValue::optnf, res, HPC];
            FilterRules[opts, $HPCOptions]
        ]
    ]


(* :iLoadNETAssembly: *)

iLoadNETAssembly[assembly_, ""] := LoadNETAssembly[assembly]

iLoadNETAssembly[assembly_, dir_] := LoadNETAssembly[assembly, dir]

iLoadNETAssembly[___] := $Failed


(* :iSetOptions: *)

iSetOptions[obj_, opts___] :=
    Block[{res},
        res = Check[Scan[(obj @ Evaluate[ToExpression[First[#]]] = Last[#]) &, Flatten[{opts}]], $Failed];
        res /; (res =!= $Failed)
    ]

iSetOptions[___] := $Failed


(* :getVersion: *)

getVersion[".NET", scheduler_, opts_] :=
    Block[{res},
        res = scheduler@GetServerVersion[]@Major[];
        res /; IntegerQ[res]
    ]

getVersion["HTTP", scheduler_, opts_] := 3

getVersion[___] := $Failed


(* :getEnginePath: *)

getEnginePath[".NET", Automatic] :=
    Block[{path},

        path = Environment["CCP_SDK"];
        If[StringQ[path], path = FileNameJoin[{path, "Bin"}]];
        If[DirectoryQ[path], Return[path]];

        path = FileNames["C:\\Program Files\\Microsoft HPC Pack 2008*SDK\\Bin"];
        If[Length[path] > 0, path = First[path]];
        If[DirectoryQ[path], Return[path]];

        ""
    ]

getEnginePath[".NET", dir_?DirectoryQ] := dir

getEnginePath["HTTP", Automatic] :=
    Block[{res},

        res

    ]

getEnginePath[___] := $Failed


(* :getConnection: *)

getConnection[cluster_, opts_] /;
StringMatchQ[cluster, ("http:" | "https:") ~~__] := "HTTP"

getConnection[cluster_, opts_] /; (Head[OptionValue[opts, "EnginePath"]] === String) := ".NET"

getConnection[cluster_, opts_] /; (OptionValue[opts, "EnginePath"] === Automatic) := ".NET"

getConnection[___] := $Failed


(* :imakeBody: *)

imakeBody[opts___] :=
    Block[{props, body},

        props = makeProperty[Flatten[{opts}]];
        (
           body = StringReplace[$HTTPBody, "`body`" -> props];
           body /; StringQ[body]

        ) /; StringQ[props]

    ]

imakeBody[___] := $Failed

makeProperty[a_ -> b_] :=
    StringJoin["<Property>\n\t<Name>", ToString[a], "</Name>\n",
                 "\t<Value>", ToString[b], "</Value>\n</Property>"]

makeProperty[a_List] := StringJoin[makeProperty /@ a]


(* *)

iStringJoin[a__] :=
    With[{res = StringReplace[Flatten[{a}], (StartOfString ~~ "/") | (Whitespace ~~ "/") -> ""]},
        StringJoin[Riffle[res, "/"]]
    ]

iStringJoin[___] := $Failed

(* ************************************************************************* **

   :HPCCloseJob:

   - Comments:

** ************************************************************************* *)


HPCCloseJob[{jobID_, scheduler_, links_}, ___?OptionQ] :=
    Quiet[
        scheduler@FinishJob[jobID, "Closed by Wolfram Mathematica."];
        ReleaseNETObject[scheduler];
        LinkClose /@ links;
    ]


HPCCloseJob[___] := $Failed


(* ************************************************************************* **

   :HPCLoginScript:

   - Comments:

** ************************************************************************* *)


HPCLoginScript[name_, args_, dir_] :=
    Block[{cmd, kernel},

        cmd = dir <> "Scripts\\" <> "HPC.cmd";
        (
           kernel = "KernelProgram" /. args;
           (
             StringJoin["\"", cmd, "\"", " ", "\"", kernel, "\"", " ",
               "\"", "-mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName",
               "\"", " ", "\"`linkname`\"", " ", "\"", name, "\""]

           ) /; StringQ[kernel]

        ) /; (Length[FileNames[cmd]] == 1)

    ]


HPCLoginScript[___] := $Failed



(* ************************************************************************* *)


End[]


EndPackage[]

