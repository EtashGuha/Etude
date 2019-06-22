(* :Name: HPC Library File *)

(* :Title: Integration of gridMathematica into Resource Management Systems *)

(* :Context: ClusterIntegration`HPC` *)

(* :Author: Charles Pooh *)

(* :Summary:

    This package provides functionalities for integrating gridMathematica
    into Microsoft High Performance Computing Server 2008.

*)

(* :Copyright: 2006 - 2008 Wolfram Research, Inc. *)

(* :Sources: *)

(* :Package Version: 2.0 *)

(* :Mathematica Version: 7.0 *)

(* :History: *)

(* :Keywords: None *)

(* :Warnings: None *)

(* :Limitations: *)

(* :Discussion: *)

(* :Requirements: *)

(* :Examples: None *)


(*****************************************************************************)


BeginPackage["ClusterIntegration`HPCWin`",
             {"ClusterIntegration`", "ClusterIntegration`HPC`", "NETLink`"}]


Begin["`Private`"]

(* options *)

Options[HPC] =
    Sort[{
		"EnginePath" -> With[{res = Environment["CCP_SDK"]},
                If[StringQ[res], res <> "\\bin",
                   "C:\\Program Files\\Microsoft HPC Pack 2008 SDK\\Bin"]],    	
        "KernelProgram" -> "c:\\Program Files\\Wolfram Research\\Mathematica\\7.0\\MathKernel.exe",
        "KernelOptions" -> "-subkernel -mathlink -LinkProtocol TCPIP -LinkMode Connect -LinkName `linkname`",
        "NetworkInterface" -> "", 
        "ToQueue" -> False
    }]


$HPCDefinitions := Join[Options[HPC],
    {
        "MaximumNumberOfProcessors" -> Automatic,
        "MinimumNumberOfProcessors" -> Automatic,
        "TaskMaximumNumberOfProcessors" -> 1,
        "TaskMinimumNumberOfProcessors" -> 1,
        "ClusterName" -> "localhost",
        "UserName" -> Null,
        "Password" -> Null
    }]


(* ************************************************************************* **

                           HPCNewComputeKernels


   Comments:

   ToDo:

** ************************************************************************* *)


HPCNewComputeKernels[numNodes_Integer, opts___?OptionQ] :=
    Block[{res, res1, job, nopts, scheduler, tasks},

        DebugPrint[" Launching remote kernels ...",
                   "\n Number of requested nodes: ", numNodes,
                   "\n Cluster Type: HPC",
                   "\n Cluster info: ", TableForm[$HPCDefinitions],
                   "\n User options: ", TableForm[opts]
        ];

        nopts = iFilterRules[opts, $HPCDefinitions];

        res = CheckAbort[

            Catch[

                scheduler = connectScheduler[nopts];
                If[!FreeQ[scheduler, $Failed], Throw[$Failed]];
                DebugPrint["1. Connected to scheduler."];

                job = createJob[scheduler, numNodes, nopts];
                If[!FreeQ[job, $Failed], Throw[$Failed]];
                DebugPrint["2. Job created."];

                tasks = Table[addTasks[job, nopts], {numNodes}];
                If[!FreeQ[tasks, $Failed], Throw[$Failed]];
                DebugPrint["3. Tasks added."];

                res1 = submitJob[scheduler, job, nopts];
                If[!FreeQ[res1, $Failed], Throw[$Failed]];
                DebugPrint["4. Job submitted."];

                res1 = waitingResources[job, nopts];
                If[!FreeQ[res1, $Failed], Throw[$Failed]];
                DebugPrint["5. Job running."];

            ],

            cancelJob[scheduler, job, tasks];
            DebugPrint["Aborted - Job cancelled."];
            Abort[]

        ];

        AbortProtect[
            If[res === $Failed, cancelJob[scheduler, job, tasks]]
        ];

        (
          res = job@Id;
          res1 = Through[tasks[[All, 1]][TaskId]];
          ReleaseNETObject[job];
          ReleaseNETObject /@ tasks[[All, 1]];

          {tasks[[All, 2]], OptionValue[nopts, "ClusterName"],
           nopts, 0, res, res1, scheduler}

        ) /; FreeQ[res, $Failed]

    ]


HPCNewComputeKernels[___] := $Failed


(* :connectCluster: *)

connectScheduler[opts_] :=
    Block[{res, scheduler, dir, name},

        name = "ClusterName" /. opts;
        dir = "EnginePath" /. opts;
        (
          res = Needs["NETLink`"];
          (
            res = InstallNET[];
            (
              res = LoadNETAssembly["Microsoft.Hpc.Scheduler", dir];
              (
                scheduler = NETNew["Microsoft.Hpc.Scheduler.Scheduler"];
                (
                   res = scheduler@Connect[name];
                   scheduler /; FreeQ[res, $Failed]

                ) /; Head[scheduler] === Symbol

              ) /; VectorQ[Flatten[{res}], (Head[#] === NETAssembly) &]

            ) /; Head[res] === LinkObject

          ) /; FreeQ[res, $Failed]

        ) /; StringQ[name] && StringQ[dir]

    ]

connectScheduler[___] :=
    (Message[HPC::load, "HPC"]; $Failed)


(* :createJob: *)

createJob[scheduler_, numNodes_, opts_] :=
    Block[{res, job, jobID, runAlways, maxNumberNodes, minNumberNodes},

        maxNumberNodes = "MaximumNumberOfProcessors" /. opts;
        If[maxNumberNodes === Automatic, maxNumberNodes = numNodes];

        minNumberNodes = "MinimumNumberOfProcessors" /. opts;
        If[minNumberNodes === Automatic, minNumberNodes = numNodes];

        job = scheduler@CreateJob[];
        (
          job@Name = StringJoin["Wolfram gridMathematica - sessionID:",
                        ToString[$SessionID], "-", ToString[$ModuleNumber]];

          job@MaximumNumberOfProcessors = maxNumberNodes;
          job@MinimumNumberOfProcessors = minNumberNodes;

          res = scheduler@AddJob[job];
          job /; (res === Null)

        ) /; FreeQ[job, $Failed] &&
             IntegerQ[minNumberNodes] && IntegerQ[maxNumberNodes]

    ]

createJob[___] := $Failed


(* :addTasks: *)

addTasks[job_, opts_] :=
    Block[{res, cmdMK, optsMK, interfaceML, link, task, runAlways,
           commandLine, maxNumberProcessors, minNumberProcessors},

        optsMK = "KernelOptions" /. opts;
        interfaceML = "NetworkInterface" /. opts;

        maxNumberProcessors = "TaskMaximumNumberOfProcessors" /. opts;
        minNumberProcessors = "TaskMinimumNumberOfProcessors" /. opts;

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
              task@Name = "MathKernel";

              task@MaximumNumberOfProcessors = maxNumberProcessors;
              task@MinimumNumberOfProcessors = minNumberProcessors;

              task@CommandLine = commandLine;

              res = job@AddTask[task];
              {task, link} /; (res === Null)

            ) /; FreeQ[task, $Failed] &&
                 IntegerQ[maxNumberProcessors] && IntegerQ[minNumberProcessors]

          ) /; Head[link] === LinkObject

        ) /; StringQ[cmdMK]

    ]

addTasks[___] := $Failed


(* :submitJob: *)

submitJob[scheduler_, job_, opts_] :=
    Block[{res, username, password, window},

        username = "UserName" /. opts;
        password = "Password" /. opts;

        window = NETNew["System.IntPtr"];
        scheduler@SetInterfaceMode[False, window];
        (
          res = scheduler@SubmitJob[job, username, password];
          res /; (res === Null)

        ) /; ((username === Null) || StringQ[username]) &&
             ((password === Null) || StringQ[password])

    ]

submitJob[___] := $Failed


(* :waitingResources: *)

$PauseTime = 0.5

waitingResources[job_, opts_] :=
    Block[{res, notRunningQ, queueQ, status},

        notRunningQ = True;
        queueQ = TrueQ["ToQueue" /. opts];
        pausetime = 2;

        While[queueQ && notRunningQ,
            status = getJobStatus[job];
            If[!FreeQ[status, $Failed] || status === "Running",
               res = status; notRunningQ = False,
               Pause[pausetime]
            ]
        ];

        res /; FreeQ[res, $Failed]

    ]

waitingResources[___] := $Failed


(* :getJobStatus: *)

getJobStatus[job_] :=
    Block[{res},
          job@Refresh[];
          res = job@State;
          res = res@ToString[];
          res /; Head[res] === String
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

iFilterRules[a_] := a

iFilterRules[a__, b_] :=
    Block[{res, opts = Flatten[{a}]},

        res = Complement[opts[[All, 1]], b[[All, 1]]];
        If[res =!= {}, Message[OptionValue::optnf, res, HPC]];

        opts = FilterRules[opts, b];
        If[opts === {}, Return[b]];

        res = FilterRules[b, Except[Alternatives @@ opts[[All, 1]]]];
        Join[opts, res]

    ]

iFilterRules[____] := {}


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

