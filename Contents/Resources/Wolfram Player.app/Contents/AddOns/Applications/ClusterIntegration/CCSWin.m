(* :Name: CCS Windows Library File *)

(* :Title: Integration with Microsoft Compute Cluster Server 2003 *)

(* :Context: ClusterIntegration`CCS` *)

(* :Author: Charles Pooh *)

(* :Summary:

    This package provides functionalities for integrating Mathematica
    with Microsoft Compute Cluster Server 2003.

*)

(* :Copyright: (c) 2006 - 2008 Wolfram Research, Inc. *)

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


BeginPackage["ClusterIntegration`CCSWin`",
             {"ClusterIntegration`", "ClusterIntegration`CCS`", "NETLink`"}]



Begin["`Private`"]


(* options *)

Options[CCS] =
    Sort[{
        "EnginePath" -> If[StringQ[#], # <> "\\bin", "c:\\Program Files\\Microsoft Compute Cluster Pack\\bin"] & @ Environment["CCP_HOME"],
        "KernelProgram" -> "c:\\Program Files\\Wolfram Research\\Mathematica\\7.0\\MathKernel.exe",
        "KernelOptions" -> "-subkernel -mathlink -LinkProtocol TCPIP -LinkMode Connect -LinkName `linkname`",
        "NetworkInterface" -> "",
        "ToQueue" -> False
    }]

$CCSDefinitions := Join[Options[CCS],
    {
        "Console" -> False,
        "RunTime" -> "Infinite",
        "MaximumNumberOfProcessors" -> Automatic,
        "MinimumNumberOfProcessors" -> Automatic,
        "TaskMaximumNumberOfProcessors" -> 1,
        "TaskMinimumNumberOfProcessors" -> 1,
        "ClusterName" -> "localhost",
        "UserName" -> "",
        "Password" -> "",
        "SharedDirectory" -> "\\\\clusterboss.wolfram.com\\TempStore\\"
    }]


(* loading libraries *)

Needs["ClusterIntegration`Library`"]


(* parameters *)

$CCPAPIDLL = "\\ccpapi.dll"

$PauseTime = 0.5


(* ************************************************************************* **

                         CCS New Compute Kernels

   Comments:

   ToDo:

** ************************************************************************* *)


CCSNewComputeKernels[numNodes_Integer, opts___?OptionQ] :=
    Block[{res, res1, res2, cluster, jobID, taskIDs, computeML},

        DebugPrint[" Launching remote kernels ...",
                   "\n Number of requested nodes: ", numNodes,
                   "\n Cluster Type: CCS",
                   "\n Cluster info: ", TableForm[$CCSDefinitions],
                   "\n Options: ", TableForm[opts]
        ];

        res = CheckAbort[

            Catch[

                cluster = connectCluster[opts];
                If[!FreeQ[cluster, $Failed], Throw[$Failed]];
                DebugPrint["1. Connected to cluster."];

                jobID = createJob[cluster, numNodes, opts];
                If[!FreeQ[jobID, $Failed], Throw[$Failed]];
                DebugPrint["2. Job created."];

                res1 = Table[addTasks[cluster, jobID, opts], {numNodes}];
                If[!FreeQ[res1, $Failed], Throw[$Failed]];
                taskIDs = res1[[All, 1]]; computeML = res1[[All, 2]];
                DebugPrint["3. Tasks added."];

                res1 = submitJob[cluster, jobID, opts];
                If[!FreeQ[res1, $Failed], Throw[$Failed]];
                DebugPrint["4. Job submitted."];

                res1 = waitingResources[cluster, jobID, opts];
                If[!FreeQ[res1, $Failed], Throw[$Failed]];
                DebugPrint["5. Job running"];

            ],

            cancelJob[cluster, jobID, computeML];
            Abort[]

        ];

        AbortProtect[
            If[res === $Failed, cancelJob[cluster, jobID, computeML]]];

        (
          res1 = cluster@Name;
          res1 = If[StringQ[res1], res1,
                    "ClusterName" /. Flatten[{opts}] /. $CCSDefinitions];

          res2 = Union[Flatten[{opts, $CCSDefinitions}],
                       SameTest -> (#[[1]] === #2[[1]] &)];

          {computeML, res1, res2, 0, jobID, taskIDs, cluster}

        ) /; FreeQ[res, $Failed]

    ]


CCSNewComputeKernels[___] := $Failed


(* :connectCluster: *)

connectCluster[opts___] :=
    Block[{res, cluster, name, ccpapi},

        name = "ClusterName" /. Flatten[{opts}] /. $CCSDefinitions;
        ccpapi = "EnginePath" /. Flatten[{opts}] /. $CCSDefinitions;
        (
          ccpapi = ccpapi <> $CCPAPIDLL;

          res = Needs["NETLink`"];
          (
            res = InstallNET[];
            (
              res = LoadNETAssembly[ccpapi];
              (
                cluster = NETNew["Microsoft.ComputeCluster.Cluster"];
                (
                   res = cluster@Connect[name];
                   cluster /; FreeQ[res, $Failed]

                ) /; Head[cluster] === Symbol

              ) /; Head[res] === NETAssembly

            ) /; Head[res] === LinkObject

          ) /; FreeQ[res, $Failed]

        ) /; Head[name] === String && Head[ccpapi] === String

    ]


connectCluster[___] :=
    (Message[ClusterEngine::load, CCS]; $Failed)


(* :createJob: *)

createJob[cluster_, numNodes_, opts___] :=
    Block[{job, jobID, runAlways, runTime, maxNumberNodes, minNumberNodes},

        runTime = "RunTime" /. Flatten[{opts}] /. $CCSDefinitions;

        maxNumberNodes = "MaximumNumberOfProcessors" /. Flatten[{opts}] /. $CCSDefinitions;
        If[maxNumberNodes === Automatic, maxNumberNodes = numNodes];

        minNumberNodes = "MinimumNumberOfProcessors" /. Flatten[{opts}] /. $CCSDefinitions;
        If[minNumberNodes === Automatic, minNumberNodes = numNodes];

        job = cluster@CreateJob[];
        (
          job@Name = "Wolfram Mathematica";

          job@MaximumNumberOfProcessors = maxNumberNodes;
          job@MinimumNumberOfProcessors = minNumberNodes;

          job@Runtime = runTime;

          jobID = cluster@AddJob[job];
          jobID /; IntegerQ[jobID]

        ) /; FreeQ[job, $Failed]

    ]


createJob[___] := $Failed


(* :addTasks: *)

addTasks[cluster_, jobID_, opts___] :=
    Block[{cmdMK, optsMK, interfaceML, link, task, taskID, runAlways, runTime,
           commandLine, maxNumberProcessors, minNumberProcessors},

        cmdMK = "KernelProgram" /. Flatten[{opts}] /. $CCSDefinitions;
        cmdMK = "\"" <> cmdMK <> "\""; (* force to pass a string in case of names with space *)

        optsMK = "KernelOptions" /. Flatten[{opts}] /. $CCSDefinitions;
        interfaceML = "NetworkInterface" /. Flatten[{opts}] /. $CCSDefinitions;

        runTime = "RunTime" /. Flatten[{opts}] /. $CCSDefinitions;

        maxNumberProcessors = "TaskMaximumNumberOfProcessors" /. Flatten[{opts}] /. $CCSDefinitions;

        minNumberProcessors = "TaskMinimumNumberOfProcessors" /. Flatten[{opts}] /. $CCSDefinitions;

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
          task = cluster@CreateTask[];
          (
            task@Name = "MathKernel";

            task@MaximumNumberOfProcessors = maxNumberProcessors;
            task@MinimumNumberOfProcessors = minNumberProcessors;

            task@Runtime = runTime;

            task@CommandLine = commandLine;

            taskID = cluster@AddTask[jobID, task];
            {taskID, link} /; IntegerQ[taskID]

          ) /; FreeQ[task, $Failed]

        ) /; Head[link] === LinkObject

    ]


addTasks[___] := $Failed


(* :submitJob: *)

submitJob[cluster_, jobID_, opts___] :=
    Block[{res, username, password, console},

        username = "UserName" /. Flatten[{opts}] /. $CCSDefinitions;
        password = "Password" /. Flatten[{opts}] /. $CCSDefinitions;
        console = "Console" /. Flatten[{opts}] /. $CCSDefinitions;

        res = cluster@SubmitJob[jobID, username, password, console, 0];
        res /; FreeQ[res, $Failed]

    ]


submitJob[___] := $Failed


(* :waitingResources: *)

waitingResources[cluster_, jobID_, opts___] :=
    Block[{res, notRunningQ, queueQ, status},

        notRunningQ = True;

        queueQ = "ToQueue" /. Flatten[{opts}] /. $CCSDefinitions;

        While[queueQ && notRunningQ,
            status = getJobStatus[cluster, jobID];
            If[!FreeQ[status, $Failed] || status === "Running",
               res = status; notRunningQ = False,
               Pause[$PauseTime]
            ]
        ];

        res /; FreeQ[res, $Failed]

    ]


waitingResources[___] := $Failed


(* :getJobStatus: *)

getJobStatus[cluster_, jobID_] :=
    Block[{res, job, jobstatus},

        job = cluster@GetJob[jobID];
        (
          jobstatus = job@Status;
          res = jobstatus@ToString[];
          res /; Head[res] === String

        ) /; FreeQ[job, $Failed]

    ]


getJobStatus[___] := $Failed


(* :runningJobQ: *)

runningJobQ[cluster_, jobID_] :=
    getJobStatus[cluster, jobID] === "Running"


(* :cancelJob: *)

cancelJob[cluster_, job_, tasks_, nodes_:{}] :=
    Block[{res},

        Quiet[
            res = Close /@ nodes;
            cluster@CancelJob[job@Id, "Failed to start. Cancelled by Wolfram Mathematica."];
            ReleaseNETObject[cluster];
            LinkClose /@ tasks[[All, 2]];
        ];

        res /; FreeQ[res, $Failed]

    ]


cancelJob[___] := $Failed


(* ************************************************************************* **

                               CCS Front End Login Script

   Comments:

   ToDo:

** ************************************************************************* *)


CCSLoginScript[name_, args_, dir_] :=
    Block[{cmd, kernel},

        cmd = dir <> "Scripts\\" <> "CCSWin.cmd";
        (
           kernel = "KernelProgram" /. args;
           (
             StringJoin["\"", cmd, "\"", " ", "\"", kernel, "\"", " ",
                        "\"`linkname`\"", " ", "\"", name, "\""]

           ) /; StringQ[kernel]

        ) /; (Length[FileNames[cmd]] == 1)

    ]


CCSLoginScript[___] := $Failed


(* ************************************************************************* **

                                  CCS Math Batch Job

   Comments:

   ToDo:

** ************************************************************************* *)


CCSMathBatchJob[notebook_, numNodes_, opts___] :=
    Block[{res, scheduler, jobID, task, code},

        DebugPrint[" Launching batch job ...",
                   "\n Notebook: ", notebook,
                   "\n Number of requested nodes: ", numNodes,
                   "\n Cluster Type: CCS",
                   "\n Cluster info: ", TableForm[$CCSDefinitions],
                   "\n Options: ", TableForm[opts]
        ];

        res = Catch[

            scheduler = connectCluster[opts];
            If[!FreeQ[scheduler, $Failed], Throw[$Failed]];
            DebugPrint["1. Connected to cluster."];

            jobID = createJob[scheduler, numNodes, opts];
            If[!FreeQ[jobID, $Failed], Throw[$Failed]];
            DebugPrint["2. Job created."];

            task = addTask[scheduler, jobID, opts];
            If[!FreeQ[jobID, $Failed], Throw[$Failed]];
            DebugPrint["3. Task created."];

            code = createPackage[notebook, jobID, numNodes, opts];
            If[!FreeQ[code, $Failed], Throw[$Failed]];
            DebugPrint["4. Package created."];

            res = submitJob[scheduler, jobID, opts];
            If[!FreeQ[res, $Failed], Throw[$Failed]];
            DebugPrint["5. Job submitted."];

            Throw[jobID];

        ];

        res = cleanup[scheduler, res, jobID, opts];
        DebugPrint["6. Cleanup done."];

        res /; (res =!= $Failed)

    ]


CCSMathBatchJob[___] := $Failed


(* :createPackage: *)

createPackage[notebook_, jobID_, numNodes_, opts___] :=
    Block[{res, file, nopts, ucode},

        ucode = Import[notebook, "Text"];
        (
           ucode = StringReplace[$initCode <> ucode <> $exitCode,
                   {
                     "_numNodes_" -> ToString[numNodes],
                     "_jobID_" -> ToString[jobID],
                     "_schedulerName_" -> ("ClusterName" /. Flatten[{opts}] /. $CCSDefinitions),
                     "_kernelProgram_" -> ("KernelProgram" /. Flatten[{opts}] /. $CCSDefinitions),
                     "_kernelOptions_" -> ("KernelOptions" /. Flatten[{opts}] /. $CCSDefinitions)
                   }];

           file = ("SharedDirectory" /. Flatten[{opts}] /. $CCSDefinitions);
           (
             file = ToFileName[file, "mccs" <> ToString[jobID] <> ".m"];

             res = Export[file, ucode, "Text"];
             res /; (res =!= $Failed)

           ) /; (file =!= "")

        ) /; StringQ[ucode]

    ]


createPackage[___] := $Failed


$initCode = "
(* initialization code automatically generated by Mathematica *)

     Unprotect[$ProcessorCount];
     $ProcessorCount = _numNodes_ - 1;
     Protect[$ProcessorCount];

     Get[\"Parallel`\"];

     BeginPackage[\"ClusterIntegration`BatchJobs`\"]

     (* loading modules *)

     Needs[\"ClusterIntegration`Library`\"];
     Needs[\"SubKernels`LinkKernels`\"];

     Begin[\"`Private`\"]

         Block[{links, subkernels, tasks},

             links = Table[LinkCreate[LinkProtocol -> \"TCPIP\"], {$ProcessorCount}];

             tasks = \"job add _jobID_ /scheduler:_schedulerName_ \\\"_kernelProgram_\\\" _kernelOptions_\";

             tasks = StringReplace[tasks, \"`linkname`\" -> First[#]] & /@ links;

             Import[\"!\" <> #, \"Text\"] & /@ tasks; (* use RunCommand *)

             Parallel`Developer`ConnectKernel[ConnectLink[#]] & /@ links;

         ]

     End[]

     EndPackage[]

"


$exitCode = "

(* exit code automatically generated by Mathematica *)

      Parallel`CloseKernels[];
      Quit[];

"


(* :addTask: *)

addTask[cluster_, jobID_, opts___] :=
    Block[{file, kernel, options, task, taskID},

        kernel = "KernelProgram" /. Flatten[{opts}] /. $CCSDefinitions;
        kernel = "\"" <> kernel <> "\"";

        file = "SharedDirectory" /. Flatten[{opts}] /. $CCSDefinitions;
        file = StringReplace[ToFileName[file, "mccs" <> ToString[jobID] <> ".m"], "\\" -> "\\\\"];
        options = StringReplace["-run \"Get[\\\"_file_\\\"]\"", "_file_" -> file];

        task = cluster@CreateTask[];
        (
          task@Name = "MathKernel";

          task@MaximumNumberOfProcessors = 1;
          task@MinimumNumberOfProcessors = 1;

          task@CommandLine = kernel <> " " <> options;

          taskID = cluster@AddTask[jobID, task];
          taskID /; IntegerQ[taskID]

        ) /; FreeQ[task, $Failed]

    ]


addTask[___] := $Failed


(* :cleanup: *)

cleanup[scheduler_, $Failed, jobID_, opts___] :=
    Block[{res},

       res = "SharedDirectory" /. Flatten[{opts}] /. $CCSDefinitions;
       res = ToFileName[res, "mccs" <> ToString[jobID] <> ".m"];
       (* DeleteFile[res]; *)
       scheduler@CancelJob[jobID, "Failed to start. Cancelled by Wolfram Mathematica."];

       {jobID, "Failed to start"}

    ]


cleanup[_, Except[$Failed], jobID_, opts___] :=
    Block[{res},

       res = "SharedDirectory" /. Flatten[{opts}] /. $CCSDefinitions;
       res = ToFileName[res, "mccs" <> ToString[jobID] <> ".m"];
       (* DeleteFile[res]; *)

       res = Button[Style["Submitted", Blue, 8], showJobStatus[jobID, opts],
                   Appearance -> None, Method -> "Queued"];
       {jobID, res}

    ]


cleanup[____] := $Failed


(* ------------------------------------------------------------------------- *)

showJobStatus[jobID_, opts___] :=
    Block[{res, name},

        name = "ClusterName" /. Flatten[{opts}] /. $CCSDefinitions;
        res = Import["!job view /scheduler:" <> name <> " " <> ToString[jobID], "Text"];
        (
          res = StringSplit[#, ":", 2] & /@ StringSplit[res, "\n"];
          res = Map[StringReplace[#, (StartOfString ~~ Whitespace) |
              (Whitespace ~~ EndOfString) -> ""] &, res, {2}];

          MessageDialog[Grid[res, Alignment -> Left],
              WindowTitle -> "Job " <> ToString[jobID] <> " Status"]

        ) /; StringQ[res]

    ]


showJobStatus[jobID_, ___] := "Job ID: " <> ToString[jobID]


(* ************************************************************************* *)


End[]


EndPackage[]
