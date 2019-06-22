(* :Name: PBS Library File *)

(* :Title: Integration of gridMathematica into PBS *)

(* :Context: ClusterIntegration`PBS` *)

(* :Author: Charles Pooh *)

(* :Summary:

    This package provides functionalities for integrating gridMathematica
    into Altair PBS Pro.

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


BeginPackage["ClusterIntegration`PBS`", "ClusterIntegration`"]


(* Usage *)

PBSNewComputeKernels::usage =
"Internal function of ClusterIntegration. Use ClusterLaunchSlaves to \
launch remote kernels."

PBSEngine::usage =
"Internal function of ClusterIntegration. Use ClusterSetEngine to set \
the default engine."

PBSCloseJob::usage =
"Internal function of ClusterIntegration. Use ClusterCloseSlaves to \
close compute nodes."


(* ************************************************************************* *)


Begin["`Private`"]


Options[PBS] =
    Sort[{
        "EnginePath" -> If[Head[#] === String, #, "/usr/pbs"] & @ Environment["PBS_EXEC"],
        "KernelProgram" -> If[Head[#] === String, #, "/usr/local/Wolfram/Mathematica/7.0/Executables/math"] & @ StringReplace[$CommandLine, {"-noinit" -> "", "-mathlink" -> ""}],
        "KernelOptions" -> "-subkernel -mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName `linkname`",
        "Duration" -> "4:00:00",
        "NetworkInterface" -> ""
    }]


$PBSDefinitions = Join[Options[PBS],
    {
        "BatchCommand" -> Automatic,
        "QueueName" -> Automatic,
        "ReservationCommand" -> Automatic,
        "StartingTime" -> Automatic,
        "ToQueue" -> False,
        "ClusterName" -> ""
    }]


Needs["ClusterIntegration`Library`"]


(* ************************************************************************* **

   :PBSLaunchRemoteKernels:

   - Comments:

** ************************************************************************* *)


PBSNewComputeKernels[numNodes_Integer, opts___?OptionQ] :=
    Block[{res, res1, clustername, jobID, computeMLinks, tasksID},

        DebugPrint[" Launching remote kernels ...",
                   "\n Number of requested nodes: ", numNodes,
                   "\n Cluster Type: PBS Professional",
                   "\n Cluster info: ", TableForm[$PBSDefinitions],
                   "\n Options: ", TableForm[Flatten[{opts}]]
        ];

        res = CheckAbort[

                Catch[

                    clustername = connectCluster[opts];
                    If[!FreeQ[clustername, $Failed], Throw[$Failed]];
                    DebugPrint["1. Connected to " <> clustername];

                    jobID = reserveJob[numNodes, opts];
                    If[!FreeQ[jobID, $Failed], Throw[$Failed]];
                    DebugPrint["2. Job reservation: " <> ToString[jobID]];

                    computeMLinks = createLinkObjects[numNodes, opts];
                    If[!FreeQ[computeMLinks, $Failed], Throw[$Failed]];
                    DebugPrint["3. Links created"];

                    tasksID = submitJob[jobID, computeMLinks, numNodes, opts];
                    If[!FreeQ[tasksID, $Failed], Throw[$Failed]];
                    DebugPrint["4. Jobs submitted: ", ToString[tasksID]];

                    res1 = waitingResources[tasksID, opts];
                    If[!FreeQ[res1, $Failed], Throw[$Failed]];
                    DebugPrint["5. Jobs running"];

                ],

                cancelJob[jobID, tasksID, computeMLinks, opts];
                DebugPrint["Aborted - Job cancelled."];
                Abort[]

        ];

        AbortProtect[
            If[!FreeQ[res, $Failed], cancelJob[jobID, tasksID, computeMLinks, opts]]];

		({computeMLinks, "ClusterName" /. Flatten[{opts}] /. $PBSDefinitions, Flatten[{opts}], 0, jobID, tasksID, ""}
        ) /; FreeQ[res, $Failed]

    ]


PBSNewComputeKernels[___] := $Failed


(* :connectCluster: *)

connectCluster[opts___] :=
    Block[{res, dir, status},

        dir = "EnginePath" /. Flatten[{opts}] /. $PBSDefinitions;
        (
          status = RunCommand[dir <> "/bin/qstat -q"];
          (
            res = StringCases[status,
                        __ ~~ "server: " ~~ x:(WordCharacter ..) ~~ ___ :> x];
            (
              $PBSDefinitions = $PBSDefinitions /.
                 {("ClusterName" -> _) -> ("ClusterName" -> First[res])};
              First[res]

            ) /; (Length[res] == 1)

          ) /; FreeQ[status, $Failed]

        ) /; (Head[dir] === String)

    ]


connectCluster[___] :=
    (Message[ClusterEngine::load, "PBS", PBS::usage]; $Failed)


(* :reserveJob: *)

reserveJob[numNodes_, opts___] /;
(("QueueName" /. Flatten[{opts}] /. $PBSDefinitions) === Automatic) :=
    Block[{res, res1, res2, dir, cmd, starttime, duration},

        dir = "EnginePath" /. Flatten[{opts}] /. $PBSDefinitions;

        cmd = "ReservationCommand" /. Flatten[{opts}] /. $PBSDefinitions;
        (
          cmd = If[cmd === Automatic,
                   dir <> "/bin/pbs_rsub -N gridMathematica -R `starttime` -D `duration` -l select=`nodes`",
                   dir <> "/bin/pbs_rsub " <> cmd];

          starttime = "StartingTime" /. Flatten[{opts}] /. $PBSDefinitions;
          (
            If[starttime === Automatic,
               starttime = DateString[DatePlus[{3, "Second"}], {"Month", "Day", "Hour", "Minute", ".", "Second"}]];

            duration = "Duration" /. Flatten[{opts}] /. $PBSDefinitions;
            (

              res2 = StringReplace[cmd, {
                         "`starttime`" -> starttime,
                         "`duration`" -> duration,
                         "`nodes`" -> ToString[numNodes]}];

              res1 = RunCommand[res2];
              (
                res = StringCases[res1,
                        x:WordCharacter .. ~~ "." ~~ y: WordCharacter .. ~~ Whitespace ~~ __ :> x <> "@" <> y] ;

                First[res] /; (Length[res] == 1)

              ) /; FreeQ[res1, $Failed]

            ) /; StringQ[duration]

          ) /; (starttime === Automatic) || StringQ[cmd]

        ) /; (cmd === Automatic) || StringQ[cmd]

    ]


reserveJob[numNodes_, opts___] :=
    Block[{res},
        res = "QueueName" /. Flatten[{opts}] /. $PBSDefinitions;
        res /; (res =!= Automatic) && (StringQ[res] || res === None)
    ]


reserveJob[___] :=
    (Message[ClusterEngine::resv, "PBS", PBS::usage]; $Failed)


(* :createLinkObjects: *)

createLinkObjects[numNodes_, opts___] :=
    Block[{interfaceML, links},

        interfaceML = "NetworkInterface" /. Flatten[{opts}] /. $PBSDefinitions;

        links = If[interfaceML === Automatic,
            Table[
                LinkCreate[LinkProtocol -> "TCPIP", LinkMode -> Listen],
                {numNodes}],
            Table[
                LinkCreate[LinkProtocol -> "TCPIP", LinkHost -> interfaceML,
                LinkMode -> Listen], {numNodes}]
        ];

        links /; VectorQ[links, Head[#] === LinkObject &]

    ]


createLinkObjects[___] := $Failed


(* :submitJob: *)

submitJob[jobID_, computeMLinks_, numNodes_, opts___] :=
    Block[{res, dir, cmd, cmdMK, nSpec, optsMK, optsMKLink,
           job, file, taskID, listJobs, jobFile, script},

        dir = "EnginePath" /. Flatten[{opts}] /. $PBSDefinitions;

        cmdMK = "KernelProgram" /. Flatten[{opts}] /. $PBSDefinitions;
        optsMK = "KernelOptions" /. Flatten[{opts}] /. $PBSDefinitions;

        nSpec = "BatchCommand" /. Flatten[{opts}] /. $PBSDefinitions;

        script = Which[
            nSpec === Automatic && jobID === None,
                "#!/bin/sh\n#PBS -N gridM-`gridMjobID`\n#PBS -e /dev/null\n#PBS -o /dev/null\n#PBS -c n\n`mathkernel`",
            nSpec === Automatic && jobID =!= None,
                "#!/bin/sh\n#PBS -N gridM-`gridMjobID`\n#PBS -e /dev/null\n#PBS -o /dev/null\n#PBS -q `queue`\n#PBS -c n\n`mathkernel`",
            True,
                nSpec];

        file = $TemporaryPrefix <> "gridM" <> ToString[$SessionID] <> ToString[$ModuleNumber] <> ".pbs";

        res = Catch[

            listJobs = {};

            (
              optsMKLink = StringReplace[optsMK, "`linkname`" -> computeMLinks[[#, 1]]];

              cmd = StringReplace[script, {
                    "`queue`" -> jobID,
                    "`mathkernel`" -> StringJoin[cmdMK, " ", optsMKLink],
                    "`gridMjobID`" -> StringTake[ToString[$SessionID + $ModuleNumber], -9]}
              ];

              jobFile = Export[file, cmd, "Text"];

              If[jobFile =!= file, Throw[$Failed]];

              job = RunCommand[dir <> "/bin/qsub " <> jobFile];

              If[!FreeQ[job, $Failed] || Head[job] =!= String, Throw[$Failed]];

              taskID = StringCases[job, ((x:DigitCharacter ..) ~~ "." ~~ WordCharacter ..) :> x];

              If[Length[taskID] != 1, Throw[$Failed]];

              AppendTo[listJobs, First[taskID]];

            ) & /@ Range[numNodes];

            Throw[listJobs];

        ];

        DeleteFile[file];

        res /; FreeQ[res, $Failed]

    ]


submitJob[___] := $Failed


(* :waitingResources: *)

waitingResources[tasksID_, opts___] :=
    Block[{res, notRunningQ, pausetime, queueQ, starttime, status},

        notRunningQ = True;

		pausetime = 2;
        queueQ = "ToQueue" /. Flatten[{opts}] /. $PBSDefinitions;
        starttime = "StartingTime" /. Flatten[{opts}] /. $PBSDefinitions;

        If[(starttime =!= Automatic) && (queueQ =!= Automatic), Message[ClusterEngine::queue, starttime]];
        queueQ = If[queueQ === Automatic, "True", queueQ];

        While[queueQ && notRunningQ,
            status = getJobStatus[#, opts] & /@ tasksID;
            If[!FreeQ[status, $Failed] || MatchQ[status, {"Running" ..}],
               res = status; notRunningQ = False,
               Pause[pausetime]
            ]
        ];

        res /; FreeQ[res, $Failed]

    ]


waitingResources[___] := $Failed


(* :runningJobsQ: *)

runningJobsQ[tasksID_, opts___] :=
    Block[{res},
        res = getJobStatus[#, opts] & /@ tasksID;
        True /; FreeQ[res, $Failed] && MatchQ[res, {"Running" ..}]
    ]


runningJobsQ[___] := False


(* :connectRemoteKernels: *)

connectRemoteKernels[jobID_, tasksID_, computeMLinks_, opts___] :=
    Block[{res, res1, timeout, i},

		timeout = 2;
        If[!runningJobsQ[tasksID, opts], Pause[timeout]];

        (
          res = {};

          Do[
             res1 = Parallel`Parallel`ConnectSlave[computeMLinks[[i]]];
             If[FreeQ[res1, $Failed],
                AppendTo[res, res1],
                res = $Failed; Break[]
             ];
            ,
             {i, Length[computeMLinks]}
          ];

          res /; FreeQ[res, $Failed]

        ) /; runningJobsQ[tasksID, opts]


    ]


connectRemoteKernels[___] :=
    (Message[ClusterEngine::connect, "PBS"]; $Failed)


(* :cancelJob: *)

cancelJob[jobID_, tasksID_, computeMLinks_, opts___?OptionQ] :=
    Block[{res, cmd, dir, queue},

        dir = "EnginePath" /. Flatten[{opts}] /. $PBSDefinitions;
        queue = "QueueName" /. Flatten[{opts}] /. $PBSDefinitions;

        Quiet[
            res = LinkClose /@ computeMLinks;
            RunCommand[dir <> "/bin/qdel " <> #] & /@ tasksID;
            If[queue === Automatic, RunCommand[dir <> "/bin/pbs_rdel " <> jobID]];
        ];

        res /; FreeQ[res, $Failed]

    ]


cancelJob[___] := $Failed


(* :getJobStatus: *)

getJobStatus[taskID_, opts___?OptionQ] :=
    Block[{res, dir, status},

        dir = "EnginePath" /. Flatten[{opts}] /. $PBSDefinitions;
        status = RunCommand[dir <> "/bin/qstat -f " <> taskID];
        (
          res = Flatten @ StringCases[status,
                     ___ ~~ "job_state =" ~~ Whitespace ~~
                     x:(WordCharacter..) ~~ ___ :> x];

          (First[res] /. $JobStatus) /; (res =!= {}) && (Length[res] == 1)

        ) /; FreeQ[status, $Failed]
    ]


getJobStatus[___] := $Failed


(* ************************************************************************* **

   :LSFEngine:

   - Comments:

** ************************************************************************* *)


PBSEngine[opts___?OptionQ] :=
    Block[{res},
        res = connectCluster[opts];
        ( {PBS, Join[Complement[$PBSDefinitions, Flatten[{opts}],
                    SameTest -> (First[#1] === First[#2] &)], Flatten[{opts}]]}
        ) /; FreeQ[res, $Failed]
    ]


PBSEngine[___] := $Failed


(* ************************************************************************* **

   :PBSCloseSlaves:

   - Comments:

** ************************************************************************* *)


PBSCloseJob[{jobID_, tasksID_, nodes_}, opts___?OptionQ] :=
    Block[{res},

        res = Close /@ nodes;
        cancelJob[jobID, tasksID, {}, opts];

        res /; FreeQ[res, $Failed]

    ]


PBSCloseJob[___] := $Failed


(* ************************************************************************* **

   Utility functions

   - Comments:

** ************************************************************************* *)


$JobStatus = {"R" -> "Running", "Q" -> "Queued", "E" -> "Exiting"}


(* ************************************************************************* *)


End[]


EndPackage[]

