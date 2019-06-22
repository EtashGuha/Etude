(* :Name: LSF Library File *)

(* :Title: Integration of gridMathematica into Resource Management Systems *)

(* :Context: ClusterIntegration`LSF` *)

(* :Author: Charles Pooh *)

(* :Summary:

    This package provides functionalities for integrating gridMathematica
    into Platform LSF.

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


BeginPackage["ClusterIntegration`LSF`", "ClusterIntegration`"]


(* Usage *)

LSFNewComputeKernels::usage =
"Internal function of ClusterIntegration. Use ClusterLaunchSlaves to \
launch remote kernels."

LSFEngine::usage =
"Internal function of ClusterIntegration. Use ClusterSetEngine to set \
the default engine."

LSFCloseJob::usage =
"Internal function of ClusterIntegration. Use ClusterCloseSlaves to \
close compute nodes."


(* ************************************************************************* *)


Begin["`Private`"]


Options[LSF] =
    Sort[{
      "EnginePath" -> If[Head[#] === String, #, "/opt/lsf/bin"] & @ Environment["LSF_BINDIR"],
      "KernelProgram" -> If[Head[#] === String, #, "/usr/local/Wolfram/Mathematica/7.0/Executables/math"] & @ StringReplace[$CommandLine, {"-noinit" -> "", "-mathlink" -> ""}],
      "KernelOptions" -> "-subkernel -mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName `linkname`",
      "NativeSpecification" -> "",
      "NetworkInterface" -> "",
      "ToQueue" -> False
    }]


Needs["ClusterIntegration`Library`"]


(* ************************************************************************* **

   :LSFLaunchRemoteKernels:

   - Comments:

** ************************************************************************* *)


LSFNewComputeKernels[numNodes_Integer, opts___?OptionQ] :=
    Block[{res, res1, session, jobTemplate, jobIDs, computeMLinks,
    	   $LSFDefinitions},

		$LSFDefinitions = Join[opts, {"ClusterName" -> "localhost"}, Options[LSF]];
    
        DebugPrint[" Launching remote kernels ...",
                   "\n Number of requested nodes: ", numNodes,
                   "\n Cluster Type: LSF",
                   "\n Cluster info: ", TableForm[Options[LSF]],
                   "\n Options: ", TableForm[Flatten[{opts}]]
        ];

        res = CheckAbort[

                Catch[

                    session = connectCluster[]; 
                    If[!FreeQ[session, $Failed], Throw[$Failed]];
                    DebugPrint["1. Connected to cluster."];

                    computeMLinks = createLinkObjects[numNodes];
                    If[!FreeQ[computeMLinks, $Failed], Throw[$Failed]];
                    DebugPrint["2. Links created."];

                    jobIDs = submitJob[computeMLinks, numNodes];
                    If[!FreeQ[jobIDs, $Failed], Throw[$Failed]];
                    DebugPrint["3. Jobs submitted."];

                    res1 = waitingResources[jobIDs];
                    If[!FreeQ[res1, $Failed], Throw[$Failed]];
                    DebugPrint["4. Jobs running"];

                ],

                cancelJob[session, computeMLinks, {}];
                DebugPrint["Aborted - Job cancelled."];
                Abort[]

        ];

        AbortProtect[
            If[res === $Failed, cancelJob[session, computeMLinks, {}, opts]]];

		( {computeMLinks, "ClusterName" /. $LSFDefinitions, $LSFDefinitions, 0, 0, jobIDs, session}
         ) /; FreeQ[res, $Failed]

    ]


LSFNewComputeKernels[___] := $Failed


(* :connectCluster: *)

connectCluster[opts___?OptionQ] :=
    Block[{res, dir, status},

        dir = "EnginePath" /. $LSFDefinitions;
        (
          status = RunCommand[dir <> "/lsid"]; 
          (
            res = Flatten @ StringCases[status,
                __ ~~ "My cluster name is " ~~ x:(WordCharacter ..) ~~ __ ~~
                  "My master name is " ~~ y:(WordCharacter ..) ~~ __ :> {x, y}]; 

            (
              $LSFDefinitions = $LSFDefinitions /.
                 {("ClusterName" -> _) -> ("ClusterName" -> First[res])};
              res

            ) /; (res =!= {}) && (Length[res] == 2)

          ) /; FreeQ[status, $Failed]

        ) /; Head[dir] === String

    ]


connectCluster[___] :=
    (Message[ClusterEngine::load, "LSF", LSF::usage]; $Failed)


(* :createLinkObjects: *)

createLinkObjects[numNodes_, opts___] :=
    Block[{interfaceML, links},

        interfaceML = "NetworkInterface" /. $LSFDefinitions;

        links = If[interfaceML === "",
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

submitJob[computeMLinks_, numNodes_, opts___] :=
    Block[{res, dir, cmd, cmdMK, nSpec, optsMK, optsMKLink,
           job, jobID, listJobs},

        dir = "EnginePath" /. $LSFDefinitions;
        cmdMK = "KernelProgram" /. $LSFDefinitions;
        nSpec = "NativeSpecification" /. $LSFDefinitions;
        optsMK = "KernelOptions" /. $LSFDefinitions;

        cmd = dir <> "/bsub " <> nSpec <>  " " <> cmdMK <> " ";

        res = Catch[

            listJobs = {};

            (
              optsMKLink = {" ", #} & /@
                StringReplace[optsMK, "`linkname`" -> computeMLinks[[#, 1]]];

              job = RunCommand[StringJoin @ Flatten[{cmd, optsMKLink}]];

              If[!FreeQ[job, $Failed] || Head[job] =!= String,
                 cancelJob[listJobs, computeMLinks, {}, opts] ; Throw[$Failed]];

              jobID = Flatten @ StringCases[job,
                          ___ ~~ "Job <" ~~ x:NumberString ~~ ">" ~~ ___ :> x];

              If[Length[jobID] != 1,
                 cancelJob[listJobs, computeMLinks, {}, opts] ; Throw[$Failed]];

              AppendTo[listJobs, ToExpression @@ jobID];

            ) & /@ Range[numNodes];

            Throw[listJobs];

        ];

        res /; FreeQ[res, $Failed]

    ]


submitJob[___] := $Failed


(* :waitingResources: *)

waitingResources[jobIDs_, opts___] :=
    Block[{res, notRunningQ, pausetime, queueQ, status},

        notRunningQ = True;

        queueQ = "ToQueue" /. $LSFDefinitions;
        pausetime = 2;

        While[queueQ && notRunningQ,
            status = getJobStatus[#, opts] & /@ jobIDs;
            If[!FreeQ[status, $Failed] || MatchQ[status, {"Running" ..}],
               res = status; notRunningQ = False,
               Pause[pausetime]
            ]
        ];

        res /; FreeQ[res, $Failed]

    ]


waitingResources[___] := $Failed


(* :runningJobsQ: *)

runningJobsQ[jobIDs_, opts___] :=
    Block[{res},
        res = getJobStatus[#, opts] & /@ jobIDs;
        True /; FreeQ[res, $Failed] && MatchQ[res, {"Running" ..}]
    ]


runningJobsQ[___] := False


(* :connectRemoteKernels: *)

connectRemoteKernels[jobIDs_, computeMLinks_, opts___] :=
    Block[{res, res1, timeout, i},

		timeout = 2;
        If[!runningJobsQ[jobIDs, opts], Pause[timeout]];

        If[!runningJobsQ[jobIDs, opts],
           cancelJob[jobIDs, computeMLinks, {}, opts]; Pause[timeout]];

        (
          res = {};

          Do[
             res1 = Parallel`Parallel`ConnectSlave[computeMLinks[[i]]];
             If[FreeQ[res1, $Failed],
                AppendTo[res, res1],
                AppendTo[res, $Failed]; Break[]
             ];
            ,
             {i, Length[computeMLinks]}
          ];

          If[!FreeQ[res, $Failed],
             cancelJob[jobIDs, computeMLinks, {}, opts]];

          res /; FreeQ[res, $Failed]

        ) /; runningJobsQ[jobIDs, opts]


    ]


connectRemoteKernels[___] :=
    (Message[ClusterEngine::connect, "LSF"]; $Failed)


(* :cancelJob: *)

cancelJob[jobIDs_List, computeMLinks_List, nodes_, opts___?OptionQ] :=
    Block[{res, cmd},

        cmd = "EnginePath" /. $LSFDefinitions;
        cmd = cmd <> "/bkill ";

        Quiet[
            res = Close /@ nodes;
            If[runningJobsQ[jobIDs, opts],
               RunCommand[cmd <> ToString[#]] & /@ jobIDs];
            LinkClose /@ computeMLinks;
        ];

        res /; FreeQ[res, $Failed]

    ]


cancelJob[___] := $Failed


(* :getJobStatus: *)

getJobStatus[jobID_Integer, opts___?OptionQ] :=
    Block[{res, dir, status},

        dir = "EnginePath" /. $LSFDefinitions;
        status = RunCommand[dir <> "/bjobs -l " <> ToString[jobID]];
        (
          res = Flatten @ StringCases[status,
                     ___ ~~ "Status" ~~ Whitespace ~~ "<"
             ~~ x:(WordCharacter..) ~~ ">" ~~ ___ :> x];

          (First[res] /. $jobStatus) /; (res =!= {}) && (Length[res] == 1)

        ) /; FreeQ[status, $Failed]
    ]


getJobStatus[___] := $Failed


$jobStatus = {"RUN" -> "Running", "SSUSP" -> "Suspended", "PEND" -> "Pending"}


(* ************************************************************************* **

   :LSFEngine:

   - Comments:

** ************************************************************************* *)


LSFEngine[opts___?OptionQ] :=
    Block[{res},
        res = connectCluster[opts];
        ( {LSF, Join[Complement[$LSFDefinitions, Flatten[{opts}],
                    SameTest -> (First[#1] === First[#2] &)], Flatten[{opts}]]}
        ) /; FreeQ[res, $Failed]
    ]


LSFEngine[___] := $Failed


(* ************************************************************************* **

   :LSFCloseSlaves:

   - Comments:

** ************************************************************************* *)


LSFCloseJob[{jobIDs_, computeMLinks_}, opts___?OptionQ] :=
    Block[{res},
        res = cancelJob[jobIDs, {}, computeMLinks, opts];
        res /; FreeQ[res, $Failed]
    ]


LSFCloseJob[___] := $Failed


(* ************************************************************************* *)


End[]


EndPackage[]

