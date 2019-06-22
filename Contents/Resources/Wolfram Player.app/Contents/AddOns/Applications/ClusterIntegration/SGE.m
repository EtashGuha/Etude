(* :Name: SGE Library File *)

(* :Title: Integration of gridMathematica into Resource Management Systems *)

(* :Context: ClusterIntegration`SGE` *)

(* :Author: Charles Pooh *)

(* :Summary:

    This package provides functionalities for integrating gridMathematica
    into Sun Grid Engine with DRMAA Java binding.

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


BeginPackage["ClusterIntegration`SGE`", {"ClusterIntegration`", "JLink`"}]


(* Usage *)

SGENewComputeKernels::usage =
"Internal function. Use LaunchKernels to launch compute kernels."

SGEEngine::usage =
"Internal function of ClusterIntegration."

SGECloseJob::usage =
"Internal function. Use CloseKernels to close compute kernels."


(* ************************************************************************* *)


Begin["`Private`"]


Options[SGE] =
    Sort[{
        "EnginePath" -> If[Head[#] === String, #, "/usr/local/sge/sge_root"] & @ Environment["SGE_ROOT"],
        "KernelProgram" -> If[Head[#] === String, #, "/usr/local/Wolfram/Mathematica/7.0/Executables/math"] & @ StringReplace[$CommandLine, {"-noinit" -> "", "-mathlink" -> ""}],
        "KernelOptions" -> "-subkernel -mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName `linkname`",
        "NativeSpecification" -> "",
        "NetworkInterface" -> "",
        "ToQueue" -> False
    }]


(* ************************************************************************* **

   						      SGE New Remote Kernels

   Comments:
   
   ToDo:

** ************************************************************************* *)


`$SGEDefinitions = {}


SGENewComputeKernels[numNodes_Integer, opts___?OptionQ] :=
    Block[{res, res1, session, jobTemplate, jobIDs, computeMLinks, 
    	   $SGEDefinitions},

		$SGEDefinitions = Join[
			opts,
			{"ClusterName" -> If[Head[#] === String, #, "localhost"] & @ 
				Environment["SGE_CLUSTER_NAME"]},
			Options[SGE]
		];
		
        DebugPrint[" Launching remote kernels ...",
                   "\n Number of requested nodes: ", numNodes,
                   "\n Cluster Type: SGE (with DRMAA Java binding)",
                   "\n Cluster info: ", TableForm[Options[SGE]],
                   "\n Options: ", TableForm[Flatten[{opts}]]
        ];

        res = CheckAbort[

                Catch[

                    session = connectScheduler[];
                    If[!FreeQ[session, $Failed], Throw[$Failed]];
                    DebugPrint["1. Connected to cluster."];

                    jobTemplate = createJobTemplate[session, numNodes];
                    If[!FreeQ[jobTemplate, $Failed], Throw[$Failed]];
                    DebugPrint["2. Job template created."];

                    computeMLinks = createLinkObjects[numNodes];
                    If[!FreeQ[computeMLinks, $Failed], Throw[$Failed]];
                    DebugPrint["3. Links created."];

                    jobIDs = submitJob[session, jobTemplate, computeMLinks, numNodes];
                    If[!FreeQ[jobIDs, $Failed], Throw[$Failed]];
                    DebugPrint["4. Jobs submitted."];

                    res1 = waitingResources[session, jobIDs];
                    If[!FreeQ[res1, $Failed], Throw[$Failed]];
                    DebugPrint["5. Jobs running"];

                ],

                cancelJob[session, jobIDs, computeMLinks];
				DebugPrint["Aborted - Job cancelled."];
                Abort[]

        ];

        AbortProtect[
            If[res === $Failed, cancelJob[session, jobIDs, computeMLinks]]
		];

        If[FreeQ[session, $Failed],
            If[FreeQ[jobTemplate, $Failed],
               session@deleteJobTemplate[jobTemplate];
            ];
            session@exit[]
        ];

       ( {computeMLinks, "ClusterName" /. $SGEDefinitions, $SGEDefinitions, 0, 0, jobIDs, session}
       ) /; FreeQ[res, $Failed]

    ]


SGENewComputeKernels[___] := $Failed


(* :connectScheduler: *)

connectScheduler[opts___] :=
    Block[{res, session, sessionfactory, factory, classpath},

        classpath = SGEJavaLibs[$SGEDefinitions];
        (
          res = Needs["JLink`"];
          (
            res = InstallJava[];
            (
              AddToClassPath[classpath];

              sessionfactory = LoadJavaClass["org.ggf.drmaa.SessionFactory",
                                       AllowShortContext ->  True];
              (
                factory = SessionFactory`getFactory[];
                (
                  session = factory@getSession[];
                  (
                     session /; (session@init[Null] === Null)

                  ) /; FreeQ[session, $Failed]

                ) /; FreeQ[factory, $Failed]

              ) /; Head[sessionfactory] === JavaClass

            ) /; Head[res] === LinkObject

          ) /; FreeQ[res, $Failed]

        ) /; FreeQ[classpath, $Failed]

    ]


connectScheduler[___] :=
    (Message[SGE::load, "SGE"]; $Failed)


(* :createJobTemplate: *)

createJobTemplate[session_, numNodes_, opts___] :=
    Block[{res, job, cmdMK, nSpec},

        cmdMK = "KernelProgram"  /. $SGEDefinitions;
        nSpec = "NativeSpecification" /. $SGEDefinitions;

        job = session@createJobTemplate[];
        (
          res = Check[
                      job@setJobName["gridMathematica"];
                      job@setRemoteCommand[cmdMK];
                      If[nSpec =!= "", job@setNativeSpecification[nSpec]]
                      ,
                      $Failed
          ];

          job /; FreeQ[res, $Failed]

        ) /; FreeQ[job, $Failed]

    ]


createJobTemplate[___] := $Failed


(* :createLinkObjects: *)

createLinkObjects[numNodes_, opts___] :=
    Block[{interfaceML, links},

        interfaceML = "NetworkInterface" /. $SGEDefinitions;

        links = If[interfaceML === "",
            Table[
                LinkCreate[LinkProtocol -> "TCPIP", LinkMode -> Listen],
                {numNodes}],
            Table[
                LinkCreate[LinkProtocol -> "TCPIP", LinkHost -> interfaceML,
                LinkMode -> Listen], {numNodes}]
        ];

        links /; FreeQ[links, $Failed]

    ]


createLinkObjects[___] := $Failed


(* :submitJob: *)

submitJob[session_, jobTemplate_, computeMLinks_, numNodes_, opts___] :=
    Block[{res, optsMK, optsMKLinkName, jobID, i},

        optsMK = "KernelOptions" /. $SGEDefinitions;

        res = {};

        JavaBlock[

            optsMKLinkName = JavaNew["java.util.Vector", 7];

            Do[
                Map[optsMKLinkName@addElement[JavaNew["java.lang.String", #]] &,
                    StringSplit[optsMK] /.
                        "`linkname`" -> ToString[computeMLinks[[i, 1]]]];

                jobTemplate@setArgs[optsMKLinkName];
                jobID = session@runJob[jobTemplate];

                If[FreeQ[jobID, $Failed] && Head[jobID] == String,
                   AppendTo[res, jobID],
                   AppendTo[res, $Failed]; Break[]
                ]
              ,
                {i, numNodes}
            ];

        ];

        If[!FreeQ[res, $Failed],
         session@control[#, session@TERMINATE] & /@ Select[res, # =!= $Failed &];
        ];

        res /; FreeQ[res, $Failed]

    ]


submitJob[___] := $Failed


(* :waitingResources: *)

waitingResources[session_, jobIDs_, opts___] :=
    Block[{res, notRunningQ, queueQ, status, timeout},

        notRunningQ = True;

        queueQ = "ToQueue" /. $SGEDefinitions;
        timeout = 2;

        While[queueQ && notRunningQ,
            status = session@getJobProgramStatus[#] & /@ jobIDs;
            If[!FreeQ[status, $Failed] || MatchQ[status, {session@RUNNING ..}],
               res = status; notRunningQ = False,
               Pause[timeout]
            ]
        ];

        res /; FreeQ[res, $Failed]

    ]


waitingResources[___] := $Failed


(* :cancelJob: *)

cancelJob[session_, jobIDs_, computeMLinks_, nodes_:{}] :=
    Block[{res},

       Quiet[
            res = Close /@ nodes;
            session@control[#, session@TERMINATE] & /@ jobIDs;
            LinkClose /@ computeMLinks;
       ];

       res /; FreeQ[res, $Failed]

    ]


cancelJob[___] := $Failed


(* ************************************************************************* **

   							   SGE Engine

   Comments:
   
   ToDo:

** ************************************************************************* *)


SGEEngine[opts___?OptionQ] :=
    Block[{res, session, sessionfactory, factory, classpath},

        classpath = SGEJavaLibs[opts];
        (
          res = Needs["JLink`"];
          (
            res = InstallJava[];
            (
                AddToClassPath[Sequence @@ classpath];

              sessionfactory = LoadJavaClass["org.ggf.drmaa.SessionFactory",
                                    AllowShortContext ->  True];
              (
                factory = SessionFactory`getFactory[];
                (
                  session = factory@getSession[];
                  (
                    session@exit[];
                    ReleaseJavaObject[session];

                    { SGE,
                      Join[Complement[$SGEDefinitions, Flatten[{opts}],
                         SameTest -> (First[#1] === First[#2] &)],
                         Flatten[{opts}]
                      ]
                    }

                  ) /; FreeQ[session, $Failed] && (session@init[Null] === Null)

                ) /; FreeQ[factory, $Failed]

              ) /; Head[sessionfactory] === JavaClass

            ) /; Head[res] === LinkObject

          ) /; FreeQ[res, $Failed]

        ) /; FreeQ[classpath, $Failed]
    ]


SGEEngine[___] := $Failed


(* ************************************************************************* **

   							  SGE Close Job

   Comments:
   
   ToDo:

** ************************************************************************* *)


SGECloseJob[{jobIDs_, computeMLinks_}, opts___?OptionQ] :=
    Block[{res, session},

        Quiet[
            session = connectCluster[opts];
            res = If[FreeQ[session, $Failed],
                     cancelJob[session, jobIDs, {}, computeMLinks],
                     Close /@ computeMLinks];
            session@exit[];
            ReleaseJavaObject[session];
        ];

        res /; FreeQ[res, $Failed]

    ]


SGECloseJob[___] := $Failed


(* ************************************************************************* **

   							Internal functions

   Comments:
   
   SGEJavaLibs[opts] gives the path to Java library in SGE
   
   ToDo:

** ************************************************************************* *)


(* :SGEJavaLibs: *)

SGEJavaLibs[opts___?OptionQ] :=
    Block[{res, libpath},

        libpath = "EnginePath" /. Flatten[{opts}] /. Options[SGE];
        (
          res = Quiet[FileNames["*.jar", libpath <> "/lib", Infinity]];
          Union[DirectoryName /@ res] /; res =!= {} && Head[res] === List

        ) /; Head[libpath] === String

    ]


SGEJavaLibs[___] := $Failed


(* ************************************************************************* *)


End[]


EndPackage[]

