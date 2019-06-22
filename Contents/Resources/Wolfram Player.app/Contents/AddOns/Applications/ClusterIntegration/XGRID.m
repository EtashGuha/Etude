(* :Name: XGRID Library File *)

(* :Title: Integration of Mathematica into Apple Xgrid clusters *)

(* :Context: ClusterIntegration`XGRID` *)

(* :Author: Charles Pooh *)

(* :Summary:

    This package provides functionalities for integrating Mathematica
    into Apple Xgrid clusters.

*)

(* :Copyright: 2008 Wolfram Research, Inc. *)

(* :Sources: *)


(* :Mathematica Version: 7.0 *)

(* :History: *)

(* :Keywords: None *)

(* :Warnings: None *)

(* :Limitations: *)

(* :Discussion: *)

(* :Requirements: *)

(* :Examples: None *)


(*****************************************************************************)


BeginPackage["ClusterIntegration`XGRID`", "ClusterIntegration`"]


(* Usage *)

XGRIDLaunchComputeKernels::usage =
"Internal function of ClusterIntegration. Use LaunchKernels to \
launch remote kernels on Xgrid clusters."

XGRIDLaunchMathJobs::usage =
"Internal function of ClusterIntegration. Use to launch math jobs \
on remote Xgrid clusters."

XGRIDCloseKernels::usage =
"Internal function of ClusterIntegration. Use CloseKernels to \
close compute kernels on Xgrid clusters."

XGRIDEngine::usage =
"Internal function of ClusterIntegration. Use to find and set \
the default Xgrid cluster."


(* ************************************************************************* *)


Begin["`Private`"]


Options[XGRID] =
    Sort[{
      "EnginePath" -> "/usr/bin/",
      "KernelProgram" -> "/Applications/Mathematica.app/Contents/MacOS/MathKernel",
      "KernelOptions" -> "-mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName _linkname_",
      "NetworkInterface" -> "",
      "Authentication" -> "-auth Password",
      "Identifier" -> 0,
      "Timeout" -> 2
    }]


$XGRIDDefinitions = Join[Options[XGRID],
    {
        "ClusterName" -> "",
        "ToQueue" -> False
    }]


Needs["ClusterIntegration`Library`"]


(* ************************************************************************* **

                           XGRIDLaunchComputeKernels

   Comments:

   ToDo:

** ************************************************************************* *)


XGRIDLaunchComputeKernels[nodes_Integer, opts___?OptionQ] :=
    Block[{res, res1, command, cluster, links, jobID},

        DebugPrint[" Launching remote kernels ...",
                   "\n Number of requested nodes: ", nodes,
                   "\n Cluster Type: Xgrid",
                   "\n Cluster info: ", TableForm[$XGRIDDefinitions],
                   "\n Options: ", TableForm[Flatten[{opts}]]
        ];

        command = initCommand[opts];
        (
            res = CheckAbort[

                    Catch[

                        cluster = connectCluster[command, opts];
                        If[!FreeQ[cluster, $Failed], Throw[$Failed]];
                        DebugPrint["1. Connected to cluster."];

                        links = createLinkObjects[nodes, opts];
                        If[!FreeQ[links, $Failed], Throw[$Failed]];
                        DebugPrint["2. Links created."];

                        jobID = submitJob[command, links, nodes, opts];
                        If[!FreeQ[jobID, $Failed], Throw[$Failed]];
                        DebugPrint["3. Jobs submitted."];

                        res1 = waitingResources[command, jobID, opts];
                        If[!FreeQ[res1, $Failed], Throw[$Failed]];
                        DebugPrint["4. Jobs running"];

                        Throw[Null]

                    ],

                    cancelJob[command, jobID, links, opts];
                    Abort[]

            ];

            AbortProtect[
                If[res === $Failed, cancelJob[command, jobID, links, opts]]];

            (
              res = Join[Complement[$XGRIDDefinitions, Flatten[{opts}],
                    SameTest -> (First[#1] === First[#2] &)], Flatten[{opts}]];

              {links, cluster, res, 0, jobID, Range[0, nodes - 1], Null}

            ) /; FreeQ[res, $Failed]


        ) /; StringQ[command]

    ]


XGRIDLaunchComputeKernels[___] := $Failed


(* :initCommand: *)

initCommand[opts___] :=
    Block[{res, id, dir, name, auth},

        id = "Identifier" /. Flatten[{opts}] /. $XGRIDDefinitions;
        dir = "EnginePath" /. Flatten[{opts}] /. $XGRIDDefinitions;
        name = "ClusterName" /. Flatten[{opts}] /. $XGRIDDefinitions;
        auth = "Authentication" /. Flatten[{opts}] /. $XGRIDDefinitions;

        ( dir <> "xgrid -h " <> name <> " " <> auth <> " -gid " <> ToString[id]
        ) /; IntegerQ[id] && StringQ[dir] && StringQ[name] && StringQ[auth]

    ]

initCommand[___] := $Failed


(* :connectCluster: *)

connectCluster[command_, opts___] :=
    Block[{res, status},

        status = RunCommand[command <> " " <> "-grid attributes"];
        (
          res = Flatten @ StringCases[status,
                  __ ~~ "name = " ~~ x:(Except[";"] ..) ~~ ";" ~~ __ :> x];
          (
            $XGRIDDefinitions = $XGRIDDefinitions /.
              {("ClusterName" -> _) -> ("ClusterName" -> First[res])};
            res

          ) /; (res =!= {}) && (Length[res] == 1)

        ) /; FreeQ[status, $Failed]

    ]

connectCluster[___] := (Message[XGRID::load, XGRID]; $Failed)


(* :createLinkObjects: *)

createLinkObjects[nodes_, opts___] :=
    Block[{interfaceML, links},

        interfaceML = "NetworkInterface" /. Flatten[{opts}] /. $XGRIDDefinitions;

        links = If[interfaceML === "",
            Table[LinkCreate[LinkProtocol -> "TCPIP",
                    LinkMode -> Listen], {nodes}],
            Table[LinkCreate[LinkProtocol -> "TCPIP",
                    LinkHost -> interfaceML, LinkMode -> Listen], {nodes}]
        ];

        links /; VectorQ[links, Head[#] === LinkObject &]

    ]

createLinkObjects[___] := $Failed


(* :submitJob: *)

submitJob[command_, links_, nodes_, opts___] :=
    Block[{res, id, kernel, options, pfile, prop, listJobs},

        kernel = "KernelProgram" /. Flatten[{opts}] /. $XGRIDDefinitions;
        options = "KernelOptions" /. Flatten[{opts}] /. $XGRIDDefinitions;
        (
          res = StringReplace[tdata, "_kerneloptions_" -> "(\"" <> options <> "\")"];
          res = StringReplace[res, {"_taskid_" -> ToString[# - 1],
                  "_linkname_" -> First[links[[#]]],
                  "_kernelprogram_" -> kernel}] & /@ Range[nodes];
          res = StringJoin @@ Riffle[res, "\n"];
          res = StringReplace[pdata, "_taskspecs_" -> res];
          (
            pfile = $TemporaryPrefix <> "batch.plist";
            res = Export[pfile, res, "Text"];
            (
              res = RunCommand[command <> " " <> "-job batch " <> pfile];
              (* DeleteFile[pfile]; *)
              (
                res = StringCases[res, __ ~~ "jobIdentifier = " ~~
                                     x:(DigitCharacter ..) ~~ __ :> x];
                  First[res] /; (Length[res] === 1)

              ) /; (res =!= $Failed)

            ) /; (res =!= $Failed)

          ) /; StringQ[res]

       ) /; StringQ[options]

    ]

submitJob[___] := $Failed


(* :property file: *)

pdata = "{
    jobSpecification = {
        applicationIdentifier = \"com.apple.xgrid.cli\";
        inputFiles =  {};
        name = \"Wolfram Mathematica\";
        submissionIdentifier = abc;
        taskSpecifications = {
            _taskspecs_
        };
    };
}"

tdata = "kernel_taskid_ = {arguments = _kerneloptions_; command = \"_kernelprogram_\";};"


(* :waitingResources: *)

$pauseTime = 0.5

waitingResources[command_, jobID_, opts___] :=
    Block[{res, notRunningQ, queueQ, status},

        notRunningQ = True;
        queueQ = "ToQueue" /. Flatten[{opts}] /. $XGRIDDefinitions;

        Pause[$pauseTime];

        While[queueQ && notRunningQ,
            status = getJobStatus[command, jobID, opts];
            If[(res === "Running"),
               res = status; notRunningQ = False]
        ];

        res /; (res === "Running")

    ]

waitingResources[___] := $Failed


(* :getJobStatus: *)

getJobStatus[command_, jobID_, opts___?OptionQ] :=
    Block[{res, status},

        status = RunCommand[command <> " " <> "-job attributes -id " <> ToString[jobID]];
        (
          res = Flatten @ StringCases[status,
                __ ~~ "jobStatus = " ~~  x:(Except[";"] ..) ~~ ";" ~~ __ :> x];

          First[res] /; (res =!= {}) && (Length[res] == 1)

        ) /; FreeQ[status, $Failed]

    ]

getJobStatus[___] := $Failed


(* :cancelJob: *)

cancelJob[command_, jobID_, links_, opts___] :=
    Quiet[
        LinkClose /@ links;
        RunCommand[command <> " -job delete - id " <> ToString[jobID]];
    ]

cancelJob[___] := $Failed


(* ************************************************************************* **

                                    XGRIDEngine

   XGRIDEngine[opts] is the discovering interface for Xgrid clusters.

   Comments:

   ToDo:

** ************************************************************************* *)


XGRIDEngine[___] := $Failed


(* ************************************************************************* **

                            XGRIDCloseKernels

   Comments:

   ToDo:

** ************************************************************************* *)


XGRIDCloseKernels[___] := $Failed


(* ************************************************************************* **

                               XGRIDLoginScript

   Comments:

   ToDo:

** ************************************************************************* *)


XGRIDLoginScript[name_, args_, dir_] :=
    Block[{cmd, kernel},

        cmd = dir <> "Scripts\\" <> "XGRID.sh";
        (
           kernel = "KernelProgram" /. args;
           (
             StringJoin["\"", cmd, "\"", " ", "-xgrid_opts \"",
                 " ", kernel, "\"", " ", "-linkname \"", name, "\""]

           ) /; StringQ[kernel]

        ) /; (Length[FileNames[cmd]] == 1)

    ]


XGRIDLoginScript[___] := $Failed


(* ************************************************************************* *)


End[]


EndPackage[]

