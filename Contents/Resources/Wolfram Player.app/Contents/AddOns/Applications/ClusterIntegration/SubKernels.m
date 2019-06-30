(* :Name: SubKernels.m *)

(* :Title: SubKernels' implementation *)

(* :Context: ClusterIntegration`SubKernels` *)

(* :Author: Charles Pooh *)

(* :Summary: This package provides subkernels implementation for CIP *)

(* :Copyright: (c) 1986-2012 Wolfram Research, Inc. *)

(* :Sources: *)

(* :Package Version: 2.0 *)

(* :Mathematica Version: 9.0 *)

(* :History: Last updated Sept 11, 2008 *)

(* :Keywords: None *)

(* :Warnings: None *)

(* :Limitations: *)

(* :Discussion: *)

(* :Requirements: *)

(* :Examples: None *)


(*****************************************************************************)


BeginPackage["ClusterIntegration`SubKernels`",  {"ClusterIntegration`", "SubKernels`"}]


Unprotect[ComputeKernel, ComputeKernelObject]


(* subkernel and class object *)

ComputeKernel::usage = "ComputeKernel[...] is a compute subkernel on a cluster or grid."

ComputeKernelObject::usage = "ComputeKernelObject[...] is the cluster kernels class object."


(* initialization *)

Needs["SubKernels`Protected`"]

ComputeKernelObject[subContext] = "ClusterIntegration`"


Begin["`Private`"]


(* local variables and functions *)

`$openkernels = {}

`configuration


(* ************************************************************************* **

                        Compute Clusters

   Comments:

   ToDo:

** ************************************************************************* *)


(* :Compute Clusters: *)

CCS /: NewKernels[CCS, eopts___?OptionQ] := NewComputeKernels[CCS["localhost"], 1, eopts]
CCS /: NewKernels[CCS[], eopts___?OptionQ] := NewComputeKernels[CCS["localhost"], 1, eopts]
CCS /: NewKernels[CCS[cluster_String, args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[CCS[cluster], args, opts, eopts]
CCS /: NewKernels[CCS[cluster_String, eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[CCS[cluster], 1, opts, eopts]
CCS /: NewKernels[CCS[args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[CCS["localhost"], args, opts, eopts]

HPC /: NewKernels[HPC, eopts___?OptionQ] := NewComputeKernels[HPC["localhost"], 1, eopts]
HPC /: NewKernels[HPC[], eopts___?OptionQ] := NewComputeKernels[HPC["localhost"], 1, eopts]
HPC /: NewKernels[HPC[cluster_String, args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[HPC[cluster], args, opts, eopts]
HPC /: NewKernels[HPC[cluster_String, eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[HPC[cluster], 1, opts, eopts]
HPC /: NewKernels[HPC[args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[HPC["localhost"], args, opts, eopts]

LSF /: NewKernels[LSF, eopts___?OptionQ] := NewComputeKernels[LSF["localhost"], 1, eopts]
LSF /: NewKernels[LSF[], eopts___?OptionQ] := NewComputeKernels[LSF["localhost"], 1, eopts]
LSF /: NewKernels[LSF[cluster_String, args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[LSF[cluster], args, opts, eopts]
LSF /: NewKernels[LSF[cluster_String, eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[LSF[cluster], 1, opts, eopts]
LSF /: NewKernels[LSF[args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[LSF["localhost"], args, opts, eopts]

PBS /: NewKernels[PBS, eopts___?OptionQ] := NewComputeKernels[PBS["localhost"], 1, eopts]
PBS /: NewKernels[PBS[], eopts___?OptionQ] := NewComputeKernels[PBS["localhost"], 1, eopts]
PBS /: NewKernels[PBS[cluster_String, args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[PBS[cluster], args, opts, eopts]
PBS /: NewKernels[PBS[cluster_String, eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[PBS[cluster], 1, opts, eopts]
PBS /: NewKernels[PBS[args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[PBS["localhost"], args, opts, eopts]

SGE /: NewKernels[SGE, eopts___?OptionQ] := NewComputeKernels[SGE["localhost"], 1, eopts]
SGE /: NewKernels[SGE[], eopts___?OptionQ] := NewComputeKernels[SGE["localhost"], 1, eopts]
SGE /: NewKernels[SGE[cluster_String, args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[SGE[cluster], args, opts, eopts]
SGE /: NewKernels[SGE[cluster_String, eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[SGE[cluster], 1, opts, eopts]
SGE /: NewKernels[SGE[args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[SGE["localhost"], args, opts, eopts]

XGRID /: NewKernels[XGRID, eopts___?OptionQ] := NewComputeKernels[XGRID["localhost"], 1, eopts]
XGRID /: NewKernels[XGRID[], eopts___?OptionQ] := NewComputeKernels[XGRID["localhost"], 1, eopts]
XGRID /: NewKernels[XGRID[cluster_String, args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[XGRID[cluster], args, opts, eopts]
XGRID /: NewKernels[XGRID[cluster_String, eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[XGRID[cluster], 1, opts, eopts]
XGRID /: NewKernels[XGRID[args:(_List | _Integer), eopts___?OptionQ], opts___?OptionQ] := NewComputeKernels[XGRID["localhost"], args, opts, eopts]


(* :NewComputeKernels: *)

NewComputeKernels[engine_[cluster_String], numNodes_, opts___?OptionQ] /; validKernelNumbersQ[numNodes] :=
    Block[{res, nopts, kcount},

        kcount = iKernelCount[numNodes];
        (
            nopts = Flatten[{"ClusterName" -> cluster, opts}];

            res = Switch[engine,
              CCS,
               ClusterIntegration`CCS`CCSNewComputeKernels[numNodes, nopts],
              HPC,
               ClusterIntegration`HPC`HPCNewComputeKernels[cluster, kcount, opts],
              LSF,
               ClusterIntegration`LSF`LSFNewComputeKernels[numNodes, nopts],
              PBS,
               ClusterIntegration`PBS`PBSNewComputeKernels[numNodes, nopts],
              SGE,
               ClusterIntegration`SGE`SGENewComputeKernels[numNodes, nopts],
              XGRID,
               ClusterIntegration`XGRID`XGRIDNewComputeKernels[numNodes, nopts],
              _,
               $Failed
            ];
            (
              res = initLink[Sequence @@ res, cluster, engine];
              If[numNodes === {1, 1}, First[res], res] /; (res =!= $Failed)

            ) /; (res =!= $Failed)

        ) /; (kcount =!= $Failed)

    ]


NewComputeKernels[___] := $Failed


(* destructor *)

ComputeKernel /: Close[kernel_ComputeKernel?subQ] := (
    $openkernels = DeleteCases[$openkernels, kernel];
    kernelClose[kernel, True]
)


(* raw constructor *)

initLink[links_List, descr_, args_, sp_, jobID_, taskIDs_List, obj_,
cluster_, engine_] :=
    With[{res = Transpose[{links, taskIDs}]},
         initLink[#1, descr, args, sp, jobID, #2, obj, cluster, engine] & @@@ res]


initLink[link:Except[_List], descr_, args_,  sp_, jobID_,
taskID:Except[_List], obj_, cluster_, engine_] :=
    Module[{kernel, speed = sp},
        kernel = ComputeKernel[lk[link, descr, args, speed, cluster, engine, jobID, taskID, obj]];
        AppendTo[$openkernels, kernel];
        kernelInit[kernel]
     ]


initLink[___] := $Failed


(* :validKernelNumbersQ: *)

validKernelNumbersQ[_Integer?NonNegative | {_Integer?NonNegative}] := True

validKernelNumbersQ[{n_Integer?NonNegative, m_Integer}] /; NonNegative[m - n] := True

validKernelNumbersQ[___] := False


(* :iKernelCount: *)

iKernelCount[n_Integer?NonNegative | {n_Integer?NonNegative}] := {n, n}

iKernelCount[{n_Integer?NonNegative, m_Integer}] /; NonNegative[m - n] := {n, m}

iKernelCount[___] := $Failed


(* ************************************************************************* **

                            ComputeKernel

   Comments:

      ComputeKernel[lk[link, descr, arglist, speed, cluster, jobID, taskID, obj] ]
         link       associated LinkObject
         descr      head node name
         arglist    list of arguments used in constructor
         speed      speed
         cluster    cluster name
         engine     engine name
         jobID      job identifier
         taskID     task identifier
         obj        cluster engine object or stream

   ToDo:

** ************************************************************************* *)


SetAttributes[ComputeKernel, HoldAll]
SetAttributes[`lk, HoldAllComplete]

ComputeKernel /: linkObject[ComputeKernel[lk[link_, ___], ___]] := link
ComputeKernel /: descr[ComputeKernel[lk[link_, descr_, ___], ___]] := descr
ComputeKernel /: arglist[ComputeKernel[lk[link_, descr_, arglist_, ___], ___]] := arglist
ComputeKernel /: kernelSpeed[ComputeKernel[lk[link_, descr_, arglist_, speed_, ___], ___]] := speed
ComputeKernel /: setSpeed[ComputeKernel[lk[link_, descr_, arglist_, speed_, ___], ___], r_] := (speed = r)

ComputeKernel /: cluster[ComputeKernel[lk[link_, descr_, arglist_, speed_, cluster_, ___], ___]] := cluster
ComputeKernel /: engine[ComputeKernel[lk[link_, descr_, arglist_, speed_, cluster_, engine_,  ___], ___]] := engine
ComputeKernel /: jobID[ComputeKernel[lk[link_, descr_, arglist_, speed_, cluster_, engine_, jobID_, ___], ___]] := jobID
ComputeKernel /: taskID[ComputeKernel[lk[link_, descr_, arglist_, speed_, cluster_, engine_, jobID_, taskID_, ___], ___]] := taskID
ComputeKernel /: clusterObject[ComputeKernel[lk[link_, descr_, arglist_, speed_, cluster_, engine_, jobID_, taskID_, cobj_, ___], ___]] := cobj


ComputeKernel /:  subQ[ComputeKernel[lk[link_, descr_, arglist_, ___] ] ] := Head[link]===LinkObject
ComputeKernel /:  LinkObject[kernel_ComputeKernel ]  := linkObject[kernel]
ComputeKernel /:  MachineName[kernel_ComputeKernel ] := descr[kernel]
ComputeKernel /:  Description[kernel_ComputeKernel ] := cluster[kernel][arglist[kernel]]
ComputeKernel /:  Abort[kernel_ComputeKernel ] := kernelAbort[kernel]
ComputeKernel /:  SubKernelType[kernel_ComputeKernel ] := ComputeKernelObject
ComputeKernel /:  Clone[kernel_ComputeKernel] := NewKernel[Description[kernel], KernelSpeed->kernelSpeed[kernel]]
ComputeKernel /:  KernelCount[kernel_ComputeKernel] := 1

ComputeKernel /:  ClusterName[kernel_ComputeKernel ] := cluster[kernel]
ComputeKernel /:  EngineName[kernel_ComputeKernel ] := engine[kernel]
ComputeKernel /:  JobID[kernel_ComputeKernel ] := jobID[kernel]
ComputeKernel /:  TaskID[kernel_ComputeKernel ] := taskID[kernel]


(* format *)

SubKernels`Private`setFormat[ComputeKernel, "ComputeKernel"]


(* ************************************************************************* **

                        Configuration module

   Comments:

   ToDo:

** ************************************************************************* *)


configuration[configQ] = True
configuration[nameConfig] = ComputeKernelObject[subName]


configuration[setConfig] := configuration[setConfig, {}]

configuration[setConfig, data_] :=
    ClusterIntegration`Palette`CIPSetConfiguration[data]


configuration[getConfig] :=
    Block[{res},
        res = ClusterIntegration`Palette`CIPGetConfiguration[];
        res /; (res =!= $Failed)
    ]


configuration[useConfig] :=
    Block[{res},
        res = ClusterIntegration`Palette`CIPConfiguredClusters[];
        res /; (res =!= $Failed)
    ]


configuration[tabConfig] :=
    Block[{res},
        res = ClusterIntegration`Palette`CIPPalette[];
        res /; (Head[res] === Panel)
    ]


(* ************************************************************************* **

                        Compute Kernel Object

   Comments:

   ToDo:

** ************************************************************************* *)


ComputeKernelObject[subKernels] := $openkernels

ComputeKernelObject[subName] = "compute kernel"

ComputeKernelObject[subConfigure] = configuration


(* handling short forms of cluster description *)

ComputeKernelObject[try][s_String] :=
    Block[{res},

        res = Select[$ConfiguredKernels, MatchQ[#,
            ClusterIntegration`Palette`ComputeKernels[_, _, _?Positive, _, {s, __}]] &];
        NewKernels[First[res]] /; (res =!= {})

    ]


(* registration *)

addImplementation[ComputeKernelObject]


(* ************************************************************************* *)


End[]


SetAttributes[
    {ComputeKernel, ComputeKernelObject},
    {ReadProtected, Protected}
]


EndPackage[]
