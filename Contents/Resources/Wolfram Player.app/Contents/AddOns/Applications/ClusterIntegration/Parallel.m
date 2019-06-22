(* :Name: Parallel.m *)

(* :Title: Extension of parallel functions *)

(* :Context: ClusterIntegration`Parallel` *)

(* :Author: Charles Pooh *)

(* :Summary: This package provides internal tools for CIP *)

(* :Copyright: (c) 2008 Wolfram Research, Inc. *)

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


BeginPackage["ClusterIntegration`Parallel`", "ClusterIntegration`"]


Begin["`Private`"]


(* declaration for Parallel functions *)

Parallel`CloseKernels
Parallel`Kernels
Parallel`Developer`SubKernel

$ContextPath = Join[$ContextPath, {"Parallel`", "Parallel`Developer`"}]


(* ************************************************************************* **

                         Parallel functions extensions

   Comments:

   ToDo:

** ************************************************************************* *)


(* CCS *)

CCS /: CloseKernels[CCS, opts___] := closeKernels[CCS[], opts]
CCS /: CloseKernels[CCS[args___], opts___] := closeKernels[CCS[args], opts]


(* HPC *)

HPC /: CloseKernels[HPC, opts___] := closeKernels[HPC[], opts]
HPC /: CloseKernels[HPC[args___], opts___] := closeKernels[HPC[args], opts]


(* LSF *)

LSF /: CloseKernels[LSF, opts___] := closeKernels[LSF[], opts]
LSF /: CloseKernels[LSF[args___], opts___] := closeKernels[LSF[args], opts]


(* PBS *)

PBS /: CloseKernels[PBS, opts___] := closeKernels[PBS[], opts]
PBS /: CloseKernels[PBS[args___], opts___] := closeKernels[PBS[args], opts]


(* SGE *)

SGE /: CloseKernels[SGE, opts___] := closeKernels[SGE[], opts]
SGE /: CloseKernels[SGE[args___], opts___] := closeKernels[SGE[args], opts]


(* XGRID *)

XGRID /: CloseKernels[XGRID, opts___] := closeKernels[XGRID[], opts]
XGRID /: CloseKernels[XGRID[args___], opts___] := closeKernels[XGRID[args], opts]


(* :closeKernels: *)

closeKernels[engine_[args_:Automatic], opts___?OptionQ] :=
    Block[{kernels},
        kernels = Select[Kernels[], subComputeKernelQ[#, engine, args, opts] &];
        CloseKernels[kernels] /; ListQ[kernels]
    ]


closeKernels[___] := $Failed


(* :subComputeKernelQ: *)

subComputeKernelQ[kernel_, engine_, args_, ___] :=
    With[{res = SubKernel[kernel]},
         (EngineName[res] === engine) &&
         ((ClusterName[res] === args) || (args === Automatic))]


subComputeKernelQ[___] := False


(* ------------------------------------------------------------------------- *)


EngineName[kernel_] :=
    Block[{res},
        res = SubKernel[kernel];
        (
          res =  EngineName[res];
          res /; Head[res] =!= EngineName

        ) /; Head[res] =!= SubKernel
    ]


ClusterName[kernel_] :=
    Block[{res},
        res = SubKernel[kernel];
        (
          res =  ClusterName[res];
          res /; Head[res] =!= ClusterName

        ) /; Head[res] =!= SubKernel
    ]


JobID[kernel_] :=
    Block[{res},
        res = SubKernel[kernel];
        (
          res =  JobID[res];
          res /; Head[res] =!= JobID

        ) /; Head[res] =!= SubKernel
    ]


TaskID[kernel_] :=
    Block[{res},
        res = SubKernel[kernel];
        (
          res =  TaskID[res];
          res /; Head[res] =!= TaskID

        ) /; Head[res] =!= SubKernel
    ]


(* ************************************************************************* *)


End[]


EndPackage[]