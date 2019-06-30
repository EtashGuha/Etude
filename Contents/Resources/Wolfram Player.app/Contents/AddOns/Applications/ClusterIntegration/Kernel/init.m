(* :Name: Kernel Init File *)

(* :Title: Mathematica Initialization File for Cluster Integration Package *)

(* :Context: ClusterIntegration` *)

(* :Author: Charles Pooh *)

(* :Summary: Initialization module for the Cluster Integration Package *)

(* :Copyright: (c) 2006 - 2008 Wolfram Research, Inc. *)

(* :Sources: *)

(* :Mathematica Version: 7.0 *)

(* :History: *)

(* :Keywords: None *)

(* :Warnings: None *)

(* :Limitations: None *)

(* :Discussion: *)

(* :Requirements: None *)

(* :Examples: None *)


(*****************************************************************************)


BeginPackage["ClusterIntegration`", "SubKernels`"]


Unprotect[CCS, HPC, LSF, PBS, SGE, XGRID, ClusterName, EngineName, JobID]


(* supported engines *)

CCS::usage = "Windows Compute Cluster Server"
HPC::usage = "Windows HPC Server 2008"
LSF::usage = "Platform\[Trademark] LSF\[RegisteredTrademark]"
PBS::usage = "Altair\[Trademark] PBS Professional\[RegisteredTrademark]"
SGE::usage = "Sun Grid Engine"
XGRID::usage = "Apple Xgrid"


(* functions *)

ClusterName::usage = "ClusterName[kernel] gives the name \
of the cluster running the specified kernel."

EngineName::usage = "EngineName[kernel] gives the name of \
the cluster management system running the specified kernel."

JobID::usage = "JobID[kernel] gives the identifier of \
the job containing the specifier kernel."

TaskID::usage = "TaskID[kernel] gives the identifier of \
the task running the specifier kernel."


(* messages *)

General::undef =  "No resource management system found. Use ClusterSetEngine \
to initialize an engine with appropriate parameters."

General::load = "Cannot find components required for `1`."

General::resv = "Reservation of resources failed. \
Refer to the section `2` in the documentation about `1`."

General::connect = "Cannot connect to compute kernels."


(* information *)

`Information`$PackageVersion = 7.0;


(* loading cluster modules *)

Get["ClusterIntegration`CCS`"];
Get["ClusterIntegration`HPC`"];
Get["ClusterIntegration`LSF`"];
Get["ClusterIntegration`PBS`"];
Get["ClusterIntegration`SGE`"];
Get["ClusterIntegration`XGRID`"];


(* loading interface modules *)

Needs["ClusterIntegration`Palette`"];


EndPackage[]


(* loading subkernels' implementation *)

Needs["ClusterIntegration`SubKernels`"]


(* attributes *)

SetAttributes[
    {CCS, HPC, LSF, PBS, SGE, XGRID},
    {ReadProtected, Protected}
]

SetAttributes[
    {ClusterName, EngineName, JobID},
    {Listable, ReadProtected, Protected}
]
