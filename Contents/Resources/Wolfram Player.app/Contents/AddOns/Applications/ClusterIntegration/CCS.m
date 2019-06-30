(* :Name: CCS Library File *)

(* :Title: Integration with Microsoft Compute Cluster Server 2003 *)

(* :Context: ClusterIntegration`CCS` *)

(* :Author: Charles Pooh *)

(* :Summary:

    This package provides functionalities for integrating Mathematica
    with Microsoft Compute Cluster Server 2003.

*)

(* :Copyright: (c) 2006 - 2008 Wolfram Research, Inc. *)

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


BeginPackage["ClusterIntegration`CCS`", "ClusterIntegration`"]


(* usage *)

CCSNewComputeKernels::usage = "Internal function. Use NewKernels to launch \
remote kernels."

CCSLoginScript::usage = "Internal functions. Use Kernel Configuration menu \
to configure kernels."

CCSMathBatchJob::usage = "Internal functions. Use the Batch Submission Job \
palette to submit batch jobs."


(* loading modules for each platform *)

Switch[$OperatingSystem,
       "Windows", Needs["ClusterIntegration`CCSWin`"],
       _, Null
]


EndPackage[]

