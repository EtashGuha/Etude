(* :Title: NullMaster.m -- client-side definitions for parallel computing  *)

(* :Context: Parallel`NullMaster` (empty) *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   minimal support for failover of parallel computations that might be invoked on
   a subkernel of another parallel computation or in a Player kernel.
 *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 7 *)


BeginPackage["Parallel`NullMaster`" ]

General::subpar = "Parallel computation is not supported in this kernel. Proceeding with sequential evaluation."
General::subnopar = "Parallel programming is not available in this kernel."

Begin["`Private`"]

`$PackageVersion = 1.0;
`$thisFile = $InputFileName

failover[par_, seq_] := (
	SetAttributes[par,HoldAll];
	par[args___, OptionsPattern[]] := (Message[par::subpar]; seq[args]);
	Protect[par];
)

failover[Parallelize, Identity]
failover[ParallelCombine, #1[#2]&]
failover[ParallelMap, Map]
failover[ParallelTable, Table]
failover[ParallelSum, Sum]
failover[ParallelProduct, Product]
failover[ParallelDo, Do]
failover[ParallelArray, Array]

(* make parallel programming fail and return $Failed
   (so these commands never make it to the master by mistake *)

nim[par_] := (
	par[___] := (Message[par::subnopar]; $Failed)
)

nim /@ {
	ParallelTry,
	LaunchKernels, AbortKernels, CloseKernels,
	ParallelEvaluate, WaitAll, WaitNext,
	ParallelNeeds, DistributeDefinitions,
	SetSharedVariable, SetSharedFunction, $SharedVariables, $SharedFunctions, UnsetShared,
	Parallel`Developer`ParallelDispatch,
	Parallel`Developer`Send, Parallel`Developer`Receive, Parallel`Developer`ReceiveIfReady 
}

(* consistent null definitions *)

Kernels[] = {}
$ConfiguredKernels={}

Protect[Kernels, $ConfiguredKernels]

(* do not touch these, as they may be used on the client side *)
{$KernelCount, $KernelID, ParallelSubmit, CriticalSection}

(* support for LaunchDefaultKernels[] on subkernel side *)

Parallel`Developer`LaunchDefaultKernels[] := False
Protect[Parallel`Developer`LaunchDefaultKernels]

End[]

(* for the Player products, do not emit fallback messages, and set further variables *)
If[Parallel`Static`$player,
	Off[General::subpar, General::subnopar];
	$KernelCount = 0;
]

EndPackage[]
