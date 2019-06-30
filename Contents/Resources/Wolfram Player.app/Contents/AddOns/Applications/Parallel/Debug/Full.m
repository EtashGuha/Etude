(* :Title: Debug/Full -- debugging support for Parallel Tools - extended version for use in Parallel Tools only *)

(* :Context: Parallel`Debug` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   user debug functions for the Parallel Computing Toolkit.
   Allow monitoring communication with remote kernels and queueing of processes.
 *)

(* :Package Version: 3.0  *)

(* :Mathematica Version: 7 *)

(* :Note: this package must be read *before* the Parallel Computing Toolkit itself
	It is normally read from Parallel`Kernel`autoinit` if Parallel`Static`$loadDebug is True *)

(* check for loading after PCT *)

If[ NameQ["Parallel`Private`$PackageVersion"],
	Parallel`Debug`$Debug::toolate = "The debugging package cannot be read after the Parallel Computing Toolkit itself.";
	Message[Parallel`Debug`$Debug::toolate];
	Abort[]
]

Get["Parallel`Debug`Standalone`"]

Needs["Parallel`Debug`Perfmon`"] (* extend by performance monitor *)

Null
