(* :Title: Null.m -- No-Debug dummy package *)

(* :Context: none *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   minimum stuff required in the Parallel`Debug` context even if not debugging
 *)

(* :Package Version: 2.1  *)

(* :Mathematica Version: 7 *)

(* symbols that should exist anyway, but Parallel`Debug` should not be on the context path *)

Begin["Parallel`Debug`"]

`$Debug = False
`$Debug::usage = "$Debug is False, if debugging is disabled."
Protect[`$Debug]

End[]

Begin["Parallel`Debug`Private`"]

`$PackageVersion = 2.0;
`$thisFile = $InputFileName

(* stuff to be used in the PCT code, null implementations *)

`RegisterTrace (* no value needed *)
`$hideVals = False; (* not used *)

`ExportEnvironmentWrapper=Identity
Clear[`RemoteEvaluateWrapper] (* should not have a value if not used *)
Clear[`RemoteKernelInit] (* should not have a value if not used *)

(* tracing; make things a bit faster by not even evaluating the arguments of the unused trace[] calls, and don't bother to define a downvalue for trace[] *)

SetAttributes[`trace, HoldAllComplete]

End[]

Null
