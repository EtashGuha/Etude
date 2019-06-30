(* :Title: LIFO.m -- LIFO queue *)

(* :Context: Parallel`Queue`LIFO *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   a fast queue that happens to be last in - first out (or a stack)
 *)

(* :Copyright: © 2004 by MathConsult Dr. R. Mäder.

   Permission is granted to use and distribute this file for any purpose
   except for inclusion in commercial software or program collections.
   This copyright notice must remain intact.
*)

(* :Package Version: 1.0  *)

(* :History:
   Version 1.0 for the Parallel Computing Toolkit, Version 2.
*)

(* :Mathematica Version: 5 (with some V4 backwards compat support) *)

(* :Note: this queue does not use the Priority selector *)


BeginPackage["Parallel`Queue`LIFO`", { "Parallel`Queue`Interface`" }]

(* stuff not in interface *)

LIFOQueue::usage = "LIFOQueue[] creates an empty queue."

lifo::usage = "lifo[...] is the print form of LIFO queues."


Begin["`Private`"] (**************************************************)

`$PackageVersion = 1.0;

(* the items are stored in an array that grows as needed; insertion/removal is
   always at the end; this makes it a bit faster than the LIFO queue *)

SetAttributes[{queue}, HoldAll]
SetAttributes[array, HoldAllComplete]

makeArray[n_] := array@@Table[Null, {n}]

`$initSize = 128;

(* n1 is the next free item. if the queue is empty, n1==1,
   max size is Length[array], then extend it *)

LIFOQueue[is:(_Integer?Positive):$initSize] := Module[{ar=makeArray[is], n1=1}, queue[ar, n1] ]

queue/: Copy[queue[br_, m1_]] := Module[{ar=br, n1=m1}, queue[ar, n1] ]

queue/: Size[queue[ar_, n1_]] := n1-1 (* is much faster *)

(* EmptyQ: use generic implementation *)

queue/: EnQueue[q:queue[ar_, n1_], val_] := (
    If[ n1 > Length[ar], (* extend (double size) ( n1 is hopefully Length[ar]+1) *)
        ar = Join[ar, makeArray[Length[ar]]] ];
    ar[[n1++]] = val;
    q
  )

queue/: Top[q:queue[ar_, n1_]/; n1>1] := ar[[n1-1]]

queue/: DeQueue[q:queue[ar_, n1_]/; n1>1] := 
  With[{res = ar[[--n1]]}, ar[[n1]] = Null; res ]

(* use generic implementations for handling DeQueue/Top of empty queues *)

queue/: delete[queue[ar_, n1_]] := (Clear[ar,n1];)

(* if we can do it fast, why not? *)
queue/: Normal[queue[ar_, n1_]] := Reverse[List@@Take[ar, n1-1]]

Parallel`Queue`Interface`Private`setFormat[queue, lifo] (* generic format *)

queue/: qQ[queue[ar_, n1_]] := IntegerQ[n1] (* type predicate *)

Protect[ queue ]

End[]

AppendTo[ $QueueTypes, LIFOQueue ];

Protect[ LIFOQueue, lifo ]

EndPackage[]
