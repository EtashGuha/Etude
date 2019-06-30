(* :Title: FIFO.m -- FIFO queue *)

(* :Context: Parallel`Queue`FIFO *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   a simple first in first out queue, implemented as a circular buffer
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


BeginPackage["Parallel`Queue`FIFO`", { "Parallel`Queue`Interface`" }]

(* stuff not in interface *)

FIFOQueue::usage = "FIFOQueue[] creates an empty queue."

fifo::usage = "fifo[...] is the print form of FIFO queues."


Begin["`Private`"] (**************************************************)

`$PackageVersion = 1.0;

SetAttributes[{queue}, HoldAll]
SetAttributes[array, HoldAllComplete]

makeArray[n_] := array@@Table[Null, {n}]

`$initSize = 128;

(* n0 is the first item, n1 is the next free item. if the queue is empty, n0==n1,
   max size is Length[array]-1, then extend it *)

FIFOQueue[is:(_Integer?Positive):$initSize] := Module[{ar=makeArray[is], n0=1, n1=1}, queue[ar, n0, n1] ]

queue/: Copy[queue[br_, m0_, m1_]] := Module[{ar=br, n0=m0, n1=m1}, queue[ar, n0, n1] ]

queue/: Size[queue[ar_, n0_, n1_]] := Mod[n1-n0, Length[ar]] (* is much faster *)

(* EmptyQ: use generic implementation *)

queue/: EnQueue[q:queue[ar_, n0_, n1_], val_] := (
    If[ Size[q] == Length[ar]-1, (* extend (double size) *)
        ar = RotateLeft[ar, n0-1]; (* make sure there is no wraparound *)
        {n0, n1} = {1, Mod[n1-n0, Length[ar]] + 1};
        ar = Join[ar, makeArray[Length[ar]]] ];
    ar[[n1]] = val;
    n1 = Mod[n1, Length[ar]] + 1;
    q
  )

queue/: Top[q:queue[ar_, n0_, n1_]]/; Size[q]>0 := ar[[n0]]

queue/: DeQueue[q:queue[ar_, n0_, n1_]]/; Size[q]>0 := 
  With[{res = ar[[n0]]},
    ar[[n0]] = Null;
    n0 = Mod[n0, Length[ar]] + 1;
    res
  ]

(* use generic implementations for handling DeQueue/Top of empty queues *)

queue/: delete[queue[ar_, n0_, n1_]] := (ClearAll[ar,n0,n1];)

(* if we can do it fast, why not? *)
queue/: Normal[q:queue[ar_, n0_, n1_]] := List@@Take[RotateLeft[ar, n0-1], Size[q]]

Parallel`Queue`Interface`Private`setFormat[queue, fifo] (* generic format *)

queue/: qQ[queue[ar_, n0_, n1_]] := IntegerQ[n0] (* type predicate *)

Protect[ queue ]

End[]

AppendTo[ $QueueTypes, FIFOQueue ];

Protect[ FIFOQueue, fifo ]

EndPackage[]
