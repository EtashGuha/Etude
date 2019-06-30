(* :Title: Priority.m -- priority queue *)

(* :Context: Parallel`Queue`Priority *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   Priority queue based on a heap
 *)

(* :Copyright: © 2004 by MathConsult Dr. R. Mäder.

   Permission is granted to use and distribute this file for any purpose
   except for inclusion in commercial software or program collections.
   This copyright notice must remain intact.
*)

(* :Package Version: 2.0  *)

(* :History:
   Version 2.0 for the Parallel Computing Toolkit, Version 2.
   Version 1.0 for the Mathematica Journal, July 1997.
*)

(* :Mathematica Version: 5 (with some V4 backwards compat support) *)

(* :Note: this queue uses the Priority selector as a real-valued priority *)

(* :Source: based on PriorityQueue.m for the Mathematica Journal V7.3 *)

(* :Source: 
    Maeder, Roman E. 1997. Interval Plotting and Global Optimisation.
        The Mathematica Journal, 7(2).
*)


BeginPackage["Parallel`Queue`Priority`", { "Parallel`Queue`Interface`" }]

(* stuff not in interface *)

priorityQueue::usage = "priorityQueue[] creates an empty queue."

priority::usage = "priority[...] is the print form of priority queues."


Begin["`Private`"] (**************************************************)

`$PackageVersion = 2.0;

SetAttributes[{queue, item}, HoldAll]
SetAttributes[array, HoldAllComplete]

makeArray[n_] := array@@Table[Null, {n}]

`$initSize = 128;

priorityQueue[is:(_Integer?Positive):$initSize] := Module[{ar = makeArray[is],n = 0}, queue[ar, n] ]

queue/: Copy[queue[ar0_,n0_]] := Module[{ar=ar0, n=n0}, queue[ar, n] ]

queue/: Size[queue[ar_,n_]] := n (* is much faster *)

(* EmptyQ: use generic implementation *)

(* default priority: 1 *)
prio[data_] := Replace[ Priority[data], Null -> 1 ]

(* we need to do only half-exchanges. In the original code,
   ar[[i]] is always the new val so we restore it only at the end *)

queue/: EnQueue[q:queue[ar_,n_], val_] :=
  Module[{i,j, pi},
    If[ n == Length[ar], (* extend (double size) *)
        ar = Join[ar, makeArray[Length[ar]]] ];
    n++;
    i = n;  (* ar[[i]] = val;*) 
    pi = prio[val];
    While[ True, (* restore heap *)
      j = Floor[i/2];
      If[ j < 1 || prio[ar[[j]]] >= pi, ar[[i]]=val; Break[] ];
      (* {ar[[i]], ar[[j]]} = {ar[[j]], ar[[i]]}; *)
      ar[[i]] = ar[[j]]; i = j;
    ];
    q
  ]

queue/: Top[queue[ar_,n_]]/;n>0 := ar[[1]]

queue/: DeQueue[queue[ar_,n_]/; n>0] := 
  Module[{i, j=1, res = ar[[1]], val, pj},
    (* ar[[j]] = ar[[n]]; *)
    val = ar[[n]]; ar[[n]] = Null; n--;
    pj = prio[val]; (* = prio[ar[[j]]; *)
    While[ j <= Floor[n/2], (* restore heap *)
        i = 2j;
        If[ i < n && prio[ar[[i+1]]] > prio[ar[[i]]], i++ ]; (* larger son *)
        If[ prio[ar[[i]]] > pj,
            (* {ar[[i]], ar[[j]]} = {ar[[j]], ar[[i]]}; *)
            ar[[j]] = ar[[i]]; j = i;
        , (* else finish early *)
            Break[];
        ];
    ];
    ar[[j]] = val;
    res
  ]

(* use generic implementations for handling DeQueue/Top of empty queues *)

queue/: delete[queue[ar_,n_]] := (Clear[ar,n];)

(* Normal: use generic implementation *)

Parallel`Queue`Interface`Private`setFormat[queue, priority] (* generic format *)

queue/: qQ[queue[ar_, n_]] := IntegerQ[n] (* type predicate *)

Protect[ queue ]

End[]

AppendTo[ $QueueTypes, priorityQueue ];

Protect[ priorityQueue, priority ]

EndPackage[]
