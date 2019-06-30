(* :Title: Lisp.m -- Lisp-like queue *)

(* :Context: Parallel`Queue`Lisp *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   a Lisp-like car/cdr cons queue
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


BeginPackage["Parallel`Queue`Lisp`", { "Parallel`Queue`Interface`" }]

(* stuff not in interface *)

LispQueue::usage = "LispQueue[] creates an empty queue."

Begin["`Private`"] (**************************************************)

`$PackageVersion = 1.0;

(* the items are stored as cons[e1, cons[e2, ...,cons[]...]], just as in Lisp *)

SetAttributes[{queue}, HoldAll]
SetAttributes[cons, HoldAllComplete]
`empty = cons[] (* shared nil *)
car[cons[car_, cdr_]] := car
cdr[cons[car_, cdr_]] := cdr

LispQueue[] := Module[{r=empty}, queue[r] ]

queue/: Copy[queue[s_]] := Module[{r=s}, queue[r] ]

(* Size: use generic implementation *)

queue/: EmptyQ[queue[r_]] := r===empty

queue/: EnQueue[q:queue[r_], val_] := (With[{rv=r}, r = cons[val,rv]]; q) (* must eval r *)

queue/: Top[q:queue[r_]]/; !EmptyQ[q] := car[r]

queue/: DeQueue[q:queue[r_]/; !EmptyQ[q]] :=
		With[{res = car[r]}, r = cdr[r]; res ]

(* use generic implementations for handling DeQueue/Top of empty queues *)

queue/: delete[queue[r_]] := (r=empty;)

(* Normal: use generic implementation *)

(* have to code around an old bug in Infix[] *)

Format[q_queue?qQ] :=
    With[{els = Normal[q]},
      Switch[Length[els],
        0,	SequenceForm["(", ")"],
        1,	SequenceForm["(", First[els], ")"],
        _,	SequenceForm["(", Infix[els, " "], ")"]
      ]
    ]

queue/: qQ[queue[r_]] := ValueQ[r] (* type predicate *)

Protect[ queue ]

End[]

AppendTo[ $QueueTypes, LispQueue ];

Protect[ LispQueue ]

EndPackage[]
