(* :Title: Interface.m -- Interface for queues for the PCT *)

(* :Context: Parallel`Queue`Interface *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   Declaration of the interface that all queues must implement,
   and one selector for the data queued.
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

(* :Mathematica Version: 4.2 *)

(* :Source: based on PriorityQueue.m for the Mathematica Journal V7.3 *)

(* :Source: 
    Maeder, Roman E. 1997. Interval Plotting and Global Optimisation.
        The Mathematica Journal, 7(2).
*)


BeginPackage[ "Parallel`Queue`Interface`" ]

$QueueTypes::usage = "$QueueTypes is the list of available queues, identified by the name of their constructor. New types defined should AppendTo this list."

EnQueue::usage = "EnQueue[q, item] inserts item into the queue q."
DeQueue::usage = "DeQueue[q] removes the largest item from the priority queue q.
	It returns the item removed."
EmptyQ::usage = "EmptyQ[q] is True if the queue q is empty."
Top::usage = "Top[q] gives the largest item in the queue q."
Size::usage = "Size[q] gives the number of elements in the queue q."
delete::usage = "delete[q] frees storage associated with the queue q."

qQ::usage = "qQ[queue] is True if queue is a valid queue (a subtype of this interface)."

(* optional copy constructory *)
Copy::usage = "Copy[q] makes a copy of the queue q (optional)."

(* optional priority or some such *)
Priority::usage = "Priority[data] gives scheduling info that a queue may use."

(* message for an attempt to dequeue or look at top of an empty queue *)
DeQueue::empty = "Queue `1` is empty."

(* message for an attempt to call an undefined method *)
General::NIM = "Method `1` not implemented in class `2`."


Begin["`Private`"] (**************************************************)

`$PackageVersion = 2.0;


$QueueTypes = {}; (* none loaded so far *)

Priority[_] = Null; (* default *)

(* generic implementations. You need to override at least one of Size/EmptyQ *)

protected = Unprotect[ Normal, Top ] (* system symbols we modify *)

qQ[_] = False (* unless known otherwise, an expression is not a queue *)

Size[q0_?qQ] :=
  Module[{n = 0, q = Copy[q0]},
    While[ !EmptyQ[q], DeQueue[q]; n++ ];
    delete[q];
    n
  ]

EmptyQ[q_?qQ] := Size[q]==0

(* attempt to look at top of empty queues *)

Top[q_?EmptyQ]     := (Message[DeQueue::empty, q]; $Failed)
DeQueue[q_?EmptyQ] := (Message[DeQueue::empty, q]; $Failed)

(* AppendTo is so abominably slow, we have to use something different *)

Normal[q0_?qQ] :=
  Module[{l = h[], q = Copy[q0]},
    While[ !EmptyQ[q], l = h[l, DeQueue[q]] ];
    delete[q];
    List @@ Flatten[l, Infinity, h]
  ]

(* Top should normally be overridden because this implementation is slow *)

Top[q0_?qQ] :=
  Module[{res, q = Copy[q0]},
    res = DeQueue[q];
    delete[q];
    res
  ]

(* complain if unimplemented (pure virtual) method is called *)

EnQueue[q_?qQ, _] := (Message[EnQueue::NIM, EnQueue, Head[q]]; $Failed)
DeQueue[q_?qQ]    := (Message[DeQueue::NIM, DeQueue, Head[q]]; $Failed)
Copy[q_?qQ]       := (Message[Copy::NIM, Copy, Head[q]]; $Failed)

(* private formatting helpers *)

With[{n=3},
  doFormat[q_, head_] :=
      Module[{s = Size[q], l, res},
	l = Min[s, n];
	If[s == l+1, l++];
	res = Take[Normal[q], l];
	If[l < s, AppendTo[res, "\[Ellipsis]"]];
	head@@res
      ]
]

setFormat[type_, head_] := (Format[q_type?qQ] := doFormat[q, head])

Protect[ Evaluate[protected] ] (* restore protection of system symbols *)

End[]

(* encourage the use of upvalues ;-) *)

Protect[ EnQueue, DeQueue, EmptyQ, Size, delete, Copy, Priority, qQ ]

EndPackage[]
