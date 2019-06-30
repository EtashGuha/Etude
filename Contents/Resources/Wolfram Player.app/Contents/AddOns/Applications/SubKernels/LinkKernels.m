(* :Title: LinkKernels.m -- connect to established link objects and link names *)

(* :Context: SubKernels`LinkKernels` *)

(* :Author: Roman E. Maeder *)

(* :Copyright: © 2008 by Wolfram Research, Inc. *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 6 *)

(* :History:
   1.0 first released version.
*)

(* this implementation of the remote kernels interface uses the methods previously in PCT/RemoteKernels
  for launching kernels through MathLink calls out of Mathematica.
 *)

BeginPackage["SubKernels`LinkKernels`", { "SubKernels`" }]

Needs["SubKernels`Protected`"]

(* the configuration language. A kernel is described as a LinkObject data element *)
 
(* short forms of kernel descriptions recognized by this implementation:
 	- "port1@host,port2@host" a TCPIP link name
 	- LinkObject[wawa] an open link object
 *)
 
 (* options *)
 
Options[ConnectLink] = {
	LinkProtocol -> "TCPIP",
	KernelSpeed -> 1
}

(* class methods and variables *)

linkKernelObject::usage = "linkKernelObject[method] is the local kernels class object."

(* additional constructors, methods *)

ConnectLink::usage = "ConnectLink[\"port@host\", opts..] connects to a waiting kernel using LinkConnect. Options are passed to LinkConnect.
	ConnectLink[link, opts..] connects to an already established LinkObject."

(* destructor *)

(* the data type is public, for easier subclassing and argument type check *)

linkKernel::usage = "linkKernel[..] is a link subkernel."

 (* remember context now *)

linkKernelObject[subContext] = Context[]


Begin["`Private`"]
 
`$PackageVersion = 0.9;
`$thisFile = $InputFileName
 
(* data type:
 	linkKernel[ lk[link, descr, arglist, speed] ]
 		link	associated LinkObject
 		descr	host name, if known, other identifier otherwise
 		arglist	list of arguments used in constructor, so that it may be relaunched if possible
		speed	speed setting, mutable
 *)

protected = Unprotect[LinkObject]

SetAttributes[linkKernel, HoldAll] (* data type *)
SetAttributes[`lk, HoldAllComplete] (* the head for the base class data *)
 
(* private selectors; pattern is linkKernel[ lk[link_, descr_, arglist_, id_, ___], ___ ]  *)
 
linkKernel/: linkObject[ linkKernel[lk[link_, ___], ___]] := link
linkKernel/: descr[linkKernel[lk[link_, descr_, ___], ___]] := descr
linkKernel/: arglist[linkKernel[lk[link_, descr_, arglist_, ___], ___]] := arglist
linkKernel/: kernelSpeed[linkKernel[lk[link_, descr_, arglist_, speed_, ___], ___]] := speed
linkKernel/: setSpeed[linkKernel[lk[link_, descr_, arglist_, speed_, ___], ___], r_] := (speed = r)

(* factory method *)

LinkObject/: NewKernels[link_LinkObject] := ConnectLink[link]

(* interface methods *)

linkKernel/:  subQ[ linkKernel[ lk[link_, descr_, arglist_, ___] ] ] := Head[link]===LinkObject

linkKernel/:  LinkObject[ kernel_linkKernel ]  := linkObject[kernel]
linkKernel/:  MachineName[ kernel_linkKernel ] := descr[kernel]
linkKernel/:  Description[ kernel_linkKernel ] := LinkObject[kernel]
linkKernel/:  Abort[ kernel_linkKernel ] := kernelAbort[kernel]
linkKernel/:  SubKernelType[ kernel_linkKernel ] := linkKernelObject
(* KernelSpeed: use generic implementation *)

(* Clone[]: connect kernels are not cloneable *)


(* list of open kernels *)

`$openkernels = {}

linkKernelObject[subKernels] := $openkernels


(* constructors *)

(* connect to waiting link given its name *)

ConnectLink[addr_String, opts:OptionsPattern[]] :=
Module[{link},
    feedbackObject["name", StringForm["\[LeftSkeleton]a mathlink\[RightSkeleton]" ]];
    Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Connecting to `1` with `2`", addr, "LinkConnect"[addr, System`Utilities`FilterOptions[LinkConnect, opts, Options[ConnectLink]]]];
    link = LinkConnect[addr, System`Utilities`FilterOptions[LinkConnect, opts, Options[ConnectLink]]];
    feedbackObject["tick"];
    Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Connected to `1`", link];
    If[ link === $Failed, link, initLink[ link, addr, {addr, opts}, OptionValue[ConnectLink, {opts}, KernelSpeed] ] ]
]

(* connect to existing link object *)

ConnectLink[link_LinkObject, opts:OptionsPattern[]] := (
    feedbackObject["name", StringForm["\[LeftSkeleton]a link object\[RightSkeleton]"]];
    Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Using existing link `1`", link];
    feedbackObject["tick"];
	initLink[ link, First[link], {link, opts}, OptionValue[ConnectLink, opts, KernelSpeed] ]
)

(* parallel version; there is nothing to do concurrently *)

ConnectLink[addrs_List, opts:OptionsPattern[]] := deleteFailed[ (feedbackObject["tick"]; ConnectLink[#, opts])& /@ addrs, ConnectLink]



(* destructor; use generic implementation *)

linkKernel/: Close[kernel_linkKernel?subQ] := (
	$openkernels = DeleteCases[$openkernels, kernel];
	kernelClose[kernel, True]
)


(* handling short forms of kernel descriptions *)

linkKernelObject[try][s_String]/; StringMatchQ[s,RegularExpression["\\d+@[\\w.]+,\\d+@[\\w.]+"]] := ConnectLink[s] (* TCPIP linkname *)

linkKernelObject[try][link_LinkObject] := ConnectLink[link] (* existing link object *)


(* class name *)

linkKernelObject[subName] = "MathLink Objects"


(* raw constructor *)
 
initLink[link_, host_, args_, sp_] :=
 Module[{kernel, speed = sp},
 	kernel = linkKernel[ lk[link, host, args, speed] ];
 	(* local init *)
 	AppendTo[$openkernels, kernel];
 	(* base class init *)
 	kernelInit[kernel]
 ]


(* config: not configurable *)


(* class variable defaults *)
 
(* format *)

setFormat[linkKernel, "link"]

Protect[Evaluate[protected]]

End[]

Protect[ ConnectLink, linkKernelObject, linkKernel ]

(* registry *)
addImplementation[linkKernelObject]

EndPackage[]
