(* :Title: Client.m -- client-side support code  *)

(* :Context: Parallel`Client` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   functions that are used to set up features of parallel computing, such
   as shared variables.
 *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 8 *)

(* :Note: keep this in sync with OldClient.m *)


BeginPackage["Parallel`Client`" ]

`HoldCompound

`CallBackPacket
`ReplyPacket
`CallBack
`remoteIdentity

`setSharedVariable
`setSharedFunction
`unsetShared

`makeDefinitions::usage = "makeDefinitions[s] makes definitions for symbols in the definition list s."

`$masterVersion=Null; (* default is the same as subkernel; newer masters set this at init time *)
`$ClientLanguageVersion = Parallel`Private`$ParallelLanguageVersion; (* identify client side *)

Begin["`Private`"]

`$PackageVersion = 1.0;
`$thisFile = $InputFileName


(* HoldCompound, client side; for the master side, see Kernels.m *)

HoldCompound=CompoundExpression; Protect[HoldCompound]


(* shared variable access, see VirtualShared.m *)

SetAttributes[{CallBack, CallBackPacket, ReplyPacket}, HoldFirst]

remoteIdentity[args___] := args

(* if the master returns a packet unevaluated, set it to Null to prevent looping *)
(* if the link fails we should quit *)
(* AbortProtect is probably the best we can do; aborting between write and read will lead
   to a messed-up state in the master kernel, which is worse than possibly losing the subkernels. *)

CallBack[packet_] := Module[{r},
  AbortProtect[
  LinkWrite[Parallel`Client`Private`$link, CallBackPacket[packet]];
  While[Head[r = LinkRead[Parallel`Client`Private`$link]] =!= ReplyPacket, 
        If[r===$Failed || Head[r]===LinkRead, Quit[]] ];
  If[ r === ReplyPacket[packet], Null, r[[1]] ]
  ]
]

Protect[ CallBackPacket, ReplyPacket, remoteIdentity, CallBack ]


(* set/unset shared variables *)

SetAttributes[{setSharedVariable, unsetShared}, HoldFirst]

(* the rule for HoldPattern[Part[s,___]] isn't used, unless you say
   Part[Unevaluated[s],...]
*)

setSharedVariable[s_, attrs_] := Module[{},
	Unprotect[s]; ClearAll[s];
	(* for all variables read access *)
	s := CallBack[s];
    s/: c:HoldPattern[Part[s,__]] := CallBack[c]; (* Part[Unevaluated[s], ...] *)
    s/: c:HoldPattern[Extract[s,__]] := CallBack[c]; (* Extract[Unevaluated[s], ...] *)
    s/: c:HoldPattern[Lookup[s,__]] := CallBack[c]; (* Lookup[s, ...]; does not work! *)
    s/: c:HoldPattern[KeyExistsQ[s,__]] := CallBack[c]; (* KeyExistsQ[Unevaluated[s], ...] *)
    s/: c:HoldPattern[Keys[s]] := CallBack[c]; (* Keys[Unevaluated[s], ...] *)
    s/: c:HoldPattern[Values[s]] := CallBack[c]; (* Values[Unevaluated[s], ...] *)

    (* for mutable variables *)
    If[ !MemberQ[attrs, Protected], With[{pp = Unprotect[Part]},
        s/: c:HoldPattern[s =rhs_] := (CallBack[c;];rhs); (* can we return the local copy of rhs? *)
        s/: c:HoldPattern[s:=rhs_] := CallBack[s:=Parallel`Developer`SendBack[rhs]];
        s/: c:(s++) := CallBack[c];
        s/: c:(s--) := CallBack[c];
        s/: c:(++s) := CallBack[c];
        s/: c:(--s) := CallBack[c];
        s/: c:AppendTo[s,rhs_]  := CallBack[c;]; (* don't waste bandwidth *)
        s/: c:PrependTo[s,rhs_] := CallBack[c;]; (* don't waste bandwidth *)
        s/: c:(s+=v_) := CallBack[c];
        s/: c:(s-=v_) := CallBack[c];
        s/: c:(s*=v_) := CallBack[c];
        s/: c:(s/=v_) := CallBack[c];
        (* associations *)
        s/: c:AssociateTo[s,__]  := CallBack[c;]; (* don't waste bandwidth *)
        s/: c:KeyDropFrom[s,__]  := CallBack[c;]; (* don't waste bandwidth *)
        (* part assignments *)
        Part/: c:HoldPattern[Part[s,args__]=rhs_] :=
        	Replace[{args}, {brgs___} :> CallBack[Part[s,brgs]=rhs]];
        Part/: c:HoldPattern[AppendTo[Part[s,args__],rhs_]] :=
        	Replace[{args}, {brgs___} :> CallBack[AppendTo[Part[s,brgs],rhs]]];
        Part/: c:HoldPattern[PrependTo[Part[s,args__],rhs_]] :=
        	Replace[{args}, {brgs___} :> CallBack[PrependTo[Part[s,brgs],rhs]]];

      Protect[pp]]
    ];
    Attributes[s] = Union[attrs,{Protected}];
]

(* how should we eval assignments happening on the remote side:
	s[args___]  = rhsi_
	s[args___] := rhsd_
	s[args___]++

   we should eval rhsi (happens automatically) and args, but what about rhsd? no.
   on the other hand, the := should not have the rhs evaluated on the master!
   Use sendBack[] around it.
   We should eval args in the usual way args of the lhs of a definition are evaluated.
   We approximate it with a closure, passing args as arguments.
   we could define an aux function for closure:
  	CallBackClosure[f, args...] --> CallBack[f[args]]
   this is just Composition[CallBack, f][args]
   but to avoid the ugly pure function f, we use substitution to insert the evaluated args into it.
*)

setSharedFunction[s_, attrs_] := Module[{},
	Unprotect[s]; ClearAll[s];
	(* for all variables read access *)
	d_s := CallBack[d]; (* or: s[___] *)
    (* for mutable variables *)
    If[ !MemberQ[attrs, Protected], With[{},
        s/: HoldPattern[s[args___]  = rhs_] :=
        	Replace[{args}, {brgs___} :> CallBack[s[brgs] = rhs]];
        s/: HoldPattern[s[args___] := rhs_] := 
        	Replace[{args}, {brgs___} :> CallBack[s[brgs]:= Parallel`Developer`SendBack[rhs]]];
        s/: HoldPattern[s[args___]++] := 
        	Replace[{args}, {brgs___} :> CallBack[s[brgs]++]];
        s/: HoldPattern[s[args___]--] := 
        	Replace[{args}, {brgs___} :> CallBack[s[brgs]--]];
        s/: HoldPattern[++s[args___]] := 
        	Replace[{args}, {brgs___} :> CallBack[++s[brgs]]];
        s/: HoldPattern[--s[args___]] := 
        	Replace[{args}, {brgs___} :> CallBack[--s[brgs]]];
        
    ]];
    Attributes[s] = Union[attrs,{Protected}];
]

unsetShared[s_] := With[{pp = Unprotect[Part]},
	Unprotect[s]; ClearAll[s];
	Quiet[ Part/: c:HoldPattern[Part[s, args__]=rhs_] =. ];
	Quiet[ Part/: c:HoldPattern[AppendTo[Part[s,args__],rhs_]] =. ];
	Quiet[ Part/: c:HoldPattern[PrependTo[Part[s,args__],rhs_]] =. ];
	Protect[pp];
]


(* DistributeDefinitions; see Parallel`Parallel` *)

makeDefinitions[c_Language`DefinitionList] := (Language`ExtendedFullDefinition[] = c;)


(* concurrency *)

(* acquire and release are shared system downvalues, set up statically here to save bandwidth *)

setSharedFunction[Parallel`Concurrency`acquire, {HoldFirst, Protected}]
setSharedFunction[Parallel`Concurrency`release, {HoldFirst, Protected}]

SetAttributes[CriticalSection, HoldAll]
CriticalSection[locks_List, code_] :=
 Module[{res},
  While[ ! Parallel`Concurrency`acquire[locks, $KernelID], Pause[0.1] ];
  res = (code);
  Parallel`Concurrency`release[locks];
  res
 ]

(* client-side queueing, disabled *)

SetAttributes[{ParallelSubmit}, HoldAll]
Protect[ParallelSubmit,WaitAll,WaitNext,Parallel`Developer`QueueRun,Parallel`Developer`DoneQ]
SetAttributes[EvaluationObject, HoldAllComplete]

(* shared system variables *)

setSharedVariable[$KernelCount, {Protected}]

End[]

EndPackage[]
