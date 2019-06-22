(* :Title: VirtualShared.m -- Virtual Shared Memory *)

(* :Context: Parallel`VirtualShared` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   Virtual shared memory for parallel evaluation of Mathematica expressions.
 *)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 7 *)

(* :History:
   3.0 for PCT 3.0, Mathematica 7
   2.0 update for PCT V2, tested on Mathematica 4.2 and 5
   1.0 first released version.
   0.9 for IMS '97, Rovaniemi, June 1997.
*)

(* :Limitations:
   This package does not work together with latency hiding.
*)

BeginPackage["Parallel`VirtualShared`"]

System`SetSharedVariable
System`SetSharedFunction
System`UnsetShared

System`$SharedVariables
System`$SharedFunctions


BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

(* used also on slave side *)
`SendBack::usage = "SendBack[expr] keeps expr unevaluated on the master kernel and evaluates it normally on the slave kernels."

EndPackage[]

(* semi-hidden methods, aka protected *)

BeginPackage["Parallel`Protected`"]

sharedSystemVariable::usage = "sharedSystemVariable[s] defines s as a shared internal variable."
sharedSystemDownValue::usage = "sharedSystemDownValue[s] defines s as a shared internal function."
declareSystemVariable::usage = "declareSystemVariable[s] declares s as a shared internal variable."
declareSystemDownValue::usage = "declareSystemDownValue[s] declares s as a shared internal function."

EndPackage[]

(* messages *)

General::badsym = "Symbol `1` cannot be shared."


Begin["`Private`"]

`$PackageVersion = 3.0;
`$thisFile = $InputFileName

Needs["Parallel`Parallel`"] (* for AddInitCode *)
Needs["Parallel`Protected`"] (* access protected stuff *)

(* debug tracers used in this package *)

Parallel`Debug`Private`RegisterTrace[ Parallel`Debug`SharedMemory, "SharedMemory is a tracer that triggers when remote kernels make callbacks for shared memory access." ]

$LoadFactor::shared = "Latency hiding ($LoadFactor>1) is incompatible with virtual shared memory."

(* peek into Parallel`Parallel`Private` and other dirty things *)

`holdCompound     = Parallel`Client`HoldCompound

(* slave-side symbols, handle with care *)

Parallel`Client`CallBackPacket
Parallel`Client`ReplyPacket
Parallel`Client`CallBack
Parallel`Developer`SendBack
Parallel`Client`remoteIdentity

(* master side attributes *)

SetAttributes[{Parallel`Client`CallBackPacket}, HoldFirst]

(* list of currently shared variables and downsymbols, internal version *)
`$sharedVariables = Hold[];
`$sharedDownValues = Hold[];
(* private ones *)
`$sharedSystemVariables = Hold[];
`$sharedSystemDownValues = Hold[];

(* cannot use latency hiding; provide a way for Parallel.m to find out whether there are shared variables *)
`$sharingActive = False;
checkSharing :=
	($sharingActive = Length[$sharedVariables]+Length[$sharedDownValues]+Length[$sharedSystemVariables]+Length[$sharedSystemDownValues] > 0)

(* readonly public version: list of held symbols *)
$SharedVariables := List@@Hold/@$sharedVariables
$SharedFunctions := List@@Hold/@$sharedDownValues
Protect[ $SharedVariables, $SharedFunctions ]

(* remote evaluation of delayed rhs:
   - before making the callback, client wraps rhs into SendBack (no attributes required)
   - inside the callback handler, SendBack is set to remoteIdentity, which is HoldAll on the master side
   - on the client side, remoteIdentity goes away
   - outside the packet handler, SendBack turns into identity and goes away for the sequential fallback
*)

SendBack = identity
SetAttributes[Parallel`Client`remoteIdentity, {HoldAllComplete}]

(* ownvalues *)

SetAttributes[varDef, HoldAll]

(* these symbols must not be shared, as well as all locked ones *)

$badsyms = {Null}

(* attributes that can be used on subkernels side *)

$ownAttrs = {Protected}

varDef[s_Symbol] := (
	If[!ValueQ[s], With[{pp=Unprotect[s]}, s=Null; Protect[pp]]]; (* must have a value *)
	(* command to send to subkernels *)
	With[{attrs=Intersection[Attributes[s], $ownAttrs]},
		holdCompound[Parallel`Client`setSharedVariable[s, attrs]]
	]
)

(* downvalues *)

(* but not: Listable,OneIdentity,Orderless,Flat *)
$downAttrs = {Protected,HoldAll,HoldFirst,HoldRest,HoldAllComplete,NHoldAll,NHoldFirst,NHoldRest,NumericFunction,SequenceHold}

downDef[s_Symbol] := (
	(* command to send to subkernels *)
	With[{attrs=Intersection[Attributes[s], $downAttrs]},
		holdCompound[Parallel`Client`setSharedFunction[s, attrs]]
	]
)


SetAttributes[unDef, HoldAll]

unDef[s_Symbol] :=
    holdCompound[ Parallel`Client`unsetShared[s] ]


(* client support code for shared variables is now in Parallel`Client` and Parallel`OldClient` *)


SetAttributes[SetSharedVariable, HoldAll]

SetSharedVariable[{___,s_Symbol,___}]/; (MemberQ[Attributes[s],Locked] || MemberQ[Attributes[s],ReadProtected] || MemberQ[$badsyms,Unevaluated[s]]) :=
	(Message[SetSharedVariable::badsym, HoldForm[s]]; Null)

SetSharedVariable[l:{s___Symbol}] :=
 Catch[
  With[{varDefs = Join@@(varDef/@Unevaluated[l])},
    AddInitCode[ varDefs ]; (* init kernels *)
    $sharedVariables = Union[$sharedVariables, Hold[s]];
    Parallel`Protected`removeDistributedSymbol[s]; (* can no longer be a distributed variable *)
    checkSharing;
  ],
 badsym ]

SetSharedVariable[s___Symbol] := SetSharedVariable[{s}]

(* SetSharedFunction does not need to be HoldFirst *)

SetSharedFunction[{___,s_Symbol,___}]/; (MemberQ[Attributes[s],Locked] || MemberQ[Attributes[s],ReadProtected] || MemberQ[$badsyms,Unevaluated[s]]) :=
	(Message[SetSharedFunction::badsym, HoldForm[s]]; Null)

SetSharedFunction[l:{s___Symbol}] :=
 Catch[
  With[{varDefs = Join@@(downDef/@l)},
    AddInitCode[ varDefs ]; (* init kernels *)
    $sharedDownValues = Union[$sharedDownValues, Hold[s]];
    Parallel`Protected`removeDistributedSymbol[s]; (* can no longer be a distributed variable *)
    checkSharing;
  ],
 badsym ]

SetSharedFunction[s___Symbol] := SetSharedFunction[{s}]

SetAttributes[UnsetShared, HoldAll]

UnsetShared[{s___Symbol}] :=
 With[{l=Intersection[Join[$sharedVariables, $sharedDownValues],Hold[s]]}, (* only shared ones, bug/308054 *)
  With[{varDefs = Join@@(unDef/@l)},
    AddInitCode[ varDefs ]; (* init kernels *)
    $sharedVariables = Complement[$sharedVariables, Hold[s]];
    $sharedDownValues = Complement[$sharedDownValues, Hold[s]];
    checkSharing;
    List@@HoldForm/@l
  ]]

UnsetShared[s___Symbol] := UnsetShared[{s}]

(* allow a string pattern to select variables *)

UnsetShared[patt_String] :=
    Replace[ Select[Join[$sharedVariables, $sharedDownValues],
		Function[s, StringMatchQ[SymbolName[Unevaluated[s]], patt], HoldFirst]],
	Hold[s___] :> UnsetShared[{s}] ]

(* for internal variables *)

SetAttributes[{sharedSystemVariable, declareSystemVariable}, HoldAll]

sharedSystemVariable[s_Symbol] :=
  With[{varDefs = varDef[s]},
    AddInitCode[ varDefs, True ]; (* init kernels, permanent *)
    declareSystemVariable[s];
  ]

sharedSystemDownValue[s_Symbol] :=
  With[{varDefs = downDef[s]},
    AddInitCode[ varDefs, True ]; (* init kernels, permanent *)
    declareSystemDownValue[s];
  ]

(* declare only, for those set up in Client.m *)

declareSystemVariable[s_Symbol] := (
	$sharedSystemVariables = Union[$sharedSystemVariables, Hold[s]];
    checkSharing;
)
declareSystemDownValue[s_Symbol] := (
    $sharedSystemDownValues = Union[$sharedSystemDownValues, Hold[s]];
    checkSharing;
)

(* cleanup after ClearKernels[]; forget all shared variables *)

Parallel`Kernels`Private`registerClearCode[ UnsetShared["*"] ]

(* overload handler for CallBackPacket; this is called inside AbortProtect[] *)

Parallel`Client`CallBackPacket/:
PacketHandler[ Parallel`Client`CallBackPacket[expr_], k_ ] :=
    Block[{SendBack = Parallel`Client`remoteIdentity},
      Module[{aborted=False},
        With[{res = CheckAbort[expr, aborted=True; $Aborted]}, (* eval *)
          Parallel`Debug`Private`trace[Parallel`Debug`SharedMemory, "`1`: `2` \[LongRightArrow] `3`", traceform[k], HoldForm[expr], HoldForm[res] ];
	  Write[k, Parallel`Client`ReplyPacket[res]]; (* send reply *)
	  If[ aborted, Abort[] ]; (* only after sending it *)
    ]]]

(* this is a packet type that expects a reply *)

registerReplyHead[Parallel`Client`CallBackPacket]


SetAttributes[{varDef,downDef}, {Protected,Locked}]

End[]

Protect[ SetSharedVariable, SetSharedFunction, UnsetShared, SendBack ]

EndPackage[]
