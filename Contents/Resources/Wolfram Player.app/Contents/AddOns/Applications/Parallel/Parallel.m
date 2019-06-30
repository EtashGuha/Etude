(* :Title: Parallel.m -- basic parallel support *)

(* :Context: Parallel`Parallel` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   Basic process control for parallel evaluation of Mathematica expressions.
 *)

(* :Package Version: 3.0  *)

(* :Mathematica Version: 8 *)

(* :History:
   3.0 for Mathematica 7.0, PCT 3.0, split off concurrency into separate file
   2.1 for Mathematica 5.2 and 6.0, workbench support
   2.0 update for PCT V1, tested on Mathematica 4.2 and 5
   1.0 first released version.
*)

BeginPackage["Parallel`Parallel`"]

System`ParallelEvaluate
System`DistributeDefinitions
System`DistributedContexts
System`ParallelNeeds


BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

`ParallelDispatch::usage = "ParallelDispatch[cmds, kernels] sends each of the cmdi to the corresponding kerneli and returns the list of results.
	ParallelDispatch[cmds] is equivalent to ParallelDispatch[cmds, Kernels[]]."
SyntaxInformation[ParallelDispatch] = { "ArgumentsPattern" -> {_, _.} }

`$DistributedDefinitions::usage = "$DistributedDefinitions is the list of all symbols whose definitions have been distributed."

ClearDistributedDefinitions::usage = "ClearDistributedDefinitions[] forgets about all distributed definitions."
SyntaxInformation[ClearDistributedDefinitions] = { "ArgumentsPattern" -> {} }

`Send::usage = "Send[kernel,cmd] sends cmd for evaluation to the given remote kernel or list of kernels."
SyntaxInformation[Send] = { "ArgumentsPattern" -> {_, _} }

`Receive::usage = "Receive[kernel] waits for a result from the given kernel or list of kernels and returns it.
	Receive[kernel, h] wraps the result in h before returning it."
SyntaxInformation[Receive] = { "ArgumentsPattern" -> {_, _.} }

`ReceiveIfReady::usage = "ReceiveIfReady[kernel] returns a waiting result and $NotReady if nothing is waiting.
	ReceiveIfReady[kernel, h] wraps the result in h before returning it."
SyntaxInformation[ReceiveIfReady] = { "ArgumentsPattern" -> {_, _.} }

`$NotReady::usage = "$NotReady is returned by ReceiveIfReady if no result is available."

EndPackage[]


(* semi-hidden methods, aka protected *)

BeginPackage["Parallel`Protected`"]

`identity::usage = "identity[___] is an identity that can handle Sequence and other ugly things."

`DistDefs::usage = "DistDefs[expr, exc, ctxts] distributes definitions for expr, except the listed symbols in exc."

`addBadContext::usage = "addBadContext[ctx] adds ctx to the list of contexts not to distribute."

`AutoDistribute::usage = "AutoDistribute[msghead, expr, opt, exceptions] distributes all defininitions needed for expr."
`DistOptCheck

`$ExcludedContexts::usage = "$ExcludedContexts is the list of contexts to always exclude from DistributeDefinitions."

`addDistributedSymbol
`removeDistributedSymbol

EndPackage[]

(* options *)

Options[DistributeDefinitions] = {
	DistributedContexts -> Automatic
}

Options[ParallelEvaluate] = {
	DistributedContexts :> $DistributedContexts
}

(* messages *)

General::nopar = "No parallel kernels available; proceeding with sequential evaluation."
General::value = "Option value `1` is outside allowed range; setting not changed." (*for SystemOptions*)
General::opttaf = "Value of option `1` -> `2` is neither one of Full, Automatic, None, nor a context or a list of contexts."

ParallelDispatch::toomany = "The number of commands `1` is larger than the number of kernels `2`."
DistributeDefinitions::ctx = "No context matching `1` found."


(* master side of symbols in Parallel`Client` *)
Parallel`Client`makeDefinitions


Begin["`Private`"] (*****************************************************)

`$PackageVersion = 3.0;
`$thisFile = $InputFileName

Needs["Parallel`Kernels`"] (* a protected base class, really *)

(* some of its internal features *)
holdCompound = Parallel`Client`HoldCompound

protected = Unprotect[MessagePacket, TextPacket]

(* busy waiting pause is through SetSystemOptions["ParallelOptions" -> "BusyWait" -> ]
   allowed values are nonnegative numbers; reflected in Parallel`Settings`$BusyWait *)


(* Send/Receive stuff *)

SetAttributes[Send, {HoldRest,SequenceHold}]

Send[kernels:{___kernel}, expr_] := Function[e, Send[e, expr]] /@ kernels
Send[k_kernel, expr_] := Catch[send[k, expr], dead,
	(Message[Kernels::rdead, k]; CloseError[k]; $Failed)& ]

(* an identity that can handle Sequence and other ugly things *)
identity[expr___] := expr

(* translate Kernels`$notReady into Parallel`Developer`$NotReady *)

ReceiveIfReady[k_kernel, h_:identity] :=
	Catch[ Replace[ receive[k, False], {HoldComplete[e___] :> h[e], $notReady -> $NotReady} ], dead,
		(Message[Kernels::rdead, k]; CloseError[k]; $Failed)& ]

(* single kernel. If no evaluation is pending, this will return $Failed *)

Receive[k_kernel, h_:identity] :=
	Catch[ Replace[receive[k, True], HoldComplete[e___] :> h[e]], dead,
		(Message[Kernels::rdead, k]; CloseError[k]; $Failed)& ]

(* an ex-kernel, marked as $Failed *)

Receive[$Failed, ___] := $Failed

(* A non-blocking version of Receive for several kernels.
We replace every kernel object in the list by the result of an evaluation returned by it until none are left.
If a kernel does not have any evaluation pending, the result for it will be $Failed and receive[] will complain.
*)

(** condition handling:
	Abort: checks for abort, aborts all subkernels, propagates abort
	we have to abort ALL kernels
**)

Receive[kernels:{___kernel}, h_:identity] :=
  With[{keys=Range[Length[kernels]]},
    Module[{tasks=AssociationThread[keys, kernels], pending=keys, progress=False},
      CheckAbort[
    	While[ Length[pending]>0,
        	Scan[Function[key,
        		With[{res=ReceiveIfReady[tasks[key], HoldComplete]},
        			If[res=!=$NotReady,
        				tasks[key] = res;
        				pending = DeleteCases[pending, key, {1}, 1];
        				progress = True;
        			]
        		]
        	], pending];
        	If[ progress, progress=False, Pause[Parallel`Settings`$BusyWait] ];
        ];
        h @@@ (tasks /@ keys) (* replace the HoldComplete wrappers. Is immune to $Failed *)
        , AbortKernels[]; Abort[]
      ]
    ]
  ]

(* junk in the list, possible failed kernels, tread carefully *)

Receive[junk_List, h_:identity] := Function[e, Receive[e, h]] /@ junk


(* parallel evaluation *)

SetAttributes[ParallelEvaluate, {HoldFirst,SequenceHold}]

(* translate kernel IDs to kernels *)

ParallelEvaluate[cmd_, ids:(_Integer|{__Integer}), o:OptionsPattern[]] := ParallelEvaluate[cmd, KernelFromID[ids], o]

(* second argument is a kernel or a list of kernels *)

ParallelEvaluate[cmd_, k:(_kernel|{___kernel}), OptionsPattern[]] := (
	Parallel`Protected`AutoDistribute[cmd, ParallelEvaluate, OptionValue[DistributedContexts]]; (* send definitions *)
	Receive[Send[k, cmd]]
)

(* default: use all kernels *)

ParallelEvaluate[cmd_, o:OptionsPattern[]] := ParallelEvaluate[cmd, Kernels[], o]

(* distribute to processors *)

(** condition handling:
	Abort: checks for abort, aborts all subkernels, propagates abort
**)

SetAttributes[ParallelDispatch, {HoldFirst,SequenceHold}]

ParallelDispatch[cmds_] := ParallelDispatch[cmds, Kernels[]]

ParallelDispatch[h_[cmds___], kernels:List[k___]]/; Length[HoldComplete[cmds]]<=Length[kernels] :=
	h @@ Receive[ Inner[Send, Take[HoldComplete[k], Length[HoldComplete[cmds]]], HoldComplete[cmds], List] ]

ParallelDispatch[h_[cmds___], kernels_List]/; Length[HoldComplete[cmds]]>Length[kernels] && Message[ParallelDispatch::toomany, Length[HoldComplete[cmds]], Length[kernels]] := Null


(* DistributeDefinitions *)

(* list of all parallelized symbols and hash of their distributed values *)

(* a version of SameQ that does not evaluate its arguments *)

SetAttributes[sameQU, HoldAll]
sameQU[a_, b_] := SameQ[Unevaluated[a], Unevaluated[b]]

`$parallelized = Hold[];
SetAttributes[{addDistributedSymbol, removeDistributedSymbol}, HoldAll]
addDistributedSymbol[s_] := ($parallelized = Union[$parallelized, Hold[s], SameTest->sameQU])
removeDistributedSymbol[s_] := ($parallelized = Complement[$parallelized, Hold[s], SameTest->sameQU])

`$distributedDefs[_] := {}

(* a symbol without any definitions *)

undefQ[vals_List] := Total[Length /@ vals[[All,2]]]===0


(* updatedDefs[HoldForm[s] -> vals, incremental:True|False] returns list of those values that need updates *)

updatedDefs[arg_] := updatedDefs[arg, True] (* incremental by default *)

(* a new symbol, not in the cache *)

updatedDefs[s:HoldForm[ss_Symbol] -> vals_List, incr_] /; !MemberQ[$parallelized, Unevaluated[ss]] :=
  If[incr && undefQ[vals],
  	  (* do nothing, there are no definitions and we are incremental *)
  	  Sequence@@{}
  	, (* else: not incremental or with a definition *)
	addDistributedSymbol[ss];
	updatedDefs[s -> vals, False] (* a new one is not incremental *)
  ]

(* not incremental; return all values *)

updatedDefs[s:HoldForm[_Symbol] -> vals_List, False] := ($distributedDefs[s] = vals; s -> vals)

(* incremental, no change: nothing to do *)

updatedDefs[s:HoldForm[_Symbol] -> vals_List, True] /; $distributedDefs[s] === vals := Sequence@@{}

(* incremental, find out which value lists to include *)

updatedDefs[s:HoldForm[_Symbol] -> vals_List, True] :=
	With[{oldvals=$distributedDefs[s]},
		$distributedDefs[s] = vals; (* update cache *)
		s -> Cases[vals, (v_ -> l_) /; (v /. oldvals) =!= l]
	]

(* apply to a definition list that may include several symbols *)

updatedDefs[Language`DefinitionList[args___], incr_] :=
	Language`DefinitionList@@Evaluate[Function[e,updatedDefs[e,incr]]/@{args}]

(* catchall, may happen for removed symbols *)

updatedDefs[_, _] := Sequence@@{}


(* distribute definitions, except for listed symbols *)

(* is something a shared variable? then do not distribute its definition *)

SetAttributes[{sharedQ}, HoldFirst]
sharedQ[s_] := MemberQ[$SharedVariables, Hold[s]] || MemberQ[$SharedFunctions, Hold[s]]

(* add back missing backticks *)

With[{tick = "`"},
	addtick[s_String] := If[StringTake[s, -1] =!= tick, StringJoin[s, tick], s];
	deltick[s_String] := If[StringTake[s, -1] === tick, StringDrop[s, -1], s];
]

$ExcludedContexts :=`$excludedContexts

(* pick up initial value from sysinit.m (11.3 and later), or the option value if not present *)

If[ListQ[Parallel`Static`$SystemContexts],
	$excludedContexts = Parallel`Static`$SystemContexts,
	$excludedContexts = addtick /@ ("ExcludedContexts" /. Options[Language`ExtendedFullDefinition])
]

(* contexts to exclude, and a list of contexts to include (remove from the exclude list) *)

`$additionalExcludes = {"System`", "JLink`", "com`", "org`", "java`", "LinearAlgebra`",
	"Parallel`", "SubKernels`", "WSTP`", "CURLLink`", "StartUp`", "EntityFramework`", "GeneralUtilities`",
	"PacletManager`", "TextSearch`"}
`$additionalIncludes = {"Global`", "Utilities`", "FE`"}

(* reasons to include contexts:
	Global`		make sure it is included, as documented
	Utilities`	somehow this (empty) context ended up in the exlude list, but there is no good reason
	FE`			this is used by DynamicModule[] for its variables (but only inside Dynamic)
 *)

(* make it prefix-free *)

cand = Sort[Join[$excludedContexts, $additionalExcludes]]
$excludedContexts = {}

While[Length[cand] > 0,
  With[{el = First[cand]},
  	AppendTo[$excludedContexts, el];
  	cand = Select[Rest[cand], StringFreeQ[StartOfString ~~ el]];
]];

$excludedContexts = Complement[$excludedContexts, $additionalIncludes]

addBadContext[s_] := ($excludedContexts = DeleteDuplicates[Append[$excludedContexts, s]])

contextQ[s_String] := StringMatchQ[s, "*`"]

SetAttributes[allContexts, HoldFirst]

allContexts[expr_] := 
	Union[Flatten[ Reap[Scan[Function[l, If[Head[Unevaluated[l]] === Symbol, Sow[Context[l]]], {HoldFirst}],
							 Unevaluated[expr], {-1}, Heads -> True]][[2]] ]]

(* exclude certain attributes of symbols *)

fixAttrs[dl_Language`DefinitionList] := 
 Replace[dl, (s_ -> {a___, Attributes -> {aa___, Temporary, oo___}, 
      o___}) :> (s -> {a, Attributes -> {aa, oo}, o}), {1}]


(* DistDefs[expr, exc, ctxts] internal workhorse *)

SetAttributes[DistDefs, HoldFirst]

DistDefs[expr_] := DistDefs[expr, {}] (* no exceptions, by default *)
DistDefs[expr_, exc_] := DistDefs[expr, exc, Full] (* all contexts, by default *)

(* the third argument is a list of matching contexts, or "Full" *)

DistDefs[expr_, exc_List, ctxts_] := Module[{updates},

	updates = Language`ExtendedFullDefinition[expr, "ExcludedContexts"->$ExcludedContexts];

	(* kick out listed exceptions *)
	updates = DeleteCases[updates, ss_HoldForm/;MemberQ[exc, ss] -> _, 1];
	(* kick out foreign symbols, not in the contexts given *)
	If[ ListQ[ctxts], With[{ctxpat = ctxts~~___},
		updates = Select[updates, MatchQ[#, HoldForm[s_]/;StringMatchQ[Context[s], ctxpat] -> _]&]
	]];
	(* kick out shared variables *)
	updates = DeleteCases[updates, HoldForm[s_]/;sharedQ[s] -> _, 1];

	(* unwanted definition parts, such as attributes *)
	updates = fixAttrs[updates];

	(* update cache and filter unchanged ones *)
	updates = updatedDefs[updates, True];
	distributeDefs[updates]; (* update running kernels *)
	List @@ First /@ updates
]


SetAttributes[DistributeDefinitions, HoldAll]

(* TODO: DistributedContexts option *)

(* whole contexts; return list of contexts handled *)

DistributeDefinitions[c_String, opts:OptionsPattern[]]/; Length[Contexts[c]]>0 :=
    (DistributeDefinitions[Evaluate[Thread[ToExpression[#,InputForm,Hold]& /@ Names[#<>"*"], Hold]],opts];#)& /@ Contexts[c]

DistributeDefinitions[c_String, opts:OptionsPattern[]]/; Length[Contexts[c]]==0 && Message[DistributeDefinitions::ctx, c] := Null

(* arbitrary expressions *)

DistributeDefinitions[expr_, OptionsPattern[]] := Module[{optval},
	optval = DistOptCheck[DistributeDefinitions, OptionValue[DistributedContexts]];
	If[optval===Automatic, optval = allContexts[expr]];
	DistDefs[expr, {}, optval]
]

(* a sequence of symbols is now just a special case of an arbitrary expression *)

DistributeDefinitions[exprs__, Longest[opts:OptionsPattern[]]] :=
	DistributeDefinitions[{exprs}, opts]


(* do the actual work; [actually, there is only one optional argument; if missing, means all running kernels] *)

distributeDefs[Language`DefinitionList[], ___] := Null (* shortcut *)


SetAttributes[Parallel`Client`makeDefinitions, HoldAll] (* master-side only *)

distributeDefs[c_Language`DefinitionList, ks___] :=
If[Parallel`Debug`Private`ExportEnvironmentWrapper===Identity, (* shortcut *)
	With[{defs = Parallel`Client`makeDefinitions[c]},
		kernelEvaluate[defs, ks]
	]
 , (* else wrap it and prepare for unwrwapping at client side *)
   With[{wrapped = Parallel`Debug`Private`ExportEnvironmentWrapper[Hold[c]]},
  	With[{defs = Parallel`Client`makeDefinitions[First[wrapped]]},
  		kernelEvaluate[defs, ks]
   ]]
]

(* Initializer for new kernels (but not running ones), is called in Parallel`Kernels`ConnectKernel *)

initDistributedDefinitions[ks_] := If[ Length[$parallelized]>0,
	distributeDefs[ Language`DefinitionList@@List@@Replace[$parallelized, s_ :> (HoldForm[s] -> $distributedDefs[HoldForm[s]]),1], ks]
]


(* read-only public version of list of distributed symbols *)

$DistributedDefinitions := List @@ Hold /@ $parallelized

(* debug functions *)

ClearDistributedDefinitions[] := (
	(* wipe them on the subkernels *)
	With[{$parallelized=$parallelized}, kernelEvaluate[{Unprotect @@ #, ClearAll @@ #}& @ $parallelized]];
	(* forget them *)
	$parallelized = Hold[];
	Clear[$distributedDefs];
	$distributedDefs[_] := {};
)

(* reset *)

Parallel`Kernels`Private`registerClearCode[ ClearDistributedDefinitions[] ]


(* Read and remember packages on subkernels *)

ParallelNeeds[s_String] := (AddInitCode[holdCompound[Needs[s]]]; addBadContext[s]; Null)

ParallelNeeds[s__String] := ParallelNeeds[{s}]

ParallelNeeds[s:{__String}] := (ParallelNeeds /@ s; Null)


(* automatic distribution of definitions *)

(* valid settings for DistributedContexts option:
	a context
	a list of contexts
	None|False implying {}
	Full|True implying any context other than a system context
	Automatic, meaning any context occurring in expr
    as well as subcontexts of the ones listed
*)

DistOptCheck[msghead_, None|False] = {}
DistOptCheck[msghead_, True|Full|All] = Full
DistOptCheck[msghead_, Automatic] = Automatic

DistOptCheck[msghead_, opt_String?contextQ] := {opt}
DistOptCheck[msghead_, {cts___String}/;And@@contextQ/@{cts}] := {cts}

DistOptCheck[msghead_, opt_] := (Message[msghead::opttaf, DistributedContexts, opt]; None)

(* second argument is the head for error messages, third a DistributedContexts option value, forth
   a list of symbols to exclude, wrapped in HoldForm *)

SetAttributes[{AutoDistribute,autoDistribute}, HoldFirst]

AutoDistribute[expr_, msghead_, opt_, args___] := autoDistribute[expr, DistOptCheck[msghead, opt], args]

(* handle option value: either a list of contexts or one of the keywords Full|Automatic *)

autoDistribute[expr_, opt_] := autoDistribute[expr, opt, {}]

autoDistribute[expr_, {}, exc_List] := Null (* no contexts: nothing to distribute *)

autoDistribute[expr_, Full, exc_List] := (DistDefs[expr, exc, Full]; Null) (* everything *)

autoDistribute[expr_, ctxts_List, exc_List] := (DistDefs[expr, exc, ctxts]; Null)

autoDistribute[expr_, Automatic, exc_List] := (DistDefs[expr, exc, allContexts[expr]]; Null) (* all occurring contexts *)


(* misc junk *)

Protect[Evaluate[protected]]


End[]

Protect[ Send, Receive, ReceiveIfReady, $NotReady ]
Protect[ ParallelEvaluate ]
Protect[ DistributeDefinitions,ParallelNeeds,$DistributedDefinitions,DistributedContexts,$ExcludedContexts ]

EndPackage[]
