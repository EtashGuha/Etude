(* :Title: Evaluate.m -- automatic parallelization  *)

(* :Context: Parallel`Evaluate` (empty context) *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   data parallelism:
   Basic overloading for easily parallelized functions, using ParallelCombine
   and possible other techniques
 *)

(* :Package Version: 3.0  *)

(* :Mathematica Version: 7 *)


BeginPackage["Parallel`Evaluate`"]

System`Parallelize


BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

`$ParallelCommands::usage = "$ParallelCommands is a list of functions that can be automatically parallelized with Parallelize[]."

EndPackage[]


BeginPackage["Parallel`Protected`"]


EndPackage[]


Options[Parallelize] = {
		DistributedContexts :> $Context,
		Method -> Automatic
}


Begin["`Private`"] (*****************************************************)

`$PackageVersion = 2.0;
`$thisFile = $InputFileName

Needs["Parallel`Parallel`"]


`$seqWarning = True; (* dynamically bound *)

SetAttributes[Parallelize, HoldFirst]

(* handle methods and apply parallelizers; freeze default Method option value, too *)

Parallelize[expr_, opts:OptionsPattern[]] :=
Module[{res},
    tryRelaunch[];

	(* freeze option settings *)
	fopts = Sequence @@ { Method->OptionValue[Method],
		DistributedContexts->Parallel`Protected`DistOptCheck[Parallelize,OptionValue[DistributedContexts]] };

	(* data parallelism with ParallelCombine and friends *)
	res = tryCombine[expr, fopts];
	If[ Head[res] =!= tryCombine, Return[res] ];

	(* expr traversal *)
	Block[{$seqWarning = False},
		res = wrapAround[expr, fopts];
		If[ Head[res] =!= wrapAround, Return[res] ];
	]; (* seqWarning *)

	(* silent failure for trivial things *)
	res = silentFail[expr];
	If[ Head[res] =!= silentFail, Return[res] ];

	(* finally give up and complain *)
	If[ $seqWarning, Message[Parallelize::nopar1, HoldForm[expr]] ];
	expr
]


(*** data parallelism ***)

SetAttributes[tryCombine, HoldFirst]


(* upvalues for parallelizable commands *)
(* note that we cannot always perform a full argument check and may therefore
   get repeated msgs on the parallel kernels *)

$autoCommands = {}
$protected = {}

$ParallelCommands := Sort[$autoCommands] (* read only *)


addCommands[{cmds___}] := (
	$protected    = Join[ $protected, Unprotect[cmds] ]; (* remember those that need to be protected again *)
	$autoCommands = Join[ $autoCommands, {cmds} ];
	SetAttributes[{cmds}, ReadProtected]; (* to hide the ugly upvalue *)
)

(* in general, we should attempt to evaluate f in tryCombine[f[args],...]; args are looked at later in ParallelCombine *)

tryCombine[f_[args__], rest___] := With[{ef=f}, tryCombine[ef[args], rest] /; ef=!=Unevaluated[f] ]


(* commands that take the structure as first argument and preserve the structure *)
(* we handle all cases with at least two arguments *)

commands = {Cases, Select}
addCommands[commands]

Scan[Function[cmd,
    cmd/: tryCombine[cmd[str_, params__], opts___] := ParallelCombine[Function[e, cmd[e, params]], str, opts]
    ], commands]


(* structure not preserved: needs custom combiner *)
(* we handle all cases with at least two arguments *)

commands = {Count, MemberQ, FreeQ}
combs = {Plus, Or, And}
addCommands[commands]

Apply[Function[{cmd,comb},
    cmd/: tryCombine[cmd[str_, params__], opts___] := ParallelCombine[Function[e, cmd[e, params]], str, comb, opts]
    ], Transpose[{commands,combs}], {1}]


(* special cases *)

addCommands[{Map, MapIndexed, MapThread, Scan, Apply, Outer, Inner, Dot, Through}]

(** Map/: tryCombine[Map[f_, args__], opts:OptionsPattern[]] := ParallelMap[f, args, opts] **)
Map/: tryCombine[Map[f_, str_], opts___] := ParallelCombine[Function[e,Map[f, Unevaluated[e]]], str, opts]
Map/: tryCombine[Map[f_, str_, level_?goodLevel], opts___] := ParallelCombine[Function[e,Map[f,Unevaluated[e],level]], str, opts]

(* scan: throw away the result (Null) *)
Scan/: tryCombine[Scan[f_, str_], opts___] := ParallelCombine[Function[e,Scan[f, Unevaluated[e]]], str, Null&, opts]
Scan/: tryCombine[Scan[f_, str_, level_?goodLevel], opts___] := ParallelCombine[Function[e,Scan[f,Unevaluated[e],level]], str, Null&, opts]

(* MapIndexed, only default level 1, eval str only once; hack needed for nonlists (because of Transpose) *)

MapIndexed/: tryCombine[MapIndexed[f_, str_], opts___] := mapIndexedHelper[f, str, opts]

mapIndexedHelper[f_, estr_List, opts___] :=
	ParallelCombine[Function[trsp,Apply[f,trsp,{1}]], Transpose[{estr, List /@ Range[Length[estr]]}], opts]

mapIndexedHelper[f_, h_[elems___], opts___] := h @@ mapIndexedHelper[f, {elems}, opts]

(* one-argument case of MapThread. evaluate f, arguments (once), works only for lists of depth at least 2 *)

MapThread/: tryCombine[MapThread[f_, str_], opts___] :=
	Module[{estr}, With[{ef=f},ParallelCombine[Function[trsp,MapThread[ef,Transpose[trsp]]], Transpose[estr], opts]]/; Head[estr=str]===List&&ArrayDepth[estr]>=2]

(* Apply: the default level of 0 cannot be parallelized *)

Apply/: tryCombine[Apply[f_, str_, level_?goodLevel], opts___] := ParallelCombine[Function[s,Apply[f,Unevaluated[s],level]], str, opts]

(* Outer: evaluate all arguments *)

Outer/: tryCombine[Outer[f_, str1_, str2__], opts___] :=
	With[{estr2=str2}, ParallelCombine[Function[s,Outer[f,s,estr2]], str1, opts]]

(* Inner and Dot work only if str1 has at least rank 2 *)
(* due to contradictory info on the role of the fifth argument of Inner
   we do not handle this case *)

Inner/: tryCombine[Inner[f_, str1_, str2_, g_:Plus], opts___]/;ArrayDepth[str1]>1 :=
	With[{estr2=str2}, ParallelCombine[Function[s,Inner[f,s,estr2,g]], str1, opts]] (* eval str2 *)

Dot/: tryCombine[Dot[str_, params__], opts___]/;ArrayDepth[str]>1 :=
	With[{eparams=params}, ParallelCombine[Function[s,Dot[s,eparams]], str, opts]] (* eval args *)

(* for rank1 we have to be a bit clever *)
(* we have to take care of Flat functions ourselves, as Plus[x] might go away *)

Inner/: tryCombine[Inner[f_, str1_, str2_, g_Symbol:Plus], opts___]/;ArrayDepth[str1]==1 && MemberQ[Attributes[g],Flat] :=
	ParallelCombine[Function[e,g@@f@@@e],Transpose[{str1,str2}], g, opts]

(* here we hope that ParallelCombine figures out the combiner to use *)

Inner/: tryCombine[Inner[f_, str1_, str2_, g_], opts___]/;ArrayDepth[str1]==1 :=
	ParallelCombine[Function[e,g@@f@@@e],Transpose[{str1,str2}], opts]

Dot/: tryCombine[Dot[str1_, str2_], opts___]/;ArrayDepth[str1]==1 := tryCombine[Inner[Times,str1,str2], opts]

(* for serveral arguments, use associativity *)

Dot/: tryCombine[Dot[str1_, str2_, str__], opts___]/;ArrayDepth[str1]==1 :=
	With[{d1 = tryCombine[Dot[str1, str2], opts]}, tryCombine[Dot[d1, str], opts] ]

(* one argument form *)
(* evaluate the composite function head (once) *)
(* the HoldFirst/Unevaluated mess is due to vanishing (OneIdentity) functions *)

Through/: tryCombine[Through[funcs_[args___]], opts___] :=
  Module[{efuncs=funcs},
	With[{eargs=args}, ParallelCombine[Function[f,Through[Unevaluated[f[eargs]]],HoldFirst], efuncs, opts]]/;
	!AtomQ[efuncs]
  ]

(* with restriction on head. have to perform the head check ourselves *)
Through/: tryCombine[Through[funcs_[args___], h_], opts___] :=
  Module[{efuncs=funcs},
	With[{eargs=args, eh=h}, ParallelCombine[Function[f,Through[Unevaluated[f[eargs]],eh],HoldFirst], efuncs, opts]]/;
	!AtomQ[efuncs] && Head[efuncs]===h
  ]

(* fall through: nonmatching, nothing to do, but in parallel :-) *)
Through/: tryCombine[Through[funcs_[args___], h_], opts___] :=
  Module[{efuncs=funcs},
	funcs[args]/;
	!AtomQ[efuncs]
  ]

(* Pick; allow for non-list first argument (after evaluation) *)

addCommands[{Pick}]
Pick/: tryCombine[Pick[list_, sel_, patt_:True], opts___] :=
	Module[{elist=list, esel=sel}, With[{h=Head[elist], epatt=patt},
		ParallelCombine[Function[ll, Function[{e1,e2},Pick[h @@ e1, e2, epatt]] @@ Transpose[ll]], Transpose[{List@@elist, esel}]]/;
		Length[elist]===Length[esel] (* leave the outer Module intact, it is needed for the /; hack *)
	]]


(* iterators *)
(* we handle all cases with at least one argument, except for Array, where we need two *)

addCommands[{Table, Sum, Product, Do, Array}]

Table/:   tryCombine[Table[args__], opts___]     := ParallelTable[args, opts]
Sum/:     tryCombine[Sum[args__], opts___]       := ParallelSum[args, opts]
Product/: tryCombine[Product[args__], opts___]   := ParallelProduct[args, opts]
Do/:      tryCombine[Do[args__], opts___]        := ParallelDo[args, opts]
Array/:   tryCombine[Array[f_, args__], opts___] := ParallelArray[f, args, opts]


(* List itself: just evaluate the arguments in parallel. Note that it is locked *)

AppendTo[$autoCommands, List]

tryCombine[list_List, opts___] := ParallelCombine[Identity, Unevaluated[list], opts]


(* listable functions *)

(* one argument case. evaluate the list (once!), except for implicit Range and Table cases (which return lists) *)

tryCombine[h_Symbol[r:(Range|Table)[___]], opts___]/; MemberQ[Attributes[h], Listable] :=
	ParallelCombine[h, r, opts]
tryCombine[h_Symbol[arg_], opts___]/; MemberQ[Attributes[h], Listable] :=
	Module[{earg}, ParallelCombine[h, earg, opts]/; Head[earg=arg]===List]

(* if argument is not a list, let the catchall at the end produce the nopar1 messages *)

(* associative operations; do not evaluate the arguments, but there should be at least 3 *)
(* Union and Intersection may have options; so, let it use the default combiner *)

tryCombine[expr:(h_Symbol[_,_,_,___,OptionsPattern[]]), opts___]/; MemberQ[Attributes[h], Flat] :=
	ParallelCombine[Identity, Unevaluated[expr], opts]

(* TODO: applying a flat function to a data parallel operation *)


Protect[Evaluate[$protected]] (* restore protection *)


(* wrap parallelize around parts of expr *)

SetAttributes[wrapAround, HoldFirst]

wrapAround[es_CompoundExpression, opts___] := Function[{e}, Parallelize[e, opts], {HoldFirst}] /@ Unevaluated[es]

wrapAround[HoldPattern[s_ = e_], opts___] := (s = Parallelize[e, opts])

(* partial evaluation *)

wrapAround[Apply[f_, arg_], opts___] := Module[{earg},
	Replace[earg, h_[args___] :> Parallelize[f[args], opts]]/; !AtomQ[earg=arg]
]


(* silent failure; careful not to evaluate anything during pattern matching *)

SetAttributes[silentFail, HoldFirst]

silentFail[Null] := Null

(* manifest numerics *)
silentFail[r_/;NumberQ[Unevaluated[r]]] := r

(* delayed assignments *)
silentFail[ass:(_ := _)] := ass



End[]

Protect[ $ParallelCommands ]
Protect[ Parallelize ]

EndPackage[]
