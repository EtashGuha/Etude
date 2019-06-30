(* :Title: Combine.m -- data parallelism *)

(* :Context: Parallel`Combine` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   partition lists and work on the pieces in parallel
 *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 7 *)


BeginPackage["Parallel`Combine`"]

System`ParallelCombine

System`ParallelMap
System`ParallelTable
System`ParallelSum
System`ParallelProduct
System`ParallelDo
System`ParallelArray

System`$DistributedContexts


BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

`PartitionSelect::usage = "PartitionSelect[list,{i,n}] selects the ith of n partitions of list."

EndPackage[]

BeginPackage["Parallel`Protected`"]

`goodLevel

`MethOptCheck

EndPackage[]

(* set options *)

(Options[#] = { DistributedContexts :> $DistributedContexts, Method -> Automatic })& /@ {
	ParallelCombine,
	ParallelMap, ParallelTable, ParallelSum, ParallelProduct, ParallelDo, ParallelArray
}

(* messages *)

General::nopar1 = "`1` cannot be parallelized; proceeding with sequential evaluation."
General::meth = "`1` is not a valid Method option specification for `2`; using `3`."


Begin["`Private`"] (*****************************************************)

`$PackageVersion = 2.0;
`$thisFile = $InputFileName

Needs["Parallel`Parallel`"]
Needs["Parallel`Kernels`"]

(* some of its internal features *)
holdCompound = Parallel`Client`HoldCompound


If[ !ValueQ[$DistributedContexts], $DistributedContexts := $Context ] (* default option value *)


(* valid settings for Method option:
	"CoarsestGrained" | "FinestGrained" | Automatic
	"EvaluationsPerKernel" -> n
	"ItemsPerEvaluation" -> n
*)

MethOptCheck[msghead_, Automatic] = Automatic
MethOptCheck[msghead_, s:("CoarsestGrained" | "FinestGrained")] = s

MethOptCheck[msghead_, s:(("EvaluationsPerKernel"|"ItemsPerEvaluation") -> n_Integer?Positive)] := s

MethOptCheck[msghead_, opt_] := (Message[msghead::meth, opt, msghead, Automatic]; Automatic)


(* automatic setting of Method option. should return a sensible batchsize, given items per kernel *)

autoSize[itemsperkernel_] := Ceiling[itemsperkernel^(2/3)]

(* the two ref parameters batches and batchsize are set according to the
   option value; at least one of them will be set to a positive integer,
   with batches taking preference *)

SetAttributes[grokMethodOption,HoldAll]

(* an item count of Indeterminate means to just create one batch of unknown size *)

grokMethodOption[cmd_, Indeterminate, nk_, batches_, batchsize_, optval_] := (
	batches=1;
)

grokMethodOption[cmd_, items_, nk_, batches_, batchsize_, optval_] :=
With[{itemsperkernel=Max[items,1]/nk},
    Replace[optval, {
    	"CoarsestGrained" :> (batches = 1),
    	"FinestGrained" :> (batchsize = 1),
    	("EvaluationsPerKernel"->e_Integer?Positive) :> (batches = Min[e,Ceiling[itemsperkernel]]),
    	("ItemsPerEvaluation"->m_Integer?Positive) :> (batchsize = Min[m,Ceiling[itemsperkernel]]),
    	Automatic :> (batchsize = autoSize[itemsperkernel]),
    	err_ :> (Message[cmd::meth, err, cmd, Automatic]; batchsize = autoSize[itemsperkernel])
    }];
    If[ !(IntegerQ[batchsize] && batchsize>0 || IntegerQ[batches] && batches>0),
    	Message[cmd::meth, optval, cmd, "\"CoarsestGrained\""]; (* any other wrong values *)
    	batches = 1;
    ]
]


(* create batch sizes that sum up to n according to ProcessorSpeed to minimize longest time taken *)

makeBatches[n_Integer, {}] := $Failed (* no sequential fallback *)

makeBatches[n_Integer?Positive, kernels_List] :=
    Module[ {sizes, cand, speeds = KernelSpeed /@ kernels},
        sizes = Floor[n (speeds/Total[speeds])]; (* first guess; we know total is > 0 *)
        While[n - Total[sizes] > 0, (* need one more *)
        	cand = Ordering[(sizes + 1)/speeds, 1];  (* index of smallest one, if given one more *)
        	sizes[[cand]] += 1;
		];
        sizes
    ]

makeBatches[n_Integer?NonPositive, _] := {} (* allow for negative n: treat as 0 *)

makeBatches[n_] := makeBatches[n, Kernels[]]

(* general split routine; at least one of batches0 and batchsize0 needs a value *)

makeSizes[0, ___] := {}

makeSizes[Indeterminate, nk_, batches0_, batchsize0_] :=
Module[{sizes, batches=batches0, batchsize=batchsize0, jobs},
    If[batches===1,
    	(* single batch per kernel *)
    	sizes = Table[Indeterminate, {nk}]
    , (* not possible *)
    	Null
	];
	sizes
]

makeSizes[items_, nk_, batches0_, batchsize0_] :=
Module[{sizes, batches=batches0, batchsize=batchsize0, jobs},
    If[batches===1,
    	(* single batch per kernel *)
    	sizes = makeBatches[items]
    , (* else figure out total number of jobs and sizes *)
    	If[!IntegerQ[batchsize], batchsize=Ceiling[items/nk/batches]]; (* batches is given, convert to size *)
    	jobs = Ceiling[items/batchsize/nk]*nk; (* total number of evaluations *)
    	batchsize = Min[batchsize, Ceiling[items/jobs]]; (* limit *)
    	With[{nless = jobs*batchsize-items},
			sizes = Table[batchsize,{jobs-nless}];
			If[nless>0&&batchsize>1, sizes=Join[sizes, Table[batchsize-1,{nless}]]]; (* avoid empty jobs *)
    	]
	];
	sizes
]

(* instanciate current directory (might be different on subkernels)
   filenames will therefore always have at least 2 arguments *)

fileNames[opts:OptionsPattern[]] := fileNames["*", opts]

fileNames[form_, opts:OptionsPattern[]] := fileNames[form, Directory[], opts]

PartitionSelect[list_, {i_, n_}] :=
	Extract[list, Position[Mod[Hash[#], n] & /@ list, i - 1, {1}]]

(* there is no internal version of DistributeDefinitions... *)

AppendInitCode[holdCompound[PartitionSelect[list_, {i_, n_}] := Extract[list, Position[Mod[Hash[#], n] & /@ list, i - 1, {1}]]]]

 (* monstly needed on subkernels! which context to use? *)
 
(* ParallelCombine *)

(* the main function is HoldAll for recognizing special cases *)

SetAttributes[ParallelCombine, HoldAllComplete]

(* honor Evaluate[] to bypass special treatment *)

ParallelCombine[func_, Evaluate[expr_], rest___] := parallelCombine[func, expr, rest]

(* recognize and freeze iterators *)

ParallelCombine[func_, r:(Range|Table)[___], rest___] := parallelCombine[func, Unevaluated[r], rest]

(* FileNames is also a form of iterator *)

ParallelCombine[func_, r:FileNames[___], rest___] := parallelCombine[func, Unevaluated[r], rest]


(* normal case, eval arguments (HoldAllComplete preserves Unevaluated, by the way) *)

ParallelCombine[args__] := parallelCombine[args]


(* default combiner is h if h is Flat, Join otherwise *)
(* we preserve expr unevaluated in case it was called as ParallelCombine[f, Unevaluated[...]] *)

(* is abortable, because ParallelDispatch and ConcurrentEvaluate are *)

(* two flat functions, Union and Intersection, have options... *)

parallelCombine[func_, (h:Union|Intersection)[args___, oo:OptionsPattern[]], opts:OptionsPattern[]] :=
  With[{hh=h[##,oo]&},
	parallelCombine[func, Unevaluated[hh[args]], hh, opts]
  ]

parallelCombine[func_, expr:h_Symbol[___], opts:OptionsPattern[]]/; MemberQ[Attributes[h], Flat] :=
	parallelCombine[func, Unevaluated[expr], h, opts]

parallelCombine[func_, expr_, opts:OptionsPattern[]] := parallelCombine[func, Unevaluated[expr], Join, opts]

(* sequential case; leave out comb, it is assumed to be OneIdentity *)

parallelCombine[func_, expr_, comb_, OptionsPattern[]]/; $seQ :=
    (Message[ParallelCombine::nopar]; func[expr])

(* to parallelize, may want at least one element *)
(**
parallelCombine[func_, expr:h_[], comb_, OptionsPattern[]] :=
	    (Message[ParallelCombine::nopar1, HoldForm[func[expr]]]; func[expr])
**)


(* non-normal second arg; best guess is func[expr], but beware of Association, which is AtomQ  *)

parallelCombine[func_, Except[_Association ,expr_/;AtomQ[Unevaluated[expr]]], comb_, OptionsPattern[]] :=
    (Message[ParallelCombine::nopar1, HoldForm[func[expr]]]; func[expr])


(* handle (frozen) iterators: Range[iter__] -> Table[v, {v,iter}] *)

parallelCombine[func_, HoldPattern[Range[iter__]], comb_, o:OptionsPattern[]] :=
With[{meth=Parallel`Protected`MethOptCheck[ParallelCombine,OptionValue[ParallelCombine,{o},Method]],
	  dist=Parallel`Protected`DistOptCheck[ParallelCombine,OptionValue[ParallelCombine,{o},DistributedContexts]]},
	parallelIterate[Range, Table, comb, func, v, {v, iter}, {meth, dist} ]
]

(* Table[e, {...}, ...] _> Table[e, {...}, ...] *)

parallelCombine[func_, HoldPattern[Table[e_,iter___]], comb_, o:OptionsPattern[]] :=
With[{meth=Parallel`Protected`MethOptCheck[ParallelCombine,OptionValue[ParallelCombine,{o},Method]],
	  dist=Parallel`Protected`DistOptCheck[ParallelCombine,OptionValue[ParallelCombine,{o},DistributedContexts]]},
	parallelIterate[Table, Table, comb, func, e, iter, {meth, dist} ]
]

(* FileNames[args__] -> fileNames (for canonicalization); there is no "e" here
   FileNames[] can have zero arguments; fileNames[] will always have at least two *)

parallelCombine[func_, HoldPattern[FileNames[args___]], comb_, o:OptionsPattern[]] :=
With[{meth=Parallel`Protected`MethOptCheck[ParallelCombine,OptionValue[ParallelCombine,{o},Method]],
	  dist=Parallel`Protected`DistOptCheck[ParallelCombine,OptionValue[ParallelCombine,{o},DistributedContexts]]},
	parallelIterate[FileNames, FileNames, comb, func, Null, fileNames[args], {meth, dist} ]
]



(* normal case; handle different methods *)

(* handling of evaluated/unevaluated arguments:
	normal case: eval on master, do not eval on subkernel
	Unevaluated[args]: neither on master nor subkernel
*)

partitionList[expr_, {}] := {}

partitionList[expr_, sizes_List] := FoldPairList[TakeDrop,expr,sizes] (* partition according to list of sizes *)

SetAttributes[ha, HoldAllComplete]

(* this also matches for Association[exprs___], despite their atomicity; they need to be valid, though (evaluate!) *)

parallelCombine[func_, h_[exprs___], comb_, o:OptionsPattern[]] :=
With[{cmds = ha[exprs], nk=$KernelCount}, With[{items = Length[cmds]},
  Module[{batches, batchsize, sizes, res},
    (* handle Method option *)
    grokMethodOption[ParallelCombine, items, nk, batches, batchsize, Evaluate[OptionValue[ParallelCombine,{o},Method]]];
    sizes = makeSizes[items, nk, batches, batchsize];
    Parallel`Protected`AutoDistribute[{func,h,exprs}, ParallelCombine, OptionValue[ParallelCombine,{o},DistributedContexts]]; (* send definitions *)

    parStart;
	With[{rule=If[Head[func]===Symbol&&Length[Intersection[Attributes[func],{HoldFirst,HoldAll,HoldAllComplete}]]>0 ||
		          h===Association,
		  	ha[args__] :> func[h[args]],
		  	ha[args__] :> func[Unevaluated[h[args]]] ]
		 },
		(* list of frozen command sequences *)
		With[{cmdl = HoldComplete @@ partitionList[cmds,sizes] /. rule}, (* ragged Partition *)
		res = If[batches===1, ParallelDispatch[cmdl], ConcurrentEvaluate[cmdl]] /. ha :> Sequence;
		res = If[ MemberQ[res, $Aborted], $Aborted, comb @@ res ]; (* remote abort received *)
	]];
    parStop;
    res
]]]


(*** a collection of Parallel* convencience commands ***)

(* parallel Map; we can deal only with certain level specs, 0 is a nono *)

goodLevel[_] := False

goodLevel[n_Integer] := n>0 || n<0
goodLevel[Infinity] := True
goodLevel[{n_Integer}] := n>0 || n<0
goodLevel[{n1_Integer, n2_Integer}] := n1>0 && (n2>n1 || n2<0)

SetAttributes[ParallelMap,HoldRest] (* to freeze expr (only) *)

ParallelMap[f_, expr_, rest___] := parallelMap[f, Unevaluated[expr], rest]

parallelMap[f_, expr_, o:OptionsPattern[]] :=
With[{meth=Parallel`Protected`MethOptCheck[ParallelMap,OptionValue[ParallelMap,{o},Method]],
	  dist=Parallel`Protected`DistOptCheck[ParallelMap,OptionValue[ParallelMap,{o},DistributedContexts]]},
	ParallelCombine[Map[f], expr, Method->meth, DistributedContexts->dist ]
]
parallelMap[f_, expr_, level_?goodLevel, o:OptionsPattern[]] :=
With[{meth=Parallel`Protected`MethOptCheck[ParallelMap,OptionValue[ParallelMap,{o},Method]],
	  dist=Parallel`Protected`DistOptCheck[ParallelMap,OptionValue[ParallelMap,{o},DistributedContexts]]},
	ParallelCombine[Function[e, Map[f,Unevaluated[e],level]], expr, Method->meth, DistributedContexts->dist] ]
parallelMap[f_, expr_, level_, OptionsPattern[]] :=
	(Message[ParallelMap::nopar1, HoldForm[Map[f,expr,level]]]; Map[f, expr, level])

(* sniff..
ParallelMap[f_, h_[exprs___]] := h @@ WaitAll[ Composition[ParallelSubmit,f] /@ {exprs} ]
*)


(* iterators; arguments are
	the original symbol, such as ParallelTable
	the sequential iterator to use, such as Table
	the function used to combine the partial results, such as Join
	a function to wrap around the partial results on the subkernels, such as Identity
	the iterator body expr
	the sequence of iterator specs, such as {i, 1, 10}
	a single *list* of option values (not optional!), in the order {Method}
	iterator variable pattern: i:(_Symbol | Except[_List, _Symbol[___]])
 *)

SetAttributes[{parallelIterate, parallelIterateE}, HoldAll]

(* no iterators given; leave out comb *)

parallelIterate[orig_, iter_, comb_, f_, expr_, opts_List] := f[iter[expr]]

(* sequential, leave out comb *)

parallelIterate[orig_, iter_, comb_, f_, args__, opts_List]/; $seQ :=
    (Message[orig::nopar]; f[iter[args]])

(* evaluate iterator elements, except the variable, then call parallelIterateE[] *)

parallelIterate[orig_, iter_, comb_, f_, expr_, {i:(_Symbol | Except[_List, _Symbol[___]]), rest__}, others___, opts_List] :=
	Replace[ {rest}, {erest___} :> parallelIterateE[orig, iter, comb, f, expr, {i, erest}, others, opts] ]

parallelIterate[orig_, iter_, comb_, f_, expr_, it_, others___, opts_List] :=
	parallelIterateE[orig, iter, comb, f, expr, Evaluate[it], others, opts]

(* filenames, is never nested *)

parallelIterate[orig_, iter_, comb_, f_, expr_, fn_fileNames, opts_List] :=
	parallelIterateE[orig, iter, comb, f, expr, Evaluate[fn], opts]

(* parallelIterateE: do the actual work *)

(* with a variable *)

parallelIterateE[orig_, iter_, comb_, f_, expr_, it:{i:(_Symbol | Except[_List, _Symbol[___]]), w0_:1, w1_, dw_:1}, others___, {meth_,dist_,___}] :=
With[{nk=$KernelCount, items = Internal`GetIteratorLength[it,orig], floatq=Precision[{w0,dw}]<Infinity},
  Module[{batches, batchsize, sizes, res},
	If[ !IntegerQ[items] || items<0, (* cannot do it if symbolic *)
		Message[orig::nopar1, HoldForm[orig[expr, it, others]]];
		Return[iter[expr, it, others]]
	];
    (* handle Method option *)
    grokMethodOption[orig, Evaluate[items], nk, batches, batchsize, meth];
    sizes = makeSizes[items, nk, batches, batchsize];
    Parallel`Protected`AutoDistribute[{f,expr,others}, orig, dist, {HoldForm[i]}]; (* send definitions *)

    parStart;
	With[{i0 = w0+dw*Drop[FoldList[Plus, 0, sizes],-1], off=If[floatq, -1/2, -1]},
	  With[{bounds = Transpose[{i0, i0+dw*(sizes+off)}]},
		With[{chunks = HoldComplete @@ bounds /. {u0_, u1_} :> f[iter[expr, {i, u0, u1, dw}, others]]},
            res = If[batches===1, ParallelDispatch[chunks], ConcurrentEvaluate[chunks]];
        ];
        res = If[ MemberQ[res, $Aborted], $Aborted, comb @@ res ]; (* remote abort received *)
	]];
	parStop;
	res
]]

(* stripped down if no variable; can now also leave out the braces *)

parallelIterateE[orig_, iter_, comb_, f_, expr_, {w1_Integer}, others___] :=
	parallelIterateE[orig, iter, comb, f, expr, w1, others]

(* TODO: once bug 296907 is fixed, leave out the braces in GetIteratorLength *)

parallelIterateE[orig_, iter_, comb_, f_, expr_, w1_Integer, others___, {meth_,dist_,___}] :=
With[{nk=$KernelCount, items = Internal`GetIteratorLength[{w1},orig]},
  Module[{batches, batchsize, sizes, res},
	If[ !IntegerQ[items] || items<0, (* cannot do it if symbolic *)
		Message[orig::nopar1, HoldForm[orig[expr, w1, others]]];
		Return[iter[expr, w1, others]]
	];
    (* handle Method option *)
    grokMethodOption[orig, Evaluate[items], nk, batches, batchsize, meth];
    sizes = makeSizes[items, nk, batches, batchsize];
    Parallel`Protected`AutoDistribute[{f,expr,others}, orig, dist]; (* send definitions *)

    parStart;
	With[{chunks = HoldComplete @@ sizes /. u1_Integer :> f[iter[expr, {u1}, others]]},
            res = If[batches===1, ParallelDispatch[chunks], ConcurrentEvaluate[chunks]];
    ];
    res = If[ MemberQ[res, $Aborted], $Aborted, comb @@ res ]; (* remote abort received *)
	parStop;
	res
]]


(* new V6 list iterators, a case of ParallelCombine. note that vals is evaluated *)

parallelIterateE[orig_, iter_, comb_, f_, expr_, {i:(_Symbol | Except[_List, _Symbol[___]]), vals_List}, others___, {meth_,dist_,___}] :=
	parallelCombine[Function[vs, f[iter[expr, {i, vs}, others]]], vals, comb, Method->meth, DistributedContexts->dist]

(* I don't know if the variable can officially be missing, but this is how the sequential code works *)

parallelIterateE[orig_, iter_, comb_, f_, expr_, {vals_List}, others___, {meth_,dist_,___}] :=
	parallelCombine[Function[vs, f[iter[expr, {vs}, others]]], vals, comb, Method->meth, DistributedContexts->dist]

(* sum and product now also allow formal sums, with only a variable; not parallel *)

parallelIterateE[orig_, iter:(Sum|Product), comb_, f_, expr_, i:(_Symbol | Except[_List, _Symbol[___]]), others___, opts_List] := (
	Message[orig::nopar1, HoldForm[orig[expr, i, others]]];
    f[iter[expr, it, others]]
)

(* FileNames; we don't know (nor do we want to) how many there are, so always CoarsestGrained!
   expr is not used and always set to Null, others is not there, too *)
(* actually the correct thing to do is to instanciate the file list for any method other than Automatic
   or CoarsestGrain! *)

parallelIterateE[orig_, iter_, comb_, f_, expr_, fileNames[args__], {meth_,dist_,___}] :=
With[{nk=$KernelCount, items = Indeterminate},
  Module[{batches, batchsize, sizes, res},
    (* handle Method option *)
    grokMethodOption[orig, Evaluate[items], nk, batches, batchsize, meth];
    sizes = makeSizes[items, nk, batches, batchsize];
    Parallel`Protected`AutoDistribute[{f}, orig, dist]; (* send definitions *)

    parStart;
	With[{chunks = HoldComplete @@ Range[nk] /. i_Integer :> f[PartitionSelect[iter[args], {i,nk}]]},
            res = If[batches===1, ParallelDispatch[chunks], ConcurrentEvaluate[chunks]];
    ];
    res = If[ MemberQ[res, $Aborted], $Aborted, comb @@ res ]; (* remote abort received *)
	parStop;
	res
]]


(* fall through: no matching case found *)

parallelIterateE[orig_, iter_, comb_, f_, args__, opts_List] :=(
	Message[orig::nopar1, HoldForm[orig[args]]];
    f[iter[args]]
)

parallelIterate[orig_, iter_, comb_, f_, args__, opts_List] :=(
	Message[orig::nopar1, HoldForm[orig[args]]];
    f[iter[args]]
)

SetAttributes[{ParallelTable,ParallelSum,ParallelProduct,ParallelDo}, HoldAll]

(* isolate options once only, to make rules easier *)
(* option value order in list: {method} *)

ParallelTable[args__, Longest[OptionsPattern[]]]   :=
	parallelIterate[ParallelTable,Table,  Join, Identity,  args, Evaluate[{OptionValue[Method],OptionValue[DistributedContexts]}]]
ParallelSum[args__, Longest[OptionsPattern[]]]     :=
	parallelIterate[ParallelSum,  Sum,    Plus, Identity,  args, Evaluate[{OptionValue[Method],OptionValue[DistributedContexts]}]]
ParallelProduct[args__, Longest[OptionsPattern[]]] :=
	parallelIterate[ParallelProduct, Product, Times, Identity, args, Evaluate[{OptionValue[Method],OptionValue[DistributedContexts]}]]
ParallelDo[args__, Longest[OptionsPattern[]]]      :=
	parallelIterate[ParallelDo,   Do,     Null&, Identity, args, Evaluate[{OptionValue[Method],OptionValue[DistributedContexts]}]]


(* Array; this will fail if h[...] evaluates to something not listlike *)

ParallelArray[f_, args__, Longest[OptionsPattern[]]]/; $seQ :=
    (Message[ParallelArray::nopar]; Array[f, args])

(* careful with option argument, needs to be greedy *)

ParallelArray[f_, n_Integer?NonNegative, opts:OptionsPattern[]] := ParallelArray[f, n, 1, opts]
ParallelArray[f_, n_Integer?NonNegative, o_, opts:OptionsPattern[]] := ParallelArray[f, n, o, List, opts]
ParallelArray[f_, n_Integer?NonNegative, o_, h_, opts:OptionsPattern[]] := ParallelArray[f, {n}, {o}, h, opts]

(* we have to distinguish between List and any other head of the origin argument *)
ParallelArray[f_, nr_List, opts:OptionsPattern[]] := ParallelArray[f, nr, 1, opts]
ParallelArray[f_, nr_List, o_/;!ListQ[o], rest___] := ParallelArray[f, nr, Table[o, {Length[nr]}], rest]
ParallelArray[f_, nr_List, o_List, opts:OptionsPattern[]] := ParallelArray[f, nr, o, List, opts]

(* new syntax: origins may be lists themselves. do not handle for now *)

ParallelArray[f_, {items_Integer?NonNegative, nr___}, {o_, or___}, h_, OptionsPattern[]]/;(Length[{nr}]==Length[{or}]&&!ListQ[o]) :=
With[{nk=$KernelCount},
  Module[{batches, batchsize, sizes, origins, res},
    (* handle Method option *)
    grokMethodOption[ParallelArray, Evaluate[items], nk, batches, batchsize, Evaluate[OptionValue[Method]]];
    sizes = makeSizes[items, nk, batches, batchsize];
	origins = Drop[FoldList[Plus, o, sizes], -1];
    Parallel`Protected`AutoDistribute[f, ParallelArray, OptionValue[DistributedContexts]]; (* send definitions *)

	parStart;
	With[{chunks = HoldComplete @@ Transpose[{sizes, origins}] /. {ni_,oi_} :> Array[f, {ni, nr}, {oi, or}, h]},
		res = If[batches===1, ParallelDispatch[chunks], ConcurrentEvaluate[chunks]];
    ];
    (* our best guess for Flat head is h, to avoid collapsing, bug 181133 *)
    With[{comb = If[Head[h]===Symbol && MemberQ[Attributes[h], Flat], h, Join]},
    	res = If[ MemberQ[res, $Aborted], $Aborted, comb @@ res ]; (* remote abort received *)
    ];
    parStop;
    res
]]

ParallelArray[f_, args__, Longest[opts:OptionsPattern[]]] :=
	(Message[ParallelArray::nopar1, HoldForm[Array[f, args]]]; Array[f, args])


End[]

Protect[ ParallelCombine ]
Protect[ ParallelMap,ParallelTable,ParallelSum,ParallelProduct,ParallelDo,ParallelArray ]

EndPackage[]
