BeginPackage["GraphStore`SPARQL`Evaluation`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`ArrayAssociation`"];
Needs["GraphStore`SPARQL`Algebra`"];

Begin["`Private`"];

EvaluateAlgebraExpression[args___] := With[{res = Catch[iEvaluateAlgebraExpression[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iEvaluateAlgebraExpression];

(* file containing a graph or dataset *)
iEvaluateAlgebraExpression[file : _File | _IRI | _URL, gp_] := iEvaluateAlgebraExpression[Import[file], gp];

(* choose the active graph *)
iEvaluateAlgebraExpression[store_RDFStore, gp_] := iEvaluateAlgebraExpression[withActiveGraph[store["DefaultGraph"], store], gp];
iEvaluateAlgebraExpression[Except[_withActiveGraph, g_], gp_] := iEvaluateAlgebraExpression[withActiveGraph[g, g], gp];

(* alternative graph forms *)
iEvaluateAlgebraExpression[withActiveGraph[file : _File | _IRI | _URL, ds_], gp_] := iEvaluateAlgebraExpression[withActiveGraph[First[Import[file]], ds], gp];

(* optimization *)
iEvaluateAlgebraExpression[g_, join[OrderlessPatternSequence[bgp : BGP[{___, RDFTriple[___, SPARQLVariable[x_], ___], ___}], path[SPARQLVariable[x_], p_, y_]]]] := With[
	{s = Normal[iEvaluateAlgebraExpression[g, bgp], ArrayAssociation]},
	SPARQLJoinAcross[
		s,
		Join @@ Function[
			With[
				{val = Lookup[#, x]},
				iEvaluateAlgebraExpression[g, path[val, p, y]] // Map[Prepend[x -> val]]
			]
		] /@ s
	]
];

(* property path *)
iEvaluateAlgebraExpression[g_, p_path] := EvaluatePropertyPath[activeGraph[g], p] // Replace[_EvaluatePropertyPath :> fail[]];

(* basic graph pattern *)
iEvaluateAlgebraExpression[g_, BGP[bgp_List]] := EvaluateBasicGraphPattern[activeGraph[g], bgp, "LeftJoinRestriction" -> $leftJoinRestriction] // Replace[_EvaluateBasicGraphPattern :> fail[]];

(* filter *)
iEvaluateAlgebraExpression[g_, filter[expr_, gp_]] := evalFilter[expr, iEvaluateAlgebraExpression[g, gp], g];

(* join *)
iEvaluateAlgebraExpression[g_, j_join] := SPARQLJoinAcross @@ Function[iEvaluateAlgebraExpression[g, #]] /@ j;

(* left join *)
iEvaluateAlgebraExpression[g_, leftJoin[p1_, p2_BGP, expr_]] := Block[
	{$leftJoinRestriction = iEvaluateAlgebraExpression[g, p1]},
	evalLeftJoin[$leftJoinRestriction, iEvaluateAlgebraExpression[g, p2], expr]
];
iEvaluateAlgebraExpression[g_, leftJoin[p1_, p2_, expr_]] := evalLeftJoin[iEvaluateAlgebraExpression[g, p1], iEvaluateAlgebraExpression[g, p2], expr];

(* union *)
iEvaluateAlgebraExpression[g_, un_union] := evalUnion @@ Function[iEvaluateAlgebraExpression[g, #]] /@ un;

(* minus *)
iEvaluateAlgebraExpression[g_, minus[p1_, p2_]] := evalMinus[
	iEvaluateAlgebraExpression[g, p1],
	iEvaluateAlgebraExpression[g, p2],
	p1,
	p2
];

(* graph *)
iEvaluateAlgebraExpression[withActiveGraph[_, store_RDFStore], graph[iri : _File | _IRI | _URL, gp_]] := Module[{tag}, Catch[
	iEvaluateAlgebraExpression[
		withActiveGraph[
			Lookup[store["NamedGraphs"], iri, Throw[{}, tag]],
			store
		],
		gp
	],
	tag
]];
iEvaluateAlgebraExpression[withActiveGraph[_, store_RDFStore], graph[SPARQLVariable[var_], gp_]] := evalUnion @@ KeyValueMap[
	Function[{n, g},
		SPARQLJoinAcross[
			iEvaluateAlgebraExpression[withActiveGraph[g, store], gp],
			{<|var -> n|>}
		]
	],
	store["NamedGraphs"]
];

(* group *)
iEvaluateAlgebraExpression[g_, group[exprlist_, p_]] := evalGroup[exprlist, iEvaluateAlgebraExpression[g, p]];

(* aggregation *)
iEvaluateAlgebraExpression[g_, aggregation[args_List, f_, opts_List, gp_]] := evalAggregation[args, f, opts, iEvaluateAlgebraExpression[g, gp]];

(* aggregate join *)
iEvaluateAlgebraExpression[g_, aggregateJoin[agg_, s_]] := evalAggregateJoin[agg, Function[iEvaluateAlgebraExpression[g, #]] /@ s];

(* extend *)
iEvaluateAlgebraExpression[g_, extend[gp_, var_, expr_]] := evalExtend[iEvaluateAlgebraExpression[g, gp], var, expr];

(* distinct *)
iEvaluateAlgebraExpression[g_, distinct[gp_]] := evalDistinct[iEvaluateAlgebraExpression[g, gp]];

(* project *)
iEvaluateAlgebraExpression[g_, project[gp_, vars_]] := evalProject[iEvaluateAlgebraExpression[g, gp], vars];

(* order by *)
iEvaluateAlgebraExpression[g_, orderBy[gp_, cond_]] := evalOrderBy[iEvaluateAlgebraExpression[g, gp], cond];

(* slice *)
iEvaluateAlgebraExpression[_, slice[_, _, 0]] := {};
iEvaluateAlgebraExpression[g_, slice[union[a1_, a2_], 0, length_Integer?Positive]] := iEvaluateAlgebraExpression[g, a1] // Replace[
	sols1_ /; Length[sols1] < length :> evalUnion[sols1, iEvaluateAlgebraExpression[g, slice[a2, 0, length - Length[sols1]]]]
];
iEvaluateAlgebraExpression[g_, slice[gp_, start_, length_]] := evalSlice[iEvaluateAlgebraExpression[g, gp], start, length];

(* service *)
iEvaluateAlgebraExpression[_, service[iri_, gp_, silentop_?BooleanQ]] := EvaluateFederatedQuery[iri, gp, "Silent" -> silentop] // Replace[_EvaluateFederatedQuery :> fail[]];

iEvaluateAlgebraExpression[g_, algValues[gp_, vars_List, values_List] /; MatchQ[Dimensions[values, 2], {_, Length[vars]}]] := SPARQLJoinAcross[
	iEvaluateAlgebraExpression[g, gp],
	DeleteCases[
		AssociationThread[vars, #] & /@ values,
		Undefined,
		{2}
	]
] // Replace[_SPARQLJoinAcross :> fail[]];
iEvaluateAlgebraExpression[g_, algValues[gp_, Except[_List, var_], values_List]] := iEvaluateAlgebraExpression[g, algValues[gp, {var}, List /@ values]];

iEvaluateAlgebraExpression[g_, query_?PossibleQueryQ] := g // query;

iEvaluateAlgebraExpression[g : withActiveGraph[_, Except[_List]], solutionListIdentity[]] := iEvaluateAlgebraExpression[g, BGP[{}]];
iEvaluateAlgebraExpression[g_, solutionListIdentity[]] := activeGraph[g] // Replace[Except[_List] :> fail[]];


clear[activeGraph];
activeGraph[withActiveGraph[active_, _]] := active;

clear[applySolution];
applySolution[{} | ArrayAssociation[{}, ___], _] := {};
applySolution[sol_, SPARQLVariable[var_]] := Lookup[sol, var, $Failed];
applySolution[sol_, l : {SPARQLVariable[_] ...}] := With[{vars = l[[All, 1]]}, Lookup[sol, vars, $Failed]];
applySolution[sol_, expr_] := ReleaseHold[
	Hold[expr] //. {
		q_Quantity :> q,
		(* 17.3 Operator Mapping *)
		(op : Alternatives[
			Not,
			Equal, Unequal, Less, Greater, LessEqual, GreaterEqual,
			Times, Plus
		])[args___] :> EvaluateSPARQLOperator[op, args],
		(* 17.4 Function Definitions *)
		f_SPARQLEvaluation[args___] :> EvaluateSPARQLFunction[f, args]
	} /. With[
		{n = Normal[sol, ArrayAssociation]},
		Map[KeyMap[SPARQLVariable], n, {Boole[ListQ[n]]}]
	]
];

clear[evalFilter];
evalFilter[True, sols_, _] := sols;
evalFilter[False, _, _] := {};
evalFilter[expr_?(FreeQ[exists]), sols_, _] := Pick[sols, TrueQ /@ applySolution[sols, expr]];
evalFilter[expr_, sols_, g_] := Select[Normal[sols, ArrayAssociation], applySolution[#, expr /. exists[gp_] :> evalExists[gp, #, g]] &];

clear[evalSubstitute];
evalSubstitute[p_, sol_] := p /. (sol // KeyMap[Replace[var_String :> SPARQLVariable[var]]]);

clear[evalExists];
evalExists[p_, sol_, g_] := Length[iEvaluateAlgebraExpression[g, evalSubstitute[p, sol]]] > 0;

clear[compatibleQ];
compatibleQ[sols__] := SameQ @@ KeyIntersection[{sols}];

clear[evalLeftJoin];
evalLeftJoin[{}, _, _] := {};
evalLeftJoin[sols1_List, {}, True] := sols1;
evalLeftJoin[sols1_List, sols2_List, expr_] := If[SameQ @@ Keys[sols1] && SameQ @@ Keys[sols2],
	With[
		{keys = Intersection[First[Keys[sols1]], First[Keys[sols2]]]},
		If[TrueQ[expr],
			leftJoinAcross[sols1, sols2, Key /@ keys],
			Flatten[
				If[keys === {},
					Function[s1,
						evalFilter[expr, Join[s1, #] & /@ sols2, Null] // Replace[{} :> {s1}]
					] /@ sols1,
					MapIndexed[
						Function[{pos2, pos1},
							With[
								{s1 = sols1[[First[pos1]]]},
								If[MissingQ[pos2],
									{s1},
									evalFilter[expr, Join[s1, #] & /@ sols2[[pos2]], Null] // Replace[{} :> {s1}]
								]
							]
						],
						Lookup[PositionIndex[sols2[[All, Key /@ keys]]], sols1[[All, Key /@ keys]]]
					]
				],
				1
			]
		]
	],
	leftJoinAcross[sols1, sols2, If[TrueQ[expr],
		compatibleQ,
		compatibleQ[##] && TrueQ[applySolution[Join[##], expr]] &
	]]
];
evalLeftJoin[x___, a_ArrayAssociation, y___] := evalLeftJoin[x, Normal[a], y];

clear[leftJoinAcross];
leftJoinAcross[l1_List, l2_List, {}] := Flatten[Outer[Join, l1, l2], 1];
leftJoinAcross[l1_List, l2_List, keys : _List | _String | _Key] := DeleteCases[
	JoinAcross[l1, l2, keys, "Left"],
	Missing["Unmatched"],
	{2}
];
leftJoinAcross[l1_List, l2_List, pred_] := Module[
	{tag, included},
	First[Last[Reap[
		Scan[
			Function[e1,
				included = False;
				Scan[
					Function[e2,
						If[pred[e1, e2],
							included = True;
							Sow[Join[e1, e2], tag]
						];
					],
					l2
				];
				If[! included,
					Sow[e1, tag]
				]
			],
			l1
		];,
		tag
	]], {}]
];

clear[evalUnion];
evalUnion[l___] := Join[l];

clear[evalMinus];
evalMinus[sols1_List, sols2_List, patt1_, patt2_] := With[{
	vars1 = varsFromCertainPatts[patt1],
	vars2 = varsFromCertainPatts[patt2]
},
	With[
		{i = Intersection[vars1, vars2]},
		If[i === {},
			sols1,
			sols1[[
				Join @@ Values[
					KeyDrop[
						PositionIndex[Lookup[sols1, i]],
						DeleteDuplicates[Lookup[sols2, i]]
					]
				]
			]]
		]
	] /; ListQ[vars1] && ListQ[vars2]
];
evalMinus[sols1_List, sols2_List, ___] := Select[
	sols1,
	Function[s1,
		AllTrue[
			sols2,
			Function[s2,
				Or[
					! compatibleQ[s1, s2],
					DisjointQ[Keys[s1], Keys[s2]]
				]
			]
		]
	]
];
evalMinus[x___, sols_ArrayAssociation, y___] := evalMinus[x, Normal[sols, ArrayAssociation], y];

clear[varsFromCertainPatts];
varsFromCertainPatts[patt_] := Catch[
	DeleteDuplicates[
		First[Last[Reap[
			iVarsFromCertainPatts[patt],
			$tag
		]], {}]
	],
	$tag
];

clear[iVarsFromCertainPatts];
iVarsFromCertainPatts[bgp : BGP[l_List]] := Scan[iVarsFromCertainPatts, l];
iVarsFromCertainPatts[t_RDFTriple] := (
	Cases[t, SPARQLVariable[var_] :> Sow[var, $tag]];
);
iVarsFromCertainPatts[_] := Throw[$Failed, $tag];

clear[evalGroup];
evalGroup[exprlist_, sols_ /; Length[sols] === 0] := <|exprlist -> sols|>;
evalGroup[exprlist : {1}, sols_] := <|exprlist -> sols|>;
evalGroup[exprlist_, sols_] := GroupBy[
	Normal[sols, ArrayAssociation],
	listEval[exprlist, #] &
];

clear[listEval];
listEval[l_, sol_?AssociationQ] := First[listEval[l, {sol}]];
listEval[{}, sols_] := ConstantArray[{}, Length[sols]];
listEval[_, {}] := {};
listEval[l : {SPARQLVariable[_] ..}, sols_] := With[{vars = l[[All, 1]]}, Lookup[sols, vars, $Failed]];
listEval[l_, sols_] := applySolution[sols, l];

clear[evalAggregation];
evalAggregation[exprlist_, SPARQLEvaluation[_SPARQLDistinct /* f_], opts_List, g_] := evalAggregation[exprlist, SPARQLEvaluation[f], Append[opts, "Distinct" -> True], g];
evalAggregation[{}, f : SPARQLEvaluation[_String?(StringMatchQ["COUNT", IgnoreCase -> True])], opts_List, g_?AssociationQ] := AssociationMap[
	Apply[Function[{key, sols},
		key -> Replace[EvaluateSPARQLAggregateFunction[f, sols, All, Sequence @@ opts], _EvaluateSPARQLAggregateFunction :> fail[]]
	]],
	g
];
evalAggregation[exprlist_List, f_, opts_List, g_?AssociationQ] := AssociationMap[
	Apply[Function[{key, sols},
		key -> Replace[EvaluateSPARQLAggregateFunction[f, sols, listEval[exprlist, sols], Sequence @@ opts], _EvaluateSPARQLAggregateFunction :> fail[]]
	]],
	g
];

clear[evalAggregateJoin];
evalAggregateJoin[agg_List, s_] := AssociationThread[agg, #] & /@ Transpose[Values[KeyIntersection[s]]];

clear[evalExtend];
evalExtend[{}, __] := {};
evalExtend[sols_List, var_, expr_] := MapThread[
	Append[#, var -> #2] &,
	{sols, applySolution[sols, expr]}
];
evalExtend[a_ArrayAssociation, args__] := evalExtend[Normal[a], args];

clear[evalDistinct];
evalDistinct[l_] := DeleteDuplicatesBy[l, KeySort];

clear[evalProject];
evalProject[l_, All] := l;
(* http://bugs.wolfram.com/show?number=343014 *)
evalProject[{}, _] := {};
evalProject[l_, vars_List] := KeyTake[l, vars];

clear[evalOrderBy];
evalOrderBy[l_, {} | _?(FreeQ[SPARQLVariable])] := l;
evalOrderBy[l_, {SPARQLVariable[var_] | (Rule | RuleDelayed)[SPARQLVariable[var_], order_String]}] := If[
	{order} === {"Descending"},
	ReverseSortBy,
	SortBy
][Normal[l, ArrayAssociation], Key[var]];
evalOrderBy[l_, condList_List] := Sort[
	Normal[l, ArrayAssociation] // Map[KeyMap[SPARQLVariable]],
	Function[Scan[
		Function[cond,
			Replace[
				cond // Replace[{
					(Rule | RuleDelayed)[expr_, "Ascending"] :> Order[expr /. #1, expr /. #2],
					(Rule | RuleDelayed)[expr_, "Descending"] :> -Order[expr /. #1, expr /. #2],
					expr_ :> Order[expr /. #1, expr /. #2]
				}],
				Except[0, order_] :> Return[order]
			]
		],
		condList
	]]
] // Map[KeyMap[First]];
evalOrderBy[l_, Except[_List, x_]] := evalOrderBy[l, {x}];

clear[evalSlice];
evalSlice[l_, start_Integer?NonNegative, length : Infinity | _Integer?NonNegative] := l // Drop[#, UpTo[start]] & // Take[#, UpTo[length]] &;

End[];
EndPackage[];
