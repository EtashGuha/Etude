BeginPackage["GraphStore`SPARQL`Algebra`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`RDF`"];

(* 18.2 Translation to the SPARQL Algebra *)

(* Graph Pattern *)
BGP;
join;
leftJoin;
filter;
union;
graph;
extend;
minus;
group;
aggregation;
aggregateJoin;

(* Solution Modifiers *)
(* ToList *)
orderBy;
project;
distinct;
(* Reduced *)
slice;
(* ToMultiSet *)

(* Property Path *)
link;
inv;
seq;
alt;
zeroOrMorePath;
oneOrMorePath;
zeroOrOnePath;
NPS;

(* other *)
(* SPARQLValues *)
algValues;
service;

solutionListIdentity;

exists;
path;

Begin["`Private`"];

OptimizeAlgebraExpression[args___] := With[{res = Catch[iOptimizeAlgebraExpression[args], $failTag]}, res /; res =!= $failTag];
ToAlgebraExpression[args___] := With[{res = Catch[iToAlgebraExpression[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* optimize algebra expression *)

clear[iOptimizeAlgebraExpression];
iOptimizeAlgebraExpression[expr_] := expr //. {
	filter[True, x_] :> x,
	join[x_] :> x,
	join[x___, BGP[{}], y___] :> join[x, y],
	orderBy[x_, None] :> x,
	project[x_, All] :> x,
	slice[x_, 0, Infinity] :> x,


	(* Foundations of SPARQL Query Optimization *)
	(* https://arxiv.org/abs/0812.3788 *)

	(* I. Idempotence and Inverse *)

	(* II. Associativity, Commutativity, Distributivity *)
	(* JUDistR *)
	join[union[a1_, a2_], a3_] :> union[join[a1, a3], join[a2, a3]],
	(* JUDistL *)
	join[a1_, union[a2_, a3_]] :> union[join[a1, a2], join[a1, a3]],

	(* III. Projection Pushing *)
	(* PBaseI *)
	project[a_, s_List] /; With[{pv = possibleVars[a]}, ListQ[pv] && ContainsAll[s, pv]] :> a,
	(* PBaseII *)
	project[a_, s_List] :> With[{pv = possibleVars[a]}, With[{i = Intersection[s, pv]}, project[a, i] /; Length[i] < Length[s]] /; ListQ[pv]],
	(* PMerge *)
	project[project[a_, s2_List], s1_List] :> project[a, Intersection[s1, s2]],
	(* PUPush *)
	project[union[a1_, a2_], s_] :> union[project[a1, s], project[a2, s]],
	(* PJPush *)
	project[join[a1_, a2_], s_List] :> With[
		{pv1 = possibleVars[a1], pv2 = possibleVars[a2]},
		With[
			{s1 = Union[s, Intersection[pv1, pv2]]},
			project[join[project[a1, s1], project[a2, s1]], s] /; AnyTrue[{pv1, pv2}, Function[pv, Complement[pv, s1] =!= {}]]
		] /; AllTrue[{pv1, pv2}, ListQ]
	],

	(* IV. Filter Decomposition and Elimination *)

	(* V. Filter Pushing *)

	(* VI. Minus and Left Outer Join Rewriting *)

	(* LEMMA 4 *)
	(* FElimI, FElimII *)
	project[filter[x_SPARQLVariable == c_ | c_ == x_SPARQLVariable, a_] /; MemberQ[certainVars[a], x], s_List] /; ! MemberQ[s, x] :> project[a /. x -> c, s]
};


clear[certainVars];
certainVars[x_] := Catch[iCertainVars[x], $tag];

iCertainVars[BGP[l_List]] := DeleteDuplicates[Cases[l, SPARQLVariable[var_String] :> var, {2}]];
iCertainVars[join[a1_, a2_]] := Union[iCertainVars[a1], iCertainVars[a2]];
iCertainVars[union[a1_, a2_]] := Intersection[iCertainVars[a1], iCertainVars[a2]];
iCertainVars[minus[a1_, _]] := iCertainVars[a1];
iCertainVars[project[a_, s_List]] := Intersection[iCertainVars[a], s];
iCertainVars[filter[_, a_]] := iCertainVars[a];
iCertainVars[_] := Throw[$Failed, $tag];

clear[possibleVars];
possibleVars[x_] := Catch[iPossibleVars[x], $tag];

iPossibleVars[BGP[l_List]] := DeleteDuplicates[Cases[l, SPARQLVariable[var_String] :> var, {2}]];
iPossibleVars[join[a1_, a2_]] := Union[iPossibleVars[a1], iPossibleVars[a2]];
iPossibleVars[union[a1_, a2_]] := Union[iPossibleVars[a1], iPossibleVars[a2]];
iPossibleVars[minus[a1_, _]] := iPossibleVars[a1];
iPossibleVars[project[a_, s_List]] := Intersection[iPossibleVars[a], s];
iPossibleVars[filter[_, a_]] := iPossibleVars[a];
iPossibleVars[_] := Throw[$Failed, $tag];

(* end optimize algebra expression *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* to algebra expression *)

clear[iToAlgebraExpression];
iToAlgebraExpression[x_] := translate[SPARQLFromLegacySyntax[x]];


clear[translate];

(* filter *)
translate[Verbatim[Condition][gp_, cond_]] := filter[
	ReplaceRepeated[cond, {
		Not[SPARQLEvaluation[_String?(StringMatchQ["EXISTS", IgnoreCase -> True])][p_]] :> Not[exists[translate[p]]],
		SPARQLEvaluation[_String?(StringMatchQ["EXISTS", IgnoreCase -> True])][p_] :> exists[translate[p]]
	}],
	translate[gp]
];

(* property paths *)
translate[{gp1___, SPARQLPropertyPath[s_, pathexpr_, o_], gp2___}] := translate[{
	gp1,
	Sequence @@ translateSPARQLPropertyPath[SPARQLPropertyPath[s, translatePropertyPathExpression[pathexpr], o]],
	gp2
}];
translate[ppp_SPARQLPropertyPath] := translate[{ppp}];
translate[p_path] := p;

(* basic graph pattern *)
translate[{x___, t : RDFTriple[_RDFCollection, _, _] | RDFTriple[_, _, _RDFCollection], y___}] := translate[
	{
		x,
		Sequence @@ Replace[ExpandRDFCollection[t], _ExpandRDFCollection :> fail[]],
		y
	}
];
translate[triples : {___RDFTriple}] := BGP[triples];
translate[t_RDFTriple] := translate[{t}];
translate[gp_List?(MemberQ[_RDFTriple])] := translate[Replace[gp, {x___, Longest[t__RDFTriple], z___} :> {x, {t}, z}]];

(* union *)
translate[un_Alternatives] := union @@ translate /@ un;

(* graph graph pattern *)
translate[SPARQLGraph[iriorvar_, gp_]] := graph[iriorvar, translate[gp]];

(* service *)
translate[SPARQLService[varoriri_, gp_, opts : OptionsPattern[]]] := service[varoriri, gp (*translate[gp]*), OptionValue[SPARQLService, {opts}, "Silent"]];

(* group graph pattern *)
translate[gp_List] := Fold[
	Function[{g, e},
		e // Replace[{
			(* optional *)
			SPARQLOptional[p_] :> Replace[translate[p], {
				filter[f_, a2_] :> leftJoin[g, a2, f],
				a_ :> leftJoin[g, a, True]
			}],
			(* minus *)
			Verbatim[Except][p_] :> minus[g, translate[p]],
			(* bind *)
			(Rule | RuleDelayed)[var_String, expr_] :> extend[g, var, expr],
			(* other *)
			_ :> join[g, translate[e]]
		}]
	],
	BGP[{}],
	gp
];

(* minus *)
translate[Verbatim[Except][c_, gp_]] := translate[Append[If[ListQ[gp], gp, {gp}], Except[c]]];

(* values *)
translate[SPARQLValues[vars_, values_]] := algValues[solutionListIdentity[], vars, values];

(* sub-select *)
translate[SPARQLSelect[where_]] := translate[where];

translate[RightComposition[s__, outer_]] := translate[outer] /. solutionListIdentity[] -> translate[RightComposition[s]];
translate[c_Composition] := translate[RightComposition @@ Reverse[c]];

translate[SPARQLOrderBy[f_]] := orderBy[solutionListIdentity[], f];

translate[SPARQLProject[All]] := solutionListIdentity[];
translate[SPARQLProject[proj_]] := Module[
	{vars = {}, algebraExpr = solutionListIdentity[]},
	Replace[
		Flatten[{proj}],
		{
			var_String :> AppendTo[vars, var],
			(Rule | RuleDelayed)[var_String, expr_] :> (
				AppendTo[vars, var];
				algebraExpr = extend[algebraExpr, var, expr];
			)
		},
		{1}
	];
	algebraExpr = project[algebraExpr, vars];
	algebraExpr
];

translate[SPARQLDistinct[OptionsPattern[]]] := distinct[solutionListIdentity[]];

translate[SPARQLLimit[limit_, offset_ : 0]] := slice[solutionListIdentity[], offset, limit];

translate[SPARQLAggregate[agg_]] := translate[SPARQLAggregate[agg, None]];
translate[SPARQLAggregate[agg_, groupby_]] := translate[SPARQLAggregate[agg, groupby, True]];
translate[SPARQLAggregate[agg_, groupby_, having_]] := translate[SPARQLAggregate[agg, groupby, having, None]];
translate[SPARQLAggregate[agg_, groupby_, having_, orderby_]] := Module[
	{algebraExpr, agg1, groupby1, having1, orderby1},
	algebraExpr = solutionListIdentity[];
	{agg1, groupby1, having1, orderby1} = {agg, groupby, having, orderby};
	groupby1 = Replace[
		Replace[groupby1, {None :> {1}, x_ :> Flatten[{x}]}],
		(Rule | RuleDelayed)[var_String, expr_] :> (
			algebraExpr = extend[algebraExpr, var, expr];
			SPARQLVariable[var]
		),
		{1}
	];
	algebraExpr = group[groupby1, algebraExpr];
	agg1 = Replace[
		Flatten[{agg1}],
		{
			var_String :> var -> SPARQLEvaluation["SAMPLE"][SPARQLVariable[var]],
			r_RuleDelayed :> Rule @@ r
		},
		{1}
	];
	having1 = Flatten[{having1}];
	Module[
		{tempvars = {}, aggs = {}},
		{agg1, having1, orderby1} = {agg1, having1, orderby1} /. f_SPARQLEvaluation[args___, o : OptionsPattern[]] :> (
			AppendTo[aggs, aggregation[{args}, f, Flatten[{o}], algebraExpr]];
			Last[AppendTo[tempvars, SPARQLVariable[Unique["agg"]]]]
		);
		algebraExpr = aggregateJoin[tempvars[[All, 1]], aggs];
	];
	Function[h,
		algebraExpr = filter[h, algebraExpr]
	] /@ having1;
	Function[{var, expr},
		algebraExpr = extend[algebraExpr, var, expr];
	] @@@ agg1;
	algebraExpr = orderBy[algebraExpr, orderby1];
	algebraExpr = project[algebraExpr, Keys[agg1]];
	algebraExpr
];

translate[query_?PossibleQueryQ] := query;


(* 9.1 Property Path Syntax *)
(* ^elt	InversePath *)
PropertyPathExpressionQ[_SPARQLInverseProperty] := True;
(* elt1 / elt2	SequencePath *)
PropertyPathExpressionQ[_PatternSequence] := True;
(* elt1 | elt2	AlternativePath *)
PropertyPathExpressionQ[_Alternatives] := True;
(* elt*	ZeroOrMorePath *)
PropertyPathExpressionQ[Verbatim[RepeatedNull][_]] := True;
(* elt+	OneOrMorePath *)
PropertyPathExpressionQ[Verbatim[Repeated][_]] := True;
(* elt?	ZeroOrOnePath *)
PropertyPathExpressionQ[Verbatim[Repeated][_, {0, 1}] | Verbatim[RepeatedNull][_, {0, 1} | 1]] := True;
(* !(iri1| ...|irij|^irij+1| ...|^irin)	NegatedPropertySet *)
PropertyPathExpressionQ[_Except] := True;
(* not a path expression *)
PropertyPathExpressionQ[_] := False;

clear[translatePropertyPathExpression];
translatePropertyPathExpression[{x_}] := translatePropertyPathExpression[x];
translatePropertyPathExpression[{x__}] := Fold[seq, translatePropertyPathExpression /@ {x}];
translatePropertyPathExpression[i_SPARQLInverseProperty] := inv @@ translatePropertyPathExpression /@ i;
translatePropertyPathExpression[Verbatim[Except][Verbatim[Alternatives][i__SPARQLInverseProperty]]] := inv[NPS[{i}[[All, 1]]]];
translatePropertyPathExpression[Verbatim[Except][Verbatim[Alternatives][i : Except[_SPARQLInverseProperty] ...]]] := NPS[{i}];
translatePropertyPathExpression[Verbatim[Except][Verbatim[Alternatives][i__]]] := alt @@ Function[x,
	translatePropertyPathExpression[Except[Alternatives @@ x]]
] /@ SplitBy[{i}, MatchQ[_SPARQLInverseProperty]];
translatePropertyPathExpression[Verbatim[Except][Except[_Alternatives, i_]]] := translatePropertyPathExpression[Except[Alternatives[i]]];
translatePropertyPathExpression[s_PatternSequence] := Fold[seq, translatePropertyPathExpression /@ s];
translatePropertyPathExpression[a_Alternatives] := alt @@ translatePropertyPathExpression /@ a;
translatePropertyPathExpression[Verbatim[RepeatedNull][p_]] := zeroOrMorePath[translatePropertyPathExpression[p]];
translatePropertyPathExpression[Verbatim[Repeated][p_]] := oneOrMorePath[translatePropertyPathExpression[p]];
translatePropertyPathExpression[Verbatim[Repeated][z_, {0, 1}] | Verbatim[RepeatedNull][z_, {0, 1} | 1]] := zeroOrOnePath[translatePropertyPathExpression[z]];
translatePropertyPathExpression[i_] := link[i];

(* 18.2.2.4 Translate Property Path Patterns *)
clear[translateSPARQLPropertyPath];
translateSPARQLPropertyPath[SPARQLPropertyPath[x_, link[iri_], y_]] := translateSPARQLPropertyPath[RDFTriple[x, iri, y]];
translateSPARQLPropertyPath[SPARQLPropertyPath[x_, inv[iri_], y_]] := translateSPARQLPropertyPath[SPARQLPropertyPath[y, iri, x]];
translateSPARQLPropertyPath[SPARQLPropertyPath[x_, seq[p_, q_], y_]] := With[
	{var = SPARQLVariable[Unique[]]},
	Join @@ translateSPARQLPropertyPath /@ {SPARQLPropertyPath[x, p, var], SPARQLPropertyPath[var, q, y]}
];
translateSPARQLPropertyPath[SPARQLPropertyPath[x_, p : _alt | _NPS | _oneOrMorePath | _seq | _zeroOrMorePath | _zeroOrOnePath, y_]] := {path[x, p, y]};
translateSPARQLPropertyPath[t_RDFTriple] := {t};

(* end to algebra expression *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
