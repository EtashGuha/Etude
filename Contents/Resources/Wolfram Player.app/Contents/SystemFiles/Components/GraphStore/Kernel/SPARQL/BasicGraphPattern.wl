(* optimization based on: *)
(* SPARQL Basic Graph Pattern Optimization Using Selectivity Estimation *)
(* wwwconference.org/www2008/papers/pdf/p595-stocker1.pdf *)

BeginPackage["GraphStore`SPARQL`BasicGraphPattern`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`ArrayAssociation`"];
Needs["GraphStore`RDF`"];

Options[EvaluateBasicGraphPattern] = {
	SPARQLEntailmentRegime :> $EntailmentRegime,
	"LeftJoinRestriction" -> None
};

Begin["`Private`"];

EvaluateBasicGraphPattern[args___] := With[{res = Catch[iEvaluateBasicGraphPattern[args], $failTag]}, res /; res =!= $failTag];
ListConnectedComponents[args___] := With[{res = Catch[iListConnectedComponents[args], $failTag]}, res /; res =!= $failTag];
SortListConnectedComponentBy[args___] := With[{res = Catch[iSortListConnectedComponentBy[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iEvaluateBasicGraphPattern];
Options[iEvaluateBasicGraphPattern] = Options[EvaluateBasicGraphPattern];

(* special case *)
iEvaluateBasicGraphPattern[_, {}, OptionsPattern[]] := {<||>};

(* entailment *)
iEvaluateBasicGraphPattern[el_, epl_, OptionsPattern[]] := With[
	{er = OptionValue[SPARQLEntailmentRegime]},
	Replace[EvaluateBasicGraphPatternWithEntailment[el, epl, er], _EvaluateBasicGraphPatternWithEntailment :> fail[]] /; er =!= None
];

(* general case *)
iEvaluateBasicGraphPattern[el_, epl : {__RDFTriple}, OptionsPattern[]] := Module[
	{res2, groups},
	groups = iSortListConnectedComponentBy[#, $namedVarPattern, triplePatternSelectivity] & /@ iListConnectedComponents[epl, $namedVarPattern];
	res2 = evaluateConnectedBasicGraphPattern[el, First[groups]];
	Function[nextGroup,
		res2 = outerJoinAcross[res2, evaluateConnectedBasicGraphPattern[el, nextGroup]];
	] /@ Rest[groups];
	res2
];

clear[outerJoinAcross];
outerJoinAcross[sols1_, sols2_] := Flatten[Outer[Join, Normal[sols1, ArrayAssociation], Normal[sols2, ArrayAssociation], 1], 1];


$namedVarPattern = _SPARQLVariable | RDFBlankNode[_?StringQ];


(* split the basic graph pattern into connected components *)
clear[iListConnectedComponents];
iListConnectedComponents[l : {} | {_}, _] := {l};
iListConnectedComponents[l_List, patt_] := With[
	{x = Unique[][#] & /@ l},
	x //
	Subsets[#, {2}] & //
	Cases[{e1_, e2_} /; IntersectingQ[Cases[e1, patt, Infinity], Cases[e2, patt, Infinity]] :> UndirectedEdge[e1, e2]] //
	Graph[x, #] & //
	ConnectedGraphComponents //
	Map[VertexList] //
	Query[All, All, 1]
];


(* order a connected basic graph pattern such that *)
(* 1) selective triple patterns are evaluated earlier and *)
(* 2) outer joins are eliminated *)
clear[iSortListConnectedComponentBy];
iSortListConnectedComponentBy[l : {} | {_}, _, _] := l;
iSortListConnectedComponentBy[l : {_, _}, _, f_] := SortBy[l, f];
iSortListConnectedComponentBy[l_List, patt_, f_] := Module[
	{res = {}, remaining = l, vars = {}, tmp},
	tmp = First[TakeSmallestBy[remaining, f, 1]];
	AppendTo[res, tmp];
	remaining = Delete[remaining, FirstPosition[remaining, tmp]];
	While[
		remaining =!= {},
		vars = DeleteDuplicates[Join[
			vars,
			Cases[tmp, patt, {0, Infinity}]
		]];
		tmp = First[TakeSmallestBy[Select[remaining, Not @* FreeQ[Alternatives @@ vars]], f, 1]];
		AppendTo[res, tmp];
		remaining = Delete[remaining, FirstPosition[remaining, tmp]];
	];
	res
];

(* selectivity: estimated fraction of matches *)
(* 0: hight selectivity *)
(* 1: low selectivity *)
clear[triplePatternSelectivity];
triplePatternSelectivity[RDFTriple[s_, p_, o_]] := Times[
	If[MatchQ[s, _SPARQLVariable], 1., 0.001],
	If[MatchQ[p, _SPARQLVariable], 1., 0.01],
	If[MatchQ[o, _SPARQLVariable], 1., 0.1]
];


(* connected basic graph pattern *)
clear[evaluateConnectedBasicGraphPattern];
evaluateConnectedBasicGraphPattern[el_List, epl_List] := Module[
	{sols, keys},
	sols = evaluateTriplePattern[el, First[epl]];
	keys = DeleteDuplicates[Cases[First[epl], $namedVarPattern]] /. SPARQLVariable[var_] :> var;
	Function[nexttp,
		With[
			{k = DeleteDuplicates[Cases[nexttp, $namedVarPattern]] /. SPARQLVariable[var_] :> var},
			sols = JoinAcross[
				sols,
				evaluateTriplePattern[el, nexttp, sols],
				Key /@ Intersection[keys, k]
			];
			keys = Union[keys, k]
		]
	] /@ Rest[epl];
	sols = KeyDrop[sols, Cases[keys, _RDFBlankNode]];
	sols
];


(* triple pattern *)
clear[evaluateTriplePattern];
(* optimization *)
evaluateTriplePattern[el_List, ep : RDFTriple[Repeated[_SPARQLVariable, {3}]]?DuplicateFreeQ] := ArrayAssociation[List @@@ el, {None, List @@ ep /. SPARQLVariable[var_] :> var}];
(* general *)
evaluateTriplePattern[el_List, ep_RDFTriple, sols_ : None] := With[
	{vars = AssociationMap[Unique[] &, DeleteDuplicates[Cases[ep, $namedVarPattern]]]},
	ArrayAssociation[
		Cases[
			el,
			RuleDelayed @@ {
				Replace[
					equivalentLiteralReplacements[ep],
					{
						v : $namedVarPattern :> Pattern[Evaluate[vars[v]], _],
						RDFBlankNode[] :> _
					},
					{1}
				],
				Values[vars]
			}
		],
		{None, Keys[vars] /. SPARQLVariable[var_] :> var}
	]
];

clear[equivalentLiteralReplacements];
equivalentLiteralReplacements[RDFTriple[s_, p_, o_RDFLiteral]] := With[{simple = FromRDFLiteral[o]}, RDFTriple[s, p, o | simple] /; simple =!= o];
equivalentLiteralReplacements[RDFTriple[s_, p_, Except[_RDFLiteral, o_]]] := With[{lit = ToRDFLiteral[o]}, RDFTriple[s, p, o | lit] /; Head[lit] === RDFLiteral];
equivalentLiteralReplacements[x_] := x;

End[];
EndPackage[];
