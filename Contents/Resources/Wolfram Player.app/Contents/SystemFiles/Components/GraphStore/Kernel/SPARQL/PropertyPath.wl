BeginPackage["GraphStore`SPARQL`PropertyPath`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`SPARQL`Algebra`"];

Begin["`Private`"];

EvaluatePropertyPath[args___] := With[{res = Catch[iEvaluatePropertyPath[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iEvaluatePropertyPath];

(* predicate property path *)
iEvaluatePropertyPath[g_, path[SPARQLVariable[x_], link[iri_], SPARQLVariable[y_]]] := <|x -> #1, y -> #2|> & @@@ Cases[g, If[x === y, RDFTriple[a_, iri, a_] :> {a, a}, RDFTriple[a_, iri, b_] :> {a, b}]];
iEvaluatePropertyPath[g_, path[SPARQLVariable[x_], link[iri_], Except[_SPARQLVariable, y_]]] := <|x -> #|> & /@ Cases[g, RDFTriple[a_, iri, y] :> a];
iEvaluatePropertyPath[g_, path[Except[_SPARQLVariable, x_], link[iri_], SPARQLVariable[y_]]] := <|y -> #|> & /@ Cases[g, RDFTriple[x, iri, a_] :> a];
iEvaluatePropertyPath[g_, path[Except[_SPARQLVariable, x_], link[iri_], Except[_SPARQLVariable, y_]]] := If[MemberQ[g, RDFTriple[x, iri, y]], {<||>}, {}];

(* inverse property path *)
iEvaluatePropertyPath[g_, path[x_, inv[p_], y_]] := iEvaluatePropertyPath[g, path[y, p, x]];

(* sequence property path *)
iEvaluatePropertyPath[g_, path[x_, seq[p_, q_], y_]] := GraphStore`SPARQL`Evaluation`Private`evalProject[
	Module[
		{var = Unique[]},
		SPARQLJoinAcross[
			iEvaluatePropertyPath[g, path[x, p, SPARQLVariable[var]]],
			iEvaluatePropertyPath[g, path[SPARQLVariable[var], q, y]]
		]
	],
	Cases[{x, y}, SPARQLVariable[var_] :> var]
];

(* alternative property path *)
iEvaluatePropertyPath[g_, path[x_, a_alt, y_]] := GraphStore`SPARQL`Evaluation`Private`evalUnion @@ Function[iEvaluatePropertyPath[g, path[x, #, y]]] /@ a;

(* zero or one path *)
iEvaluatePropertyPath[g_, path[Except[_SPARQLVariable, x_], zeroOrOnePath[p_], SPARQLVariable[y_]]] := Prepend[iEvaluatePropertyPath[g, path[x, p, SPARQLVariable[y]]], <|y -> x|>] // DeleteDuplicates;
iEvaluatePropertyPath[g_, path[SPARQLVariable[x_], zeroOrOnePath[p_], Except[_SPARQLVariable, y_]]] := Prepend[iEvaluatePropertyPath[g, path[SPARQLVariable[x], p, y]], <|x -> y|>] // DeleteDuplicates;
iEvaluatePropertyPath[g_, path[Except[_SPARQLVariable, x_], zeroOrOnePath[p_], Except[_SPARQLVariable, y_]]] := If[x === y || iEvaluatePropertyPath[g, path[x, p, y]] =!= {}, {<||>}, {}];
iEvaluatePropertyPath[g_, path[SPARQLVariable[x_], zeroOrOnePath[p_], SPARQLVariable[y_]]] := Join[
	<|x -> #, y -> #|> & /@ nodes[g],
	iEvaluatePropertyPath[g, path[SPARQLVariable[x], p, SPARQLVariable[y]]]
] // DeleteDuplicates;

(* zero or more path *)
iEvaluatePropertyPath[g_, path[Except[_SPARQLVariable, x_], zeroOrMorePath[p_], SPARQLVariable[y_]]] := Prepend[
	iEvaluatePropertyPath[g, path[x, oneOrMorePath[p], SPARQLVariable[y]]],
	<|y -> x|>
];
iEvaluatePropertyPath[g_, path[SPARQLVariable[x_], zeroOrMorePath[p_], SPARQLVariable[y_]]] := Join[
	<|x -> #, y -> #|> & /@ nodes[g],
	iEvaluatePropertyPath[g, path[SPARQLVariable[x], oneOrMorePath[p], SPARQLVariable[y]]]
] // DeleteDuplicates;
iEvaluatePropertyPath[g_, path[SPARQLVariable[x_], zeroOrMorePath[p_], Except[_SPARQLVariable, y_]]] := iEvaluatePropertyPath[g, path[y, zeroOrMorePath[inv[p]], SPARQLVariable[x]]];
iEvaluatePropertyPath[g_, path[Except[_SPARQLVariable, x_], zeroOrMorePath[p_], Except[_SPARQLVariable, y_]]] := If[x === y, {<||>}, iEvaluatePropertyPath[g, path[x, oneOrMorePath[p], y]]];

(* one or more path *)
iEvaluatePropertyPath[g_, path[Except[_SPARQLVariable, x_], oneOrMorePath[p_], SPARQLVariable[y_]]] := Module[
	{visited = {}},
	ALP[g, #, p, visited] & /@ pathEnds[g, x, p];
	<|y -> #|> & /@ visited
];
iEvaluatePropertyPath[g_, path[SPARQLVariable[x_], oneOrMorePath[p_], SPARQLVariable[y_]]] := Join @@ Function[n,
	iEvaluatePropertyPath[g, path[n, oneOrMorePath[p], SPARQLVariable[y]]] // Map[
		Prepend[x -> n]
	]
] /@ nodes[g];
iEvaluatePropertyPath[g_, path[SPARQLVariable[x_], oneOrMorePath[p_], Except[_SPARQLVariable, y_]]] := iEvaluatePropertyPath[g, path[y, oneOrMorePath[inv[p]], SPARQLVariable[x]]];
iEvaluatePropertyPath[g_, path[Except[_SPARQLVariable, x_], oneOrMorePath[p_], Except[_SPARQLVariable, y_]]] := If[
	Or[
		iEvaluatePropertyPath[g, path[x, p, y]] =!= {},
		MemberQ[
			Module[
				{var = Unique[]},
				iEvaluatePropertyPath[g, path[x, oneOrMorePath[p], SPARQLVariable[var]]][[All, Key[var]]]
			],
			y
		]
	],
	{<||>},
	{}
];

(* negated property set *)
iEvaluatePropertyPath[g_, path[x_, nps_NPS, y_]] := evalNPS[g, {x, nps, y}];


clear[nodes];
nodes[g_] := DeleteDuplicates[Join[g[[All, 1]], g[[All, 3]]]];

clear[pathEnds];
pathEnds[g_, start_, p_] := Module[
	{end = Unique[]},
	iEvaluatePropertyPath[g, path[start, p, SPARQLVariable[end]]][[All, Key[end]]]
];

clear[ALP];
SetAttributes[ALP, HoldAll];
ALP[g_, start_, p_, visited_Symbol] := If[! MemberQ[visited, start],
	AppendTo[visited, start];
	ALP[g, #, p, visited] & /@ pathEnds[g, start, p]
];

clear[variableToPatternRules];
variableToPatternRules[vars_?AssociationQ] := {
	v : _SPARQLVariable | RDFBlankNode[_?StringQ] :> Pattern[Evaluate[vars[v]], _],
	RDFBlankNode[] :> _
};

clear[patternToSolutionRule];
patternToSolutionRule[patt_, vars_List, syms_List] := patt :> KeySelect[
	AssociationThread[vars, syms],
	Not @* MatchQ[_RDFBlankNode]
];

clear[evalNPS];
evalNPS[g_, {x_, NPS[i : {(_File | _IRI | _URL) ...}], y_}] := Cases[
	g,
	With[
		{vars = AssociationMap[
			Unique[] &,
			DeleteDuplicates[Cases[{x, y}, _SPARQLVariable | RDFBlankNode[_?StringQ]]]
		]},
		patternToSolutionRule[
			Replace[
				RDFTriple[x, Except[Alternatives @@ i], y],
				variableToPatternRules[vars],
				{1}
			],
			Keys[vars] /. SPARQLVariable[var_] :> var,
			Values[vars]
		]
	]
];


End[];
EndPackage[];
