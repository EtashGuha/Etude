BeginPackage["GraphStore`Entity`Evaluation`", {"GraphStore`Entity`"}];

Needs["GraphStore`SPARQL`"];
Needs["GraphStore`SPARQL`Algebra`"];

Begin["`Private`"];

EntityEvaluateAlgebraExpression[args___] := With[{res = Catch[iEntityEvaluateAlgebraExpression[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iEntityEvaluateAlgebraExpression];
iEntityEvaluateAlgebraExpression[ent : HoldPattern[Entity[_String] | {Entity[_String] ..}], exprIn_] := Module[
	{expr = exprIn, pl = {}, epl = {}, el = {}},
	expr = expr /. i_IRI :> EntityFromIRI[i];
	(* property paths *)
	pl = Join[
		pl,
		Flatten[
			Replace[
				DeleteDuplicates[Cases[expr, _path, {0, Infinity}]],
				{
					path[_, p_, _] :> Cases[p, _EntityProperty, {0, Infinity}]
				},
				{1}
			]
		]
	];
	(* basic graph patterns *)
	pl = Join[
		pl,
		Flatten[
			Cases[expr, BGP[l_List] :> l, {0, Infinity}] // DeleteDuplicates // Map[Function[l,
				With[{edges = evalBGP[l]},
					If[ListQ[edges],
						el = Join[el, edges]; Nothing,
						Replace[l, {
							RDFTriple[e_Entity, p_EntityProperty, _] :> (AppendTo[epl, {e, p}]; Nothing),
							RDFTriple[e_Entity, _SPARQLVariable, _] :> (AppendTo[epl, {e}]; Nothing),
							RDFTriple[_, p_EntityProperty, _] :> p,
							_ :> Return[EvaluateAlgebraExpression[EntityRDFStore[ent], expr], Module]
						}, {1}]
					]
				]
			]]
		]
	];
	pl = DeleteDuplicates[pl];
	(* data of properties *)
	el = Join[
		el,
		First[EntityRDFStore[ent, pl]]
	];
	(* data of entity-property pairs *)
	el = Join[
		el,
		Join @@ First /@ EntityRDFStore @@@ epl
	];
	EvaluateAlgebraExpression[RDFStore[el], expr]
];

clear[evalBGP];
evalBGP[l : {Repeated[RDFTriple[_, HoldPattern[_EntityProperty], _], {2, Infinity}]}] := Module[
	{g, path, lastEnt, lastEl, el = {}},
	g = Graph[
		l,
		Cases[
			Subsets[l, {2}],
			{e1_, e2_} /; IntersectingQ[Cases[e1, _SPARQLVariable], Cases[e2, _SPARQLVariable]] :> UndirectedEdge[e1, e2]
		]
	];
	If[PathGraphQ[g] && AcyclicGraphQ[g],
		path = FindPath[g, Sequence @@ Pick[VertexList[g], VertexDegree[g], 1]] // First;
		If[path[[1, 3]] =!= path[[2, 1]],
			path = Reverse[path];
		];
		If[AnyTrue[Partition[path, 2, 1], #[[1, 3]] =!= #[[2, 1]] &],
			Return[$Failed]
		];
		lastEnt = If[MatchQ[path[[1, 1]], _Entity],
			{path[[1, 1]]},
			{Entity[EntityTypeName[path[[1, 2]]]]}
		];
		Do[
			If[! MatchQ[lastEnt, {___Entity}],
				Return[$Failed]
			];
			lastEl = First[EntityRDFStore[lastEnt, tp[[2]]]];
			el = Join[el, lastEl];
			lastEnt = lastEl[[All, 3]];
			,
			{tp, path}
		];
		Return[el]
	];
	$Failed
];
evalBGP[_] := $Failed;

End[];
EndPackage[];
