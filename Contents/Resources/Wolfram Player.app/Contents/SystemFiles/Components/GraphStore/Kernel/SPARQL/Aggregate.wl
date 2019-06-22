BeginPackage["GraphStore`SPARQL`Aggregate`", {"GraphStore`", "GraphStore`SPARQL`"}];

Begin["`Private`"];

op_SPARQLAggregate[data_] := With[
	{res = EvaluateAlgebraExpression[data, OptimizeAlgebraExpression[ToAlgebraExpression[op]]]},
	DeleteCases[res, $Failed, {2}] /; ! MatchQ[res, _EvaluateAlgebraExpression]
];

End[];
EndPackage[];
