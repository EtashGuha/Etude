BeginPackage["GraphStore`SPARQL`SolutionModifiers`", {"GraphStore`", "GraphStore`SPARQL`"}];

Options[SPARQLDistinct] = {
	Method -> "Distinct"
};

Begin["`Private`"];

Function[symbol,
	op_symbol[data_] := With[
		{res = EvaluateAlgebraExpression[data, OptimizeAlgebraExpression[ToAlgebraExpression[op]]]},
		EvaluateHeldBNodes[DeleteCases[res, $Failed, {2}]] /; ! MatchQ[res, _EvaluateAlgebraExpression]
	];
] /@ {
	SPARQLDistinct,
	SPARQLLimit,
	SPARQLProject,
	SPARQLOrderBy
};

End[];
EndPackage[];
