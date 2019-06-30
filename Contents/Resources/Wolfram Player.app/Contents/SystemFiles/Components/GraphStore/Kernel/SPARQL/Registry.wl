BeginPackage["GraphStore`SPARQL`Registry`", {"GraphStore`", "GraphStore`SPARQL`"}];
Begin["`Private`"];

$evaluators = {};

SPARQLAlgebraEvaluatorRegister[patt_, handler_] := (
	PrependTo[$evaluators, patt -> handler];
	Null
);
SPARQLAlgebraEvaluators[] := $evaluators;
SPARQLAlgebraEvaluatorUnregister[patt_] := (
	$evaluators = DeleteCases[$evaluators, Verbatim[patt] -> _, {1}, 1];
	Null
);

End[];
EndPackage[];
