BeginPackage["GraphStore`SPARQL`"];

ChooseRDFStore;
EvaluateAlgebraExpression;
EvaluateBasicGraphPattern;
EvaluateBasicGraphPatternWithEntailment;
EvaluateFederatedQuery;
EvaluateHeldBNodes;
EvaluatePropertyPath;
EvaluateSPARQLAggregateFunction;
EvaluateSPARQLFunction;
EvaluateSPARQLOperator;
HeldBNode;
ListConnectedComponents;
OptimizeAlgebraExpression;
PossibleQueryQ;
PossibleUpdateQ;
PropertyPathExpressionQ;
SortListConnectedComponentBy;
SPARQLAggregateFunctionQ;
SPARQLAlgebraEvaluatorRegister;
SPARQLAlgebraEvaluators;
SPARQLAlgebraEvaluatorUnregister;
SPARQLDistinct;
SPARQLEntailmentRegime;
SPARQLFromLegacySyntax;
SPARQLFromSolutionModifierOperatorSyntax;
SPARQLJoinAcross;
SPARQLLimit;
SPARQLOrderBy;
SPARQLProject;
SPARQLResultsEqual;
ToAlgebraExpression;
ToSPARQL;
$Base;
$EntailmentRegime;

Begin["`Private`"];

With[
	{path = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{path, #}]] & /@ {
		"AggregateFunction.wl",
		"Aggregate.wl",
		"Algebra.wl",
		"BasicGraphPattern.wl",
		"Entailment.wl",
		"Evaluation.wl",
		"FederatedQuery.wl",
		"Function.wl",
		"HeldBNode.wl",
		"JoinAcross.wl",
		"LegacySyntax.wl",
		"Operator.wl",
		"PropertyPath.wl",
		"Protocol.wl",
		"Query.wl",
		"Registry.wl",
		"ResultsEqual.wl",
		"SolutionModifiers.wl",
		"ToSPARQL.wl",
		"Update.wl"
	};
];

End[];
EndPackage[];
