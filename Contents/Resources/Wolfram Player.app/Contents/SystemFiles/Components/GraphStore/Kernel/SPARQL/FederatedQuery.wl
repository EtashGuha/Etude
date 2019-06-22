BeginPackage["GraphStore`SPARQL`FederatedQuery`", {"GraphStore`", "GraphStore`SPARQL`"}];

Options[EvaluateFederatedQuery] = {
	"Silent" -> False
};

Begin["`Private`"];

EvaluateFederatedQuery[args___] := With[{res = Catch[iEvaluateFederatedQuery[args], $failTag]}, res /; res =!= $failTag]


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iEvaluateFederatedQuery];
Options[iEvaluateFederatedQuery] = Options[EvaluateFederatedQuery];
iEvaluateFederatedQuery[url_, pattern_, OptionsPattern[]] := SPARQLExecute[url, SPARQLSelect[pattern]] // Replace[{
	_SPARQLExecute :> If[TrueQ[OptionValue["Silent"]], {<||>}, fail[]]
}];


End[];
EndPackage[];
