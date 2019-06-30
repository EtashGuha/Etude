BeginPackage["GraphStore`SPARQL`HeldBNode`", {"GraphStore`", "GraphStore`SPARQL`"}];
Begin["`Private`"];

EvaluateHeldBNodes[args___] := With[{res = Catch[iEvaluateHeldBNodes[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iEvaluateHeldBNodes];
iEvaluateHeldBNodes[sols_List] := Module[
	{bn},
	Function[sol,
		bn = <||>;
		sol /. HeldBNode[s_String] :> With[{tmp = Lookup[bn, s, bn[[s]] = EvaluateSPARQLFunction["BNODE"]]}, tmp /; True]
	] /@ sols
];

End[];
EndPackage[];
