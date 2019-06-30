BeginPackage["GraphStore`SPARQL`AggregateFunction`", {"GraphStore`", "GraphStore`SPARQL`"}];
Begin["`Private`"];

EvaluateSPARQLAggregateFunction[args___] := With[{res = Catch[iEvaluateSPARQLAggregateFunction[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iEvaluateSPARQLAggregateFunction];

(* 18.5.1.2 Count *)
iEvaluateSPARQLAggregateFunction["COUNT", sols_, All, OptionsPattern["Distinct" -> False]] := If[OptionValue["Distinct"],
	CountDistinct[sols],
	Length[sols],
	fail[]
];
iEvaluateSPARQLAggregateFunction["COUNT", _, exprList_List, OptionsPattern["Distinct" -> False]] := If[OptionValue["Distinct"],
	CountDistinct[exprList],
	Length[exprList],
	fail[]
];
(* 18.5.1.3 Sum *)
iEvaluateSPARQLAggregateFunction["SUM", _, exprList_List, OptionsPattern["Distinct" -> False]] := If[OptionValue["Distinct"],
	Total[Flatten[DeleteDuplicates[exprList], 1]],
	Total[Flatten[exprList, 1]],
	fail[]
];
(* 18.5.1.4 Avg *)
iEvaluateSPARQLAggregateFunction["AVG", _, {}, OptionsPattern[]] := 0;
iEvaluateSPARQLAggregateFunction["AVG", sols_, exprList_, opts : OptionsPattern[]] := Replace[
	iEvaluateSPARQLAggregateFunction["SUM", sols, exprList, opts] / iEvaluateSPARQLAggregateFunction["COUNT", sols, exprList, opts],
	{
		i_Integer :> i,
		x_?NumericQ :> N[x],
		_ :> $Failed
	}
];
(* 18.5.1.5 Min *)
iEvaluateSPARQLAggregateFunction["MIN", _, {}, OptionsPattern[]] := $Failed;
iEvaluateSPARQLAggregateFunction["MIN", _, exprList : {__}, OptionsPattern[]] := Min[Flatten[exprList, 1]];
(* 18.5.1.6 Max *)
iEvaluateSPARQLAggregateFunction["MAX", _, {}, OptionsPattern[]] := $Failed;
iEvaluateSPARQLAggregateFunction["MAX", _, exprList : {__}, OptionsPattern[]] := Max[Flatten[exprList, 1]];
(* 18.5.1.7 GroupConcat *)
iEvaluateSPARQLAggregateFunction["GROUP_CONCAT", _, exprList_List, OptionsPattern[{"Distinct" -> False, "Separator" -> " "}]] := StringRiffle[
	Flatten[
		If[OptionValue["Distinct"],
			DeleteDuplicates[exprList],
			exprList,
			fail[]
		],
		1
	],
	OptionValue["Separator"]
];
(* 18.5.1.8 Sample *)
iEvaluateSPARQLAggregateFunction["SAMPLE", _, {}, OptionsPattern[]] := $Failed;
iEvaluateSPARQLAggregateFunction["SAMPLE", _, exprList : {__}, OptionsPattern[]] := RandomChoice[Flatten[exprList, 1]];

iEvaluateSPARQLAggregateFunction[SPARQLEvaluation[f_String], args___] := iEvaluateSPARQLAggregateFunction[f, args];
iEvaluateSPARQLAggregateFunction[f_String, args___] := With[{u = ToUpperCase[f]}, iEvaluateSPARQLAggregateFunction[u, args] /; u =!= f];

iEvaluateSPARQLAggregateFunction[SPARQLEvaluation[f_], _, exprList_List, opts : OptionsPattern[]] := f[exprList];


SPARQLAggregateFunctionQ[_String?(StringMatchQ[Alternatives[
	"COUNT",
	"SUM",
	"MIN",
	"MAX",
	"AVG",
	"GROUP_CONCAT",
	"SAMPLE"
], IgnoreCase -> True])] := True;
SPARQLAggregateFunctionQ[SPARQLEvaluation[f_String]] := SPARQLAggregateFunctionQ[f];
SPARQLAggregateFunctionQ[SPARQLEvaluation[_SPARQLDistinct /* f_String]] := SPARQLAggregateFunctionQ[f];
SPARQLAggregateFunctionQ[_] := False;


End[];
EndPackage[];
