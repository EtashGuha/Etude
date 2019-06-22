BeginPackage["GraphStore`SubsetCases`"];

SubsetCases;

Begin["`Private`"];

SubsetCases[args___] := With[{res = Catch[iSubsetCases[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iSubsetCases];

(* general patterns *)
iSubsetCases[l_, patt_?variableLengthQ] := Cases[Subsets[l], applyOrderless[patt]];

(* alternatives *)
iSubsetCases[l_, a_Alternatives] := Join @@ Function[iSubsetCases[l, #]] /@ a;
iSubsetCases[l_, (rule : Rule | RuleDelayed)[a_Alternatives, rhs_]] := Join @@ Function[iSubsetCases[l, rule[#, rhs]]] /@ a;

(* condition *)
iSubsetCases[l_, Verbatim[Condition][patt_, cond_]] := Pick[
	iSubsetCases[l, patt],
	iSubsetCases[l, patt :> TrueQ[cond]]
];
iSubsetCases[l_, (rule : Rule | RuleDelayed)[Verbatim[Condition][patt_, cond_], rhs_]] := iSubsetCases[l, rule[patt, rhs /; cond]];

(* pattern *)
iSubsetCases[l_, (rule : Rule | RuleDelayed)[Verbatim[Pattern][name_Symbol, patt_], rhs_]] := MapThread[
	ReleaseHold[# /. name -> #2] &,
	{
		iSubsetCases[l, rule[patt, Hold[rhs]]],
		iSubsetCases[l, patt]
	}
];
iSubsetCases[l_, Verbatim[Pattern][_Symbol, patt_]] := iSubsetCases[l, patt];

(* rule *)
iSubsetCases[l_, rule : (Rule | RuleDelayed)[patt_, _]] := Cases[
	l[[#]] & /@ subsetPosition[l, patt],
	rule
];

(* other pattern types *)
iSubsetCases[l_, Except[_Rule | _RuleDelayed, patt_]] := l[[#]] & /@ subsetPosition[l, patt];


clear[variableLengthQ];
variableLengthQ[(Rule | RuleDelayed)[patt_, _]] := variableLengthQ[patt];
variableLengthQ[Verbatim[Condition][patt_, _]] := variableLengthQ[patt];
variableLengthQ[Verbatim[PatternTest][p_, _]] := variableLengthQ[p];
variableLengthQ[patt_] := AnyTrue[patt, variableLengthSequenceQ];

clear[variableLengthSequenceQ];
variableLengthSequenceQ[a_Alternatives] := AnyTrue[a, variableLengthSequenceQ];
variableLengthSequenceQ[_BlankSequence] := True;
variableLengthSequenceQ[_BlankNullSequence] := True;
variableLengthSequenceQ[_Repeated] := True;
variableLengthSequenceQ[_RepeatedNull] := True;
variableLengthSequenceQ[Verbatim[Condition][patt_, _]] := variableLengthSequenceQ[patt];
variableLengthSequenceQ[Verbatim[Pattern][_, obj_]] := variableLengthSequenceQ[obj];
variableLengthSequenceQ[Verbatim[PatternTest][p_, _]] := variableLengthSequenceQ[p];
variableLengthSequenceQ[_] := False;

clear[applyOrderless];
applyOrderless[(rule : Rule | RuleDelayed)[patt_, rhs_]] := rule[applyOrderless[patt], rhs];
applyOrderless[Verbatim[Condition][patt_, test_]] := With[{p = applyOrderless[patt]}, Condition @@ Hold[p, test]];
applyOrderless[Verbatim[PatternTest][p_, test_]] := PatternTest[applyOrderless[p], test];
applyOrderless[h_[patt__]] := h[OrderlessPatternSequence[patt]];
applyOrderless[patt_] := patt;

clear[subsetPosition];
subsetPosition[_, _[]] := {{}};
subsetPosition[l_, _[p_]] := position[l, p];
subsetPosition[l_, pl_] := Module[
	{res},
	res = position[l, First[pl]];
	Do[
		res = Select[
			Join @@@ Tuples[{res, position[l, pl[[i]]]}],
			With[
				{pli = pl[[;; i]]},
				Function[pos,
					And[
						DuplicateFreeQ[pos],
						MatchQ[l[[pos]], pli]
					]
				]
			]
		],
		{i, 2, Length[pl]}
	];
	res = DeleteDuplicatesBy[res, Sort];
	res
];

clear[position];
position[expr_, patt_] := Position[expr, patt, {1}, Heads -> False];

End[];
EndPackage[];
