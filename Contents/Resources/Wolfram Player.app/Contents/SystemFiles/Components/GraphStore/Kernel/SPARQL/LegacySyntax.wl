BeginPackage["GraphStore`SPARQL`LegacySyntax`", {"GraphStore`", "GraphStore`SPARQL`"}];
Begin["`Private`"];

(* from legacy to modern *)
SPARQLFromLegacySyntax[expr_] := expr //. {
	SPARQLEvaluation[f_][args___, opts : OptionsPattern[]] /; FilterRules[{opts}, "Distinct"] =!= {} :> With[
		{tmp = SPARQLEvaluation[If[TrueQ[Lookup[FilterRules[{opts}, "Distinct"], "Distinct"]], SPARQLDistinct[], Identity] /* f][
			args,
			Sequence @@ FilterRules[{opts}, Except["Distinct"]]
		]},
		tmp /; True
	],

	SPARQLSelect[
		where_,
		opts : OptionsPattern[]
	] /; Or[
		MatchQ[where, _Rule | _RuleDelayed],
		{opts} =!= {}
	] :> With[
		{tmp = With[
			{legacyOpts = {
				"Distinct" -> False,
				"Limit" -> Infinity,
				"Offset" -> 0,
				"OrderBy" -> None,
				"Reduced" -> False
			}},
			RightComposition[
				SPARQLSelect[If[MatchQ[where, _Rule | _RuleDelayed], First[where], where]],
				OptionValue[legacyOpts, {opts}, "OrderBy"] // Replace[{
					None :> Identity,
					x_ :> SPARQLOrderBy[x]
				}],
				If[MatchQ[where, _Rule | _RuleDelayed],
					SPARQLProject[Last[where]],
					Identity
				],
				If[TrueQ[OptionValue[legacyOpts, {opts}, "Distinct"]],
					SPARQLDistinct[],
					Identity
				],
				If[TrueQ[OptionValue[legacyOpts, {opts}, "Reduced"]],
					SPARQLDistinct[Method -> "Reduced"],
					Identity
				],
				SPARQLLimit @@ OptionValue[legacyOpts, {opts}, {"Limit", "Offset"}]
			]
		]},
		tmp /; True
	],

	(head : Rule | RuleDelayed)[SPARQLVariable[x_], Except["Ascending" | "Descending", y_]] :> head[x, y],

	(head : SPARQLAggregate | SPARQLProject | SPARQLValues)[
		vars : _SPARQLVariable | _List?(MemberQ[_SPARQLVariable]),
		rest___
	] :> With[
		{tmp = head[
			Replace[
				vars,
				SPARQLVariable[x_] :> x,
				{Boole[ListQ[vars]]}
			],
			rest
		]},
		tmp /; True
	]
};

(* from modern to legacy *)
SPARQLFromSolutionModifierOperatorSyntax[expr_] := expr //. {
	SPARQLSelect[select_SPARQLSelect] :> select,
	(select : SPARQLSelect[patt_, opts : OptionsPattern[]] | _) /* SPARQLOrderBy[order_] :> SPARQLSelect[
		Sequence @@ If[{patt} =!= {} && FilterRules[{opts}, "OrderBy"] === {},
			{patt, opts},
			{select}
		],
		"OrderBy" -> order
	],
	(select : SPARQLSelect[patt_, opts : OptionsPattern[]] | _) /* SPARQLProject[proj_] :> SPARQLSelect[
		Sequence @@ If[{patt} =!= {} && ! MatchQ[patt, _Rule | _RuleDelayed],
			{patt -> proj, opts},
			{select -> proj}
		]
	],
	(select : SPARQLSelect[patt_, opts : OptionsPattern[]] | _) /* SPARQLDistinct[dopts : OptionsPattern[]] :> SPARQLSelect[
		Sequence @@ If[{patt} =!= {} && FilterRules[{opts}, {"Distinct", "Reduced"}] === {},
			{patt, opts},
			{select}
		],
		OptionValue[SPARQLDistinct, {dopts}, Method] -> True
	],
	(select : SPARQLSelect[patt_, opts : OptionsPattern[]] | _) /* SPARQLLimit[l_, o_ : 0] :> SPARQLSelect[
		Sequence @@ If[{patt} =!= {} && FilterRules[{opts}, {"Offset", "Limit"}] === {},
			{patt, opts},
			{select}
		],
		Sequence @@ If[o === 0, {}, {"Offset" -> o}],
		"Limit" -> l
	]
};

End[];
EndPackage[];
