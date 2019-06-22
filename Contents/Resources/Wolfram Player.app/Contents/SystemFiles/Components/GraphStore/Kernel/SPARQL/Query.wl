BeginPackage["GraphStore`SPARQL`Query`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`ArrayAssociation`"];
Needs["GraphStore`RDF`"];

$Base = None;
$EntailmentRegime = Automatic;

Begin["`Private`"];

SPARQLAsk[args___][g_] := With[{res = Catch[iSPARQLAsk[g, args], $failTag]}, res /; res =!= $failTag];

SPARQLConstruct[args___][g_] := With[{res = Catch[iSPARQLConstruct[g, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLQuery] = {
	"Base" -> Automatic,
	SPARQLEntailmentRegime -> Automatic,
	"From" -> Automatic,
	"FromNamed" -> Automatic
};
SPARQLQuery[args___][g_] := With[{res = Catch[iSPARQLQuery[g, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLSelect] = {
	"Distinct" -> False,
	"Limit" -> Infinity,
	"Offset" -> 0,
	"OrderBy" -> None,
	"Reduced" -> False
};
SPARQLSelect[args___][g_] := With[{res = Catch[iSPARQLSelect[g, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLService] = {
	"Silent" -> False
};

op_SPARQLValues[data_] := With[
	{res = EvaluateAlgebraExpression[data, OptimizeAlgebraExpression[ToAlgebraExpression[op]]]},
	DeleteCases[res, $Failed, {2}] /; ! MatchQ[res, _EvaluateAlgebraExpression]
];


fail[___, f_Failure, ___] := Throw[f, $failTag];
fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


PossibleQueryQ[_SPARQLAggregate | _SPARQLAsk | _SPARQLConstruct | _SPARQLQuery | _SPARQLSelect | _SPARQLValues] := True;
PossibleQueryQ[_SPARQLOrderBy | _SPARQLProject | _SPARQLDistinct | _SPARQLLimit] := True;
PossibleQueryQ[comp : _Composition | _RightComposition] := AllTrue[comp, PossibleQueryQ];
PossibleQueryQ[_] := False;


ChooseRDFStore[{Automatic, Automatic}, gs_] := gs;
ChooseRDFStore[{from_, fromNamed_}, store_RDFStore] := RDFStore[
	Switch[from,
		Automatic, {},
		_List, First[RDFMerge[Replace[Lookup[store["NamedGraphs"], #, Import[#]["DefaultGraph"]], l_List :> RDFStore[l]] & /@ from]],
		_, Lookup[store["NamedGraphs"], from, Import[from]["DefaultGraph"]]
	],
	Switch[fromNamed,
		Automatic, <||>,
		_List, AssociationMap[Lookup[store["NamedGraphs"], #, Import[#]["DefaultGraph"]] &, fromNamed],
		_, <|fromNamed -> Lookup[store["NamedGraphs"], fromNamed, Import[fromNamed]["DefaultGraph"]]|>
	]
];
ChooseRDFStore[{from_, named_}, Except[_RDFStore]] := RDFStore[from, named];


(* -------------------------------------------------- *)
(* query *)

clear[iSPARQLQuery];
Options[iSPARQLQuery] = Options[SPARQLQuery];
iSPARQLQuery[g_, query : _String | _File | _IRI | _URL, opts : OptionsPattern[]] := iSPARQLQuery[
	g,
	Replace[
		If[StringQ[query], ImportString, Import][query, "SPARQLQuery"],
		SPARQLQuery[q__] :> SPARQLQuery[q, opts]
	],
	opts
];
iSPARQLQuery[g_, query_, OptionsPattern[]] := Block[{
	$Base = getBase[{g}, OptionValue["Base"]],
	$EntailmentRegime = OptionValue[SPARQLEntailmentRegime] // Replace[
		Automatic :> $EntailmentRegime
	]
},
	Module[
		{queryexp, from, fromNamed},
		queryexp = query;
		{from, fromNamed} = OptionValue[{"From", "FromNamed"}];
		If[MatchQ[queryexp, _Composition | _RightComposition],
			queryexp = SPARQLSelect[queryexp]
		];
		queryexp = queryexp /. l_RDFLiteral :> FromRDFLiteral[l];
		ChooseRDFStore[{from, fromNamed}, g] // queryexp // Replace[_queryexp :> fail[]]
	]
];

clear[getBase];
getBase[gl_List, base_] := If[base === Automatic,
	gl // Replace[{
		{file : _File | _IRI | _URL} :> file,
		_ :> None
	}],
	base
];

(* end query *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* query forms *)


(* select *)
clear[iSPARQLSelect];
Options[iSPARQLSelect] = Options[SPARQLSelect];
iSPARQLSelect[g_, gpse_, opts : OptionsPattern[]] := SPARQLFromLegacySyntax[SPARQLSelect[gpse, opts]][g] /; MatchQ[gpse, _Rule | RuleDelayed] || {opts} =!= {};
iSPARQLSelect[g_, gp_] := Block[
	{$EntailmentRegime = Replace[
		$EntailmentRegime,
		Automatic :> Lookup[Options[g], SPARQLEntailmentRegime, None]
	]},
	Module[
		(* res1 instead of res: https://bugs.wolfram.com/show?number=342958 *)
		{res1, alg},

		alg = SPARQLSelect[gp] // ToAlgebraExpression // OptimizeAlgebraExpression;

		res1 = FirstCase[
			SPARQLAlgebraEvaluators[],
			(_?(Curry[MatchQ, 2][g]) -> handler_) :> handler,
			EvaluateAlgebraExpression
		][g, alg] // Replace[{
			f_?FailureQ :> fail[f],
			Except[_ArrayAssociation | {___Association}] :> fail[]
		}];
		(* remove temporary variables (SPARQLVariable[Unique[]]) *)
		With[{tempvars = DeleteDuplicates[Cases[alg, SPARQLVariable[s_Symbol] :> s, Infinity]]},
			If[tempvars =!= {},
				res1 = KeyDrop[res1, tempvars];
			]
		];
		res1 = Normal[res1, ArrayAssociation];
		res1 = EvaluateHeldBNodes[res1];
		If[! FreeQ[res1, $Failed],
			res1 = DeleteCases[res1, $Failed, {2}]
		];

		res1
	]
];


(* construct *)
clear[iSPARQLConstruct];
iSPARQLConstruct[g_, (Rule | RuleDelayed)[gp_, template_]] := With[
	{t = Flatten[{template}]},
	RDFStore[Select[
		Union @@ Function[Replace[t, #, {2}]] /@ (iSPARQLSelect[g, gp] // Map[KeyMap[Replace[var_String :> SPARQLVariable[var]]]]),
		Not @* MemberQ[_SPARQLVariable | $Failed]
	]]
];
iSPARQLConstruct[g_, Except[_Rule | _RuleDelayed, template_]] := iSPARQLConstruct[g, template :> template];


(* ask *)
clear[iSPARQLAsk];
iSPARQLAsk[g_, gp_] := iSPARQLQuery[
	g,
	SPARQLSelect[gp] /*
	SPARQLLimit[1]
] =!= {};

(* end query forms *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
