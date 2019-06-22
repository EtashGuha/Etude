(* SPARQL 1.1 Query Language *)
(* https://www.w3.org/TR/sparql11-query/ *)

BeginPackage["GraphStore`Formats`SPARQLQuery`", {"GraphStore`"}];

Needs["GraphStore`Formats`SPARQLUpdate`"];
Needs["GraphStore`Formats`Utilities`"];
Needs["GraphStore`IRI`"];
Needs["GraphStore`Parsing`"];
Needs["GraphStore`RDF`"];
Needs["GraphStore`SPARQL`"];

ExportSPARQLQuery;
Options[ExportSPARQLQuery] = {
	"Base" -> None,
	"Indentation" -> "  ",
	"Prefixes" -> <||>
};

ImportSPARQLQuery;
Options[ImportSPARQLQuery] = {
	"Base" -> Automatic
};

ImportSPARQLQueryBase;
ImportSPARQLQueryPrefixes;

$UseSPARQLSolutionModiferOperatorSyntax = False;

Begin["`Private`"];

ExportSPARQLQuery[args___] := Catch[iExportSPARQLQuery[args], $failTag, (Message[Export::fmterr, "SPARQLQuery"]; #) &];
ImportSPARQLQuery[file_, opts : OptionsPattern[]] := Catch[iImportSPARQLQuery[file, FilterRules[{opts}, Options[ImportSPARQLQuery]]], $failTag, (Message[Import::fmterr, "SPARQLQuery"]; #) &];
ImportSPARQLQueryBase[file_, opts : OptionsPattern[]] := {"Base" -> Replace[Import[file, "SPARQLQuery", opts], {SPARQLQuery[__, "Base" -> base_, ___] :> base, _ :> None}]};
ImportSPARQLQueryPrefixes[file_, opts : OptionsPattern[]] := Catch[iImportSPARQLQueryPrefixes[file, FilterRules[{opts}, Options[ImportSPARQLQueryPrefixes]]], $failTag, (Message[Import::fmterr, "SPARQLQuery"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* vocabulary *)

clear[xsd];
xsd[s_String] := IRI["http://www.w3.org/2001/XMLSchema#" <> s];

(* end vocabulary *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* export *)

clear[iExportSPARQLQuery];
Options[iExportSPARQLQuery] = Options[ExportSPARQLQuery];
iExportSPARQLQuery[file_, data_, opts : OptionsPattern[]] := Block[
	{$indentation = OptionValue["Indentation"]},
	Export[
		file,
		queryUnitToString[fromSPARQLPropertyPath[normalizeSyntax[ToSPARQL[SPARQLFromLegacySyntax[data]]]], opts],
		"Text",
		CharacterEncoding -> "UTF-8"
	]
];

clear[iExportSPARQLUpdate];
Options[iExportSPARQLUpdate] = Options[ExportSPARQLUpdate];
iExportSPARQLUpdate[file_, data_, opts : OptionsPattern[]] := Block[
	{$indentation = OptionValue["Indentation"]},
	Export[
		file,
		updateUnitToString[fromSPARQLPropertyPath[normalizeSyntax[ToSPARQL[SPARQLFromLegacySyntax[data]]]], opts],
		"Text",
		CharacterEncoding -> "UTF-8"
	]
];


clear[normalizeSyntax];
normalizeSyntax[expr_] := expr //. {
	c_Composition :> RightComposition @@ Reverse[c]
};

clear[fromSPARQLPropertyPath];
fromSPARQLPropertyPath[x_] := x /. {
	SPARQLPropertyPath[s_, {pathexpr_}, o_] :> RDFTriple[s, pathexpr, o],
	SPARQLPropertyPath[s_, {pathexpr__}, o_] :> RDFTriple[s, PatternSequence[pathexpr], o]
};


clear[indent];
indent[s_String] := $indentation <> StringReplace[s, "\n" :> "\n" <> $indentation];

clear[stringRiffle];
stringRiffle[l_List] := stringRiffle[l, " "];
stringRiffle[l_List, sep_String] := StringJoin[Riffle[l, sep]];

clear[compact];
compact[expr_, prefixes_, base_] := Module[
	{tmp},
	RDFCompactIRIs[
		expr /. RDFLiteral[s_, dt : xsd["integer"] | xsd["decimal"] | xsd["double"] | xsd["boolean"] | xsd["string"]] :> RDFLiteral[s, tmp @@ dt],
		prefixes,
		base
	] /. tmp[dt_] :> IRI[dt]
];

clear[collectPrefixes];
collectPrefixes[expr_] := Sort[DeleteDuplicates[Cases[expr, IRI[{pre_, _}] :> pre, {0, Infinity}]]];


(* [1] *)
clear[queryUnitToString];
queryUnitToString[x_, opts : OptionsPattern[]] := queryToString[x, opts];

(* [2] *)
clear[queryToString];
queryToString[x_, opts : OptionsPattern[]] /; {opts} =!= {} := Function[{prefixes, base},
	Module[
		{tmp},
		tmp = compact[x, prefixes, base];
		StringJoin[{
			prologueToString[If[prefixes === Automatic, AssociationMap[RDFPrefixData, collectPrefixes[tmp]], prefixes], base],
			queryToString[tmp]
		}]
	]
] @@ OptionValue[ExportSPARQLQuery, {opts}, {"Prefixes", "Base"}];
queryToString[SPARQLQuery[RightComposition[query_, values_SPARQLValues], qrest___]] := StringRiffle[{
	queryToString[SPARQLQuery[query, qrest]],
	valuesClauseToString[values]
}, "\n"];
queryToString[x : SPARQLQuery[_SPARQLSelect | _RightComposition, opts : OptionsPattern[]]] := StringJoin[{
	prologueToString[<||>, OptionValue[SPARQLQuery, {opts}, "Base"]],
	selectQueryToString[x]
}];
queryToString[x : SPARQLQuery[_SPARQLConstruct, opts : OptionsPattern[]]] := StringJoin[{
	prologueToString[<||>, OptionValue[SPARQLQuery, {opts}, "Base"]],
	constructQueryToString[x]
}];
queryToString[x : SPARQLQuery[_SPARQLAsk, opts : OptionsPattern[]]] := StringJoin[{
	prologueToString[<||>, OptionValue[SPARQLQuery, {opts}, "Base"]],
	askQueryToString[x]
}];
queryToString[SPARQLQuery[q_String]] := queryToString[Quiet[ImportString[q, "SPARQLQuery"]]];
queryToString[Except[_SPARQLQuery, x_]] := queryToString[SPARQLQuery[x]];

(* [3] *)
clear[updateUnitToString];
updateUnitToString[x_, opts : OptionsPattern[]] := updateToString[x, opts];

(* [4] *)
clear[prologueToString];
prologueToString[prefixes_, base_] := StringJoin[{
	baseDeclToString[base],
	prefixDeclToString[prefixes]
}] // Replace[Except["", s_] :> s <> "\n"];

(* [5] *)
clear[baseDeclToString];
baseDeclToString[base_IRI] := StringRiffle[{"BASE", iriToString[base]}] <> "\n";
baseDeclToString[s_String] := baseDeclToString[IRI[s]];
baseDeclToString[_] := "";

(* [6] *)
clear[prefixDeclToString];
prefixDeclToString[prefix_?AssociationQ] := StringJoin[KeyValueMap[StringRiffle[{"PREFIX", # <> ":", iriToString[Replace[#2, i_String :> IRI[i]]]}] <> "\n" &, prefix]];

(* [7] *)
clear[selectQueryToString];
selectQueryToString[SPARQLQuery[
	Alternatives[
		SPARQLSelect[where_],
		RightComposition[
			SPARQLSelect[where_],
			orderby : Repeated[_SPARQLOrderBy, {0, 1}],
			project : Repeated[_SPARQLProject, {0, 1}],
			distinct : Repeated[_SPARQLDistinct, {0, 1}],
			limit : Repeated[_SPARQLLimit, {0, 1}]
		]
	],
	opts : OptionsPattern[]
]] := StringJoin[{
	selectClauseToString[First[{project}, Identity], First[{distinct}, Identity]],
	datasetClauseToString[OptionValue[SPARQLQuery, {opts}, {"From", "FromNamed"}]],
	whereClauseToString[where],
	solutionModifierToString[orderby /* limit]
}];
selectQueryToString[SPARQLQuery[
	RightComposition[
		SPARQLSelect[where_],
		distinct : Repeated[_SPARQLDistinct, {0, 1}],
		agg_SPARQLAggregate,
		limit : Repeated[_SPARQLLimit, {0, 1}]
	],
	opts : OptionsPattern[]
]] := StringJoin[{
	selectClauseToString[SPARQLProject[First[agg]], First[{distinct}, Identity]],
	datasetClauseToString[OptionValue[SPARQLQuery, {opts}, {"From", "FromNamed"}]],
	whereClauseToString[where],
	solutionModifierToString[agg /* limit]
}];
selectQueryToString[SPARQLQuery[
	RightComposition[
		Except[_SPARQLSelect, x_],
		outer : Alternatives[
			PatternSequence[
				Repeated[_SPARQLOrderBy, {0, 1}],
				Repeated[_SPARQLProject, {0, 1}],
				Repeated[_SPARQLDistinct, {0, 1}],
				Repeated[_SPARQLLimit, {0, 1}]
			],
			PatternSequence[
				Repeated[_SPARQLDistinct, {0, 1}],
				_SPARQLAggregate,
				Repeated[_SPARQLLimit, {0, 1}]
			]
		] /; {outer} =!= {}
	],
	opts : OptionsPattern[]
]] := selectQueryToString[SPARQLQuery[
	SPARQLSelect[x] /* outer,
	opts
]];

(* [8] *)
clear[subSelectToString];
subSelectToString[RightComposition[select_, values_SPARQLValues]] := StringRiffle[{
	subSelectToString[select],
	valuesClauseToString[values]
}, "\n"];
subSelectToString[Alternatives[
	SPARQLSelect[where_],
	RightComposition[
		SPARQLSelect[where_],
		orderby : Repeated[_SPARQLOrderBy, {0, 1}],
		project : Repeated[_SPARQLProject, {0, 1}],
		distinct : Repeated[_SPARQLDistinct, {0, 1}],
		limit : Repeated[_SPARQLLimit, {0, 1}]
	]
]] := StringJoin[{
	selectClauseToString[First[{project}, Identity], First[{distinct}, Identity]],
	" ",
	whereClauseToString[where],
	solutionModifierToString[orderby /* limit]
}];
subSelectToString[RightComposition[
	SPARQLSelect[where_],
	distinct : Repeated[_SPARQLDistinct, {0, 1}],
	agg_SPARQLAggregate,
	limit : Repeated[_SPARQLLimit, {0, 1}]
]] := StringJoin[{
	selectClauseToString[SPARQLProject[First[agg]], First[{distinct}, Identity]],
	" ",
	whereClauseToString[where],
	solutionModifierToString[agg /* limit]
}];
subSelectToString[RightComposition[
	Except[_SPARQLSelect, x_],
	outer : Alternatives[
		PatternSequence[
			Repeated[_SPARQLOrderBy, {0, 1}],
			Repeated[_SPARQLProject, {0, 1}],
			Repeated[_SPARQLDistinct, {0, 1}],
			Repeated[_SPARQLLimit, {0, 1}]
		],
		PatternSequence[
			Repeated[_SPARQLDistinct, {0, 1}],
			_SPARQLAggregate,
			Repeated[_SPARQLLimit, {0, 1}]
		]
	] /; {outer} =!= {}
]] := subSelectToString[SPARQLSelect[x] /* outer];

(* [9] *)
clear[selectClauseToString];
selectClauseToString[project : _SPARQLProject | Identity, distinct : _SPARQLDistinct | Identity] := StringRiffle[{
	"SELECT",
	distinct // Replace[{
		SPARQLDistinct[opts : OptionsPattern[]] :> Switch[OptionValue[SPARQLDistinct, {opts}, Method],
			"Distinct", "DISTINCT",
			"Reduced", "REDUCED",
			_, fail[]
		],
		Identity :> Nothing,
		_ :> fail[]
	}],
	project // Replace[{
		SPARQLProject[rules_] :> StringRiffle[
			Replace[
				Flatten[{rules}],
				{
					(Rule | RuleDelayed)[var_, expr_] :> "(" <> StringRiffle[{expressionToString[expr], "AS", varToString[SPARQLVariable[var]]}] <> ")",
					var_ :> varToString[SPARQLVariable[var]]
				},
				{1}
			]
		],
		Identity :> "*",
		_ :> fail[]
	}]
}];

(* [10] *)
clear[constructQueryToString];
constructQueryToString[SPARQLQuery[SPARQLConstruct[(Rule | RuleDelayed)[where_, template_]], qopts : OptionsPattern[]]] := StringJoin[{
	"CONSTRUCT ",
	groupGraphPatternToString[template],
	datasetClauseToString[OptionValue[SPARQLQuery, {qopts}, {"From", "FromNamed"}]],
	whereClauseToString[where]
}];
constructQueryToString[SPARQLQuery[SPARQLConstruct[where_], qopts : OptionsPattern[]]] := StringJoin[{
	"CONSTRUCT",
	datasetClauseToString[OptionValue[SPARQLQuery, {qopts}, {"From", "FromNamed"}]],
	whereClauseToString[where]
}];

(* [12] *)
clear[askQueryToString];
askQueryToString[SPARQLQuery[SPARQLAsk[where_], qopts : OptionsPattern[]]] := StringJoin[{
	"ASK",
	datasetClauseToString[OptionValue[SPARQLQuery, {qopts}, {"From", "FromNamed"}]],
	groupGraphPatternToString[where]
}];

(* [13] *)
clear[datasetClauseToString];
datasetClauseToString[{from_, fromNamed_}] := If[Automatic === from === fromNamed,
	" ",
	StringJoin[{
		If[from === Automatic, Nothing, "\nFROM " <> defaultGraphClauseToString[#] & /@ Flatten[{from}]],
		If[fromNamed === Automatic, Nothing, "\nFROM " <> namedGraphClauseToString[#] & /@ Flatten[{fromNamed}]],
		"\n"
	}]
];

(* [14] *)
clear[defaultGraphClauseToString];
defaultGraphClauseToString[x_] := sourceSelectorToString[x];

(* [15] *)
clear[namedGraphClauseToString];
namedGraphClauseToString[x_] := StringRiffle[{"NAMED", sourceSelectorToString[x]}];

(* [16] *)
clear[sourceSelectorToString];
sourceSelectorToString[x_] := iriToString[x];

(* [17] *)
clear[whereClauseToString];
whereClauseToString[x_] := StringRiffle[{"WHERE", groupGraphPatternToString[x]}];

(* [18] *)
clear[solutionModifierToString];
solutionModifierToString[Identity] := "";
solutionModifierToString[SPARQLAggregate[agg_]] := solutionModifierToString[SPARQLAggregate[agg, None]];
solutionModifierToString[SPARQLAggregate[agg_, groupby_]] := solutionModifierToString[SPARQLAggregate[agg, groupby, True]];
solutionModifierToString[SPARQLAggregate[agg_, groupby_, having_]] := solutionModifierToString[SPARQLAggregate[agg, groupby, having, None]];
solutionModifierToString[SPARQLAggregate[agg_, groupby_, having_, orderby_]] := StringJoin[{
	groupClauseToString[groupby],
	havingClauseToString[having],
	orderClauseToString[orderby]
}];
solutionModifierToString[x_SPARQLOrderBy] := orderClauseToString[x];
solutionModifierToString[x_SPARQLLimit] := limitOffsetClausesToString[x];
solutionModifierToString[c_RightComposition] := StringJoin[solutionModifierToString /@ List @@ c];

(* [19] *)
clear[groupClauseToString];
groupClauseToString[None] := "";
groupClauseToString[l : {__}] := "\nGROUP BY " <> StringRiffle[groupConditionToString /@ l];
groupClauseToString[x_] := groupClauseToString[{x}];

(* [20] *)
clear[groupConditionToString];
groupConditionToString[x : SPARQLEvaluation[_IRI]] := functionCallToString[x];
groupConditionToString[(Rule | RuleDelayed)[var_, expr_]] := StringJoin["(", expressionToString[expr], " AS ", varToString[SPARQLVariable[var]], ")"];
groupConditionToString[x_SPARQLVariable] := varToString[x];
groupConditionToString[x_] := builtInCallToString[x];

(* [21] *)
clear[havingClauseToString];
havingClauseToString[True] := "";
havingClauseToString[l : {__}] := "\nHAVING " <> StringRiffle[havingConditionToString /@ l];
havingClauseToString[x_] := havingClauseToString[{x}];

(* [22] *)
clear[havingConditionToString];
havingConditionToString[x_] := constraintToString[x];

(* [23] *)
clear[orderClauseToString];
orderClauseToString[SPARQLOrderBy[cond_]] := "\nORDER BY " <> StringRiffle[orderConditionToString /@ Flatten[{cond}]];
orderClauseToString[Identity] := "";
orderClauseToString[None] := "";
orderClauseToString[l : {__}] := "\nORDER BY " <> StringRiffle[orderConditionToString /@ l];
orderClauseToString[x_] := orderClauseToString[{x}];

(* [24] *)
clear[orderConditionToString];
orderConditionToString[(Rule | RuleDelayed)[expr_, "Ascending"]] := StringRiffle[{"ASC", brackettedExpressionToString[expr]}];
orderConditionToString[(Rule | RuleDelayed)[expr_, "Descending"]] := StringRiffle[{"DESC", brackettedExpressionToString[expr]}];
orderConditionToString[x_SPARQLVariable] := varToString[x];
orderConditionToString[x_] := constraintToString[x];

(* [25] *)
clear[limitOffsetClausesToString];
limitOffsetClausesToString[SPARQLLimit[l_, o_ : 0]] := limitClauseToString[l] <> offsetClauseToString[o];
limitOffsetClausesToString[Identity] := "";

(* [26] *)
clear[limitClauseToString];
limitClauseToString[Infinity] := "";
limitClauseToString[i_Integer?NonNegative] := "\nLIMIT " <> ToString[i];

(* [27] *)
clear[offsetClauseToString];
offsetClauseToString[0] := "";
offsetClauseToString[i_Integer?NonNegative] := "\nOFFSET " <> ToString[i];

(* [28] *)
clear[valuesClauseToString];
valuesClauseToString[SPARQLValues[args___]] := StringRiffle[{"VALUES", dataBlockToString[args]}];

(* [29] *)
clear[updateToString];
updateToString[x_, opts : OptionsPattern[]] /; {opts} =!= {} := Function[{prefixes, base},
	Module[
		{tmp},
		tmp = compact[x, prefixes, base];
		StringJoin[{
			prologueToString[If[prefixes === Automatic, AssociationMap[RDFPrefixData, collectPrefixes[tmp]], prefixes], base],
			updateToString[tmp]
		}]
	]
] @@ OptionValue[ExportSPARQLUpdate, {opts}, {"Prefixes", "Base"}];
updateToString[SPARQLUpdate[update_, opts : OptionsPattern[]]] := StringJoin[{
	prologueToString[<||>, OptionValue[SPARQLUpdate, {opts}, "Base"]],
	updateToString[update]
}];
updateToString[SPARQLUpdate[u_String]] := updateToString[Quiet[ImportString[u, "SPARQLUpdate"]]];
updateToString[Identity] := "";
updateToString[c_RightComposition] := StringRiffle[updateToString /@ List @@ c, ";\n"];
updateToString[x_] := update1ToString[x];

(* [30] *)
clear[update1ToString];
update1ToString[x_SPARQLLoad] := loadToString[x];
update1ToString[x_SPARQLClear] := clearToString[x];
update1ToString[x_SPARQLDrop] := dropToString[x];
update1ToString[x_SPARQLAdd] := addToString[x];
update1ToString[x_SPARQLMove] := moveToString[x];
update1ToString[x_SPARQLCopy] := copyToString[x];
update1ToString[x_SPARQLCreate] := createToString[x];
update1ToString[x_SPARQLInsertData] := insertDataToString[x];
update1ToString[x_SPARQLDeleteData] := deleteDataToString[x];
update1ToString[x : SPARQLDelete[Except[_Rule | _RuleDelayed]]] := deleteWhereToString[x];
update1ToString[x : _SPARQLDeleteInsert | _SPARQLDelete | _SPARQLInsert] := modifyToString[x];

(* [31] *)
clear[loadToString];
loadToString[SPARQLLoad[x_, opts : OptionsPattern[]]] := StringRiffle[{
	"LOAD",
	If[TrueQ[OptionValue[SPARQLLoad, {opts}, "Silent"]], "SILENT", Nothing],
	If[MatchQ[x, _Rule | _RuleDelayed],
		StringRiffle[{iriToString[First[x]], "INTO", graphRefToString[Last[x]]}],
		iriToString[x]
	]
}];

(* [32] *)
clear[clearToString];
clearToString[SPARQLClear[x_, opts : OptionsPattern[]]] := StringRiffle[{
	"CLEAR",
	If[TrueQ[OptionValue[SPARQLClear, {opts}, "Silent"]], "SILENT", Nothing],
	graphRefAllToString[x]
}];

(* [33] *)
clear[dropToString];
dropToString[SPARQLDrop[x_, opts : OptionsPattern[]]] := StringRiffle[{
	"DROP",
	If[TrueQ[OptionValue[SPARQLDrop, {opts}, "Silent"]], "SILENT", Nothing],
	graphRefAllToString[x]
}];

(* [34] *)
clear[createToString];
createToString[SPARQLCreate[x_, opts : OptionsPattern[]]] := StringRiffle[{
	"Create",
	If[TrueQ[OptionValue[SPARQLCreate, {opts}, "Silent"]], "SILENT", Nothing],
	graphRefToString[x]
}];

(* [35] *)
clear[addToString];
addToString[SPARQLAdd[from_, to_, opts : OptionsPattern[]]] := StringRiffle[{
	"ADD",
	If[TrueQ[OptionValue[SPARQLAdd, {opts}, "Silent"]], "SILENT", Nothing],
	graphOrDefaultToString[from],
	"TO",
	graphOrDefaultToString[to]
}];

(* [36] *)
clear[moveToString];
moveToString[SPARQLMove[from_, to_, opts : OptionsPattern[]]] := StringRiffle[{
	"MOVE",
	If[TrueQ[OptionValue[SPARQLMove, {opts}, "Silent"]], "SILENT", Nothing],
	graphOrDefaultToString[from],
	"TO",
	graphOrDefaultToString[to]
}];

(* [37] *)
clear[copyToString];
copyToString[SPARQLCopy[from_, to_, opts : OptionsPattern[]]] := StringRiffle[{
	"COPY",
	If[TrueQ[OptionValue[SPARQLCopy, {opts}, "Silent"]], "SILENT", Nothing],
	graphOrDefaultToString[from],
	"TO",
	graphOrDefaultToString[to]
}];

(* [38] *)
clear[insertDataToString];
insertDataToString[SPARQLInsertData[x_]] := StringRiffle[{"INSERT DATA", quadDataToString[x]}];

(* [39] *)
clear[deleteDataToString];
deleteDataToString[SPARQLDeleteData[x_]] := StringRiffle[{"DELETE DATA", quadDataToString[x]}];

(* [40] *)
clear[deleteWhereToString];
deleteWhereToString[SPARQLDelete[x_]] := StringRiffle[{"DELETE WHERE", quadPatternToString[x]}];

(* [41] *)
clear[modifyToString];
modifyToString[SPARQLDeleteInsert[del_, ins_, where_, opts : OptionsPattern[]]] := StringJoin[{
	With[{w = OptionValue[SPARQLDeleteInsert, {opts}, "With"]}, If[MatchQ[w, _IRI], StringRiffle[{"WITH", iriToString[w]}] <> "\n", Nothing]],
	deleteClauseToString[del],
	" ",
	insertClauseToString[ins],
	usingClauseToString[OptionValue[SPARQLDeleteInsert, {opts}, {"Using", "UsingNamed"}]],
	"WHERE ",
	groupGraphPatternToString[where]
}];
modifyToString[SPARQLDelete[(Rule | RuleDelayed)[where_, del_], opts : OptionsPattern[]]] := StringJoin[{
	With[{w = OptionValue[SPARQLDelete, {opts}, "With"]}, If[MatchQ[w, _IRI], StringRiffle[{"WITH", iriToString[w]}] <> "\n", Nothing]],
	deleteClauseToString[del],
	usingClauseToString[OptionValue[SPARQLDelete, {opts}, {"Using", "UsingNamed"}]],
	"WHERE ",
	groupGraphPatternToString[where]
}];
modifyToString[SPARQLInsert[(Rule | RuleDelayed)[where_, ins_], opts : OptionsPattern[]]] := StringJoin[{
	With[{w = OptionValue[SPARQLInsert, {opts}, "With"]}, If[MatchQ[w, _IRI], StringRiffle[{"WITH", iriToString[w]}] <> "\n", Nothing]],
	insertClauseToString[ins],
	usingClauseToString[OptionValue[SPARQLInsert, {opts}, {"Using", "UsingNamed"}]],
	"WHERE ",
	groupGraphPatternToString[where]
}];

(* [42] *)
clear[deleteClauseToString];
deleteClauseToString[x_] := StringRiffle[{"DELETE", quadPatternToString[x]}];

(* [43] *)
clear[insertClauseToString];
insertClauseToString[x_] := StringRiffle[{"INSERT", quadPatternToString[x]}];

(* [44] *)
clear[usingClauseToString];
usingClauseToString[{using_, usingNamed_}] := If[Automatic === using === usingNamed,
	" ",
	StringJoin[{
		If[using === Automatic, Nothing, "\nUSING " <> iriToString[#] & /@ Flatten[{using}]],
		If[usingNamed === Automatic, Nothing, "\nUSING NAMED " <> iriToString[#] & /@ Flatten[{usingNamed}]],
		"\n"
	}]
];

(* [45] *)
clear[graphOrDefaultToString];
graphOrDefaultToString["Default"] := "DEFAULT";
graphOrDefaultToString[x_] := StringRiffle[{"GRAPH", iriToString[x]}];

(* [46] *)
clear[graphRefToString];
graphRefToString[x_] := StringRiffle[{"GRAPH", iriToString[x]}];

(* [47] *)
clear[graphRefAllToString];
graphRefAllToString["Default"] := "DEFAULT";
graphRefAllToString["Named"] := "NAMED";
graphRefAllToString["All" | All] := "ALL";
graphRefAllToString[x_] := graphRefToString[x];

(* [48] *)
clear[quadPatternToString];
quadPatternToString[x_] := StringRiffle[{
	"{",
	indent[quadsToString[x]],
	"}"
}, "\n"];

(* [49] *)
clear[quadDataToString];
quadDataToString[x_] := StringRiffle[{
	"{",
	indent[quadsToString[x]],
	"}"
}, "\n"];

(* [50] *)
clear[quadsToString];
quadsToString[x_] := StringRiffle[Switch[#, _SPARQLGraph, quadsNotTriplesToString, _, triplesTemplateToString][#] & /@ Flatten[{x}], "\n"];

(* [51] *)
clear[quadsNotTriplesToString];
quadsNotTriplesToString[SPARQLGraph[varoriri_, template_]] := StringRiffle[{
	"GRAPH",
	varOrIriToString[varoriri],
	StringRiffle[{
		"{",
		indent[triplesTemplateToString[template]],
		"}"
	}, "\n"]
}];

(* [52] *)
clear[triplesTemplateToString];
triplesTemplateToString[x_] := stringRiffle[stringRiffle[{triplesSameSubjectToString[#], "."}] & /@ Flatten[{x}], "\n"];

(* [53] *)
clear[groupGraphPatternToString];
groupGraphPatternToString[sub : _SPARQLSelect | _RightComposition] := StringRiffle[{
	"{",
	indent[subSelectToString[sub]],
	"}"
}, "\n"];
groupGraphPatternToString[{}] := "{\n}";
groupGraphPatternToString[ggp_List] := StringRiffle[{
	"{",
	indent[groupGraphPatternSubToString[ggp]],
	"}"
}, "\n"];
groupGraphPatternToString[t_RDFTriple] := groupGraphPatternToString[{t}];
groupGraphPatternToString[Verbatim[Condition][l_List, cond_]] := groupGraphPatternToString[Append[l, filter[cond]]];
groupGraphPatternToString[Verbatim[Condition][patt_, cond_]] := groupGraphPatternToString[{patt, filter[cond]}];

(* [54] *)
clear[groupGraphPatternSubToString];
groupGraphPatternSubToString[l_List] := StringRiffle[If[MatchQ[#, _RDFTriple], triplesBlockToString, graphPatternNotTriplesToString][#] & /@ l, "\n"];

(* [55] *)
clear[triplesBlockToString];
triplesBlockToString[x_] := StringRiffle[{triplesSameSubjectPathToString[x], "."}];

(* [56] *)
clear[graphPatternNotTriplesToString];
graphPatternNotTriplesToString[x_SPARQLOptional] := optionalGraphPatternToString[x];
graphPatternNotTriplesToString[x_Except] := minusGraphPatternToString[x];
graphPatternNotTriplesToString[x_SPARQLGraph] := graphGraphPatternToString[x];
graphPatternNotTriplesToString[x_SPARQLService] := serviceGraphPatternToString[x];
graphPatternNotTriplesToString[x_filter] := filterToString[x];
graphPatternNotTriplesToString[x : _Rule | _RuleDelayed] := bindToString[x];
graphPatternNotTriplesToString[x_SPARQLValues] := inlineDataToString[x];
graphPatternNotTriplesToString[x_] := groupOrUnionGraphPatternToString[x];

(* [57] *)
clear[optionalGraphPatternToString];
optionalGraphPatternToString[SPARQLOptional[ggp_]] := StringRiffle[{"OPTIONAL", groupGraphPatternToString[ggp]}];

(* [58] *)
clear[graphGraphPatternToString];
graphGraphPatternToString[SPARQLGraph[varoriri_, ggp_]] := StringRiffle[{"GRAPH", varOrIriToString[varoriri], groupGraphPatternToString[ggp]}];

(* [59] *)
clear[serviceGraphPatternToString];
serviceGraphPatternToString[SPARQLService[varoriri_, ggp_, opts : OptionsPattern[]]] := StringRiffle[{
	"SERVICE",
	If[TrueQ[OptionValue[SPARQLService, {opts}, "Silent"]], "SILENT", Nothing],
	varOrIriToString[varoriri],
	groupGraphPatternToString[ggp]
}];

(* [60] *)
clear[bindToString];
bindToString[(Rule | RuleDelayed)[var_, expr_]] := "BIND(" <> StringRiffle[{expressionToString[expr], "AS", varToString[SPARQLVariable[var]]}] <> ")";

(* [61] *)
clear[inlineDataToString];
inlineDataToString[SPARQLValues[args___]] := StringRiffle[{"VALUES", dataBlockToString[args]}];

(* [62] *)
clear[dataBlockToString];
dataBlockToString[Except[_List, var_], data_, OptionsPattern[]] := inlineDataOneVarToString[var, data];
dataBlockToString[vars_List, data_, OptionsPattern[]] := inlineDataFullToString[vars, data];

(* [63] *)
clear[inlineDataOneVarToString];
inlineDataOneVarToString[var_, data_List] := StringRiffle[{varToString[SPARQLVariable[var]], "{", Sequence @@ dataBlockValueToString /@ data, "}"}];

(* [64] *)
clear[inlineDataFullToString];
inlineDataFullToString[{}, {}] := "() { }";
inlineDataFullToString[vars_List, data_List] /; MatchQ[Dimensions[data, 2], {_, Length[vars]}] := StringRiffle[{
	"(" <> StringRiffle[varToString[SPARQLVariable[#]] & /@ vars] <> ") {",
	indent[StringRiffle[
		"(" <> StringRiffle[dataBlockValueToString /@ #] <> ")" & /@ data,
		"\n"
	]],
	"}"
}, "\n"];

(* [65] *)
clear[dataBlockValueToString];
dataBlockValueToString[x_IRI] := iriToString[x];
dataBlockValueToString[x : RDFLiteral[_, xsd["integer"] | xsd["decimal"] | xsd["double"]] | _?NumberQ] := numericLiteralToString[x];
dataBlockValueToString[x : RDFLiteral[_, xsd["boolean"]] | _?BooleanQ] := booleanLiteralToString[x];
dataBlockValueToString[x : _RDFLiteral | _RDFString | _String] := rDFLiteralToString[x];
dataBlockValueToString[Undefined] := "UNDEF";
dataBlockValueToString[x_] := dataBlockValueToString[ToRDFLiteral[x] // Replace[_ToRDFLiteral :> fail[]]];

(* [66] *)
clear[minusGraphPatternToString];
minusGraphPatternToString[Verbatim[Except][ggp_]] := StringRiffle[{"MINUS", groupGraphPatternToString[ggp]}];

(* [67] *)
clear[groupOrUnionGraphPatternToString];
groupOrUnionGraphPatternToString[u_Alternatives] := StringRiffle[groupGraphPatternToString /@ List @@ u, " UNION "];
groupOrUnionGraphPatternToString[x_] := groupGraphPatternToString[x];

(* [68] *)
clear[filterToString];
filterToString[filter[x : SPARQLEvaluation[_String?(StringMatchQ["EXISTS", IgnoreCase -> True])][___]]] := StringRiffle[{"FILTER", existsFuncToString[x]}];
filterToString[filter[x : Not[SPARQLEvaluation[_String?(StringMatchQ["EXISTS", IgnoreCase -> True])][___]]]] := StringRiffle[{"FILTER", notExistsFuncToString[x]}];
filterToString[filter[cond_]] := StringRiffle[{"FILTER", constraintToString[cond]}];

(* [69] *)
clear[constraintToString];
constraintToString[expr_] := "(" <> expressionToString[expr] <> ")";

(* [70] *)
clear[functionCallToString];
functionCallToString[SPARQLEvaluation[i_][args___]] := iriToString[i] <> argListToString[{args}];

(* [71] *)
clear[argListToString];
argListToString[{}] := "()";
argListToString[{expr__, opts : OptionsPattern[]}] := StringJoin[{
	"(",
	If[TrueQ[OptionValue["Distinct" -> False, {opts}, "Distinct"]], "DISTINCT ", Nothing],
	StringRiffle[expressionToString /@ {expr}, ", "],
	")"
}];

(* [72] *)
clear[expressionListToString];
expressionListToString[{}] := "()";
expressionListToString[l_List] := "(" <> StringRiffle[expressionToString /@ l, ", "] <> ")";

(* [75] *)
clear[triplesSameSubjectToString];
triplesSameSubjectToString[RDFTriple[s_, p_, o_]] := stringRiffle[{varOrTermToString[s], verbToString[p], objectListToString[o]}];
triplesSameSubjectToString[RDFTriple[s_, pol : SPARQLPredicateObjectList[__]]] := StringRiffle[{varOrTermToString[s], propertyListNotEmptyToString[pol]}];

(* [77] *)
clear[propertyListNotEmptyToString];
propertyListNotEmptyToString[pol : SPARQLPredicateObjectList[{_, _} ..]] := StringRiffle[Function[{p, o}, StringRiffle[{verbToString[p], objectListToString[o]}]] @@@ List @@ pol, "; "];

(* [78] *)
clear[verbToString];
verbToString[IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]] := "a";
verbToString[x_] := varOrIriToString[x];

(* [79] *)
clear[objectListToString];
objectListToString[ol_SPARQLObjectList] := StringRiffle[objectToString /@ List @@ ol, ", "];
objectListToString[x_] := objectToString[x];

(* [80] *)
clear[objectToString];
objectToString[x_] := graphNodeToString[x];

(* [81] *)
clear[triplesSameSubjectPathToString];
triplesSameSubjectPathToString[RDFTriple[s_, p_, o_]] := triplesSameSubjectPathToString[RDFTriple[s, SPARQLPredicateObjectList[{p, o}]]];
triplesSameSubjectPathToString[RDFTriple[s_, pol : SPARQLPredicateObjectList[__]]] := StringRiffle[{varOrTermToString[s], propertyListPathNotEmptyToString[pol]}];

(* [83] *)
clear[propertyListPathNotEmptyToString];
propertyListPathNotEmptyToString[SPARQLPredicateObjectList[{p1_, o1_}, rest : {_, _} ...]] := StringRiffle[Join[
	{StringRiffle[{If[MatchQ[p1, _SPARQLVariable], verbSimpleToString, verbPathToString][p1], objectListPathToString[o1]}]},
	Function[{p, o},
		StringRiffle[{If[MatchQ[p, _SPARQLVariable], verbSimpleToString, verbPathToString][p], objectListToString[o]}]
	] @@@ {rest}
], "; "];

(* [84] *)
clear[verbPathToString];
verbPathToString[x_] := pathToString[x];

(* [85] *)
clear[verbSimpleToString];
verbSimpleToString[x_] := varToString[x];

(* [86] *)
clear[objectListPathToString];
objectListPathToString[ol_SPARQLObjectList] := StringRiffle[objectPathToString /@ List @@ ol, ", "];
objectListPathToString[x_] := objectPathToString[x];

(* [87] *)
clear[objectPathToString];
objectPathToString[x_] := graphNodePathToString[x];

(* [88] *)
clear[pathToString];
pathToString[x_] := pathAlternativeToString[x];

(* [89] *)
clear[pathAlternativeToString];
pathAlternativeToString[a_Alternatives] := StringRiffle[pathSequenceToString /@ List @@ a, " | "];
pathAlternativeToString[x_] := pathSequenceToString[x];

(* [90] *)
clear[pathSequenceToString];
pathSequenceToString[s_PatternSequence] := StringRiffle[pathEltOrInverseToString /@ List @@ s, " / "];
pathSequenceToString[x_] := pathEltOrInverseToString[x];

(* [91] *)
clear[pathEltToString];
pathEltToString[Verbatim[Repeated][x_, {0, 1}] | Verbatim[RepeatedNull][x_, {0, 1} | 1]] := pathPrimaryToString[x] <> "?";
pathEltToString[Verbatim[RepeatedNull][x_]] := pathPrimaryToString[x] <> "*";
pathEltToString[Verbatim[Repeated][x_]] := pathPrimaryToString[x] <> "+";
pathEltToString[x_] := pathPrimaryToString[x];

(* [92] *)
clear[pathEltOrInverseToString];
pathEltOrInverseToString[SPARQLInverseProperty[x_]] := "^" <> pathEltToString[x];
pathEltOrInverseToString[x_] := pathEltToString[x];

(* [94] *)
clear[pathPrimaryToString];
pathPrimaryToString[i_IRI] := iriToString[i];
pathPrimaryToString[Verbatim[Except][x_]] := "!" <> pathNegatedPropertySetToString[x];
pathPrimaryToString[x_] := "(" <> pathToString[x] <> ")";

(* [95] *)
clear[pathNegatedPropertySetToString];
pathNegatedPropertySetToString[a_Alternatives] := "(" <> StringRiffle[pathOneInPropertySetToString /@ List @@ a, " | "] <> ")";
pathNegatedPropertySetToString[x_] := pathOneInPropertySetToString[x];

(* [96] *)
clear[pathOneInPropertySetToString];
pathOneInPropertySetToString[SPARQLInverseProperty[x_]] := "^" <> iriToString[x];
pathOneInPropertySetToString[x_] := iriToString[x];

(* [98] *)
clear[triplesNodeToString];
triplesNodeToString[x_RDFCollection] := collectionToString[x];
triplesNodeToString[x : RDFBlankNode[_List | _SPARQLPredicateObjectList]] := blankNodePropertyListToString[x];

(* [99] *)
clear[blankNodePropertyListToString];
blankNodePropertyListToString[RDFBlankNode[pol_SPARQLPredicateObjectList]] := "[" <> propertyListNotEmptyToString[pol] <> "]";
blankNodePropertyListToString[RDFBlankNode[po_List]] := blankNodePropertyListToString[RDFBlankNode[SPARQLPredicateObjectList[po]]];

(* [100] *)
clear[triplesNodePathToString];
triplesNodePathToString[x_RDFCollection] := collectionPathToString[x];
triplesNodePathToString[x : RDFBlankNode[_List | _SPARQLPredicateObjectList]] := blankNodePropertyListPathToString[x];

(* [101] *)
clear[blankNodePropertyListPathToString];
blankNodePropertyListPathToString[RDFBlankNode[pol_SPARQLPredicateObjectList]] := "[" <> propertyListPathNotEmptyToString[pol] <> "]";
blankNodePropertyListPathToString[RDFBlankNode[po_List]] := blankNodePropertyListPathToString[RDFBlankNode[SPARQLPredicateObjectList[po]]];

(* [102] *)
clear[collectionToString];
collectionToString[RDFCollection[l_List]] := "(" <> StringRiffle[graphNodeToString /@ l] <> ")";

(* [103] *)
clear[collectionPathToString];
collectionPathToString[RDFCollection[l_List]] := "(" <> StringRiffle[graphNodePathToString /@ l] <> ")";

(* [104] *)
clear[graphNodeToString];
graphNodeToString[x : _RDFCollection | RDFBlankNode[_List | _SPARQLPredicateObjectList]] := triplesNodeToString[x];
graphNodeToString[x_] := varOrTermToString[x];

(* [105] *)
clear[graphNodePathToString];
graphNodePathToString[x : _RDFCollection | RDFBlankNode[_List | _SPARQLPredicateObjectList]] := triplesNodePathToString[x];
graphNodePathToString[x_] := varOrTermToString[x];

(* [106] *)
clear[varOrTermToString];
varOrTermToString[var_SPARQLVariable] := varToString[var];
varOrTermToString[x_] := graphTermToString[x];

(* [107] *)
clear[varOrIriToString];
varOrIriToString[x_SPARQLVariable] := varToString[x];
varOrIriToString[x_IRI] := iriToString[x];

(* [108] *)
clear[varToString];
varToString[SPARQLVariable[var_String]] := "?" <> var;

(* [109] *)
clear[graphTermToString];
graphTermToString[x_IRI] := iriToString[x];
graphTermToString[x : RDFLiteral[_, xsd["integer"] | xsd["decimal"] | xsd["double"]] | _?NumberQ] := numericLiteralToString[x];
graphTermToString[x : RDFLiteral[_, xsd["boolean"]] | _?BooleanQ] := booleanLiteralToString[x];
graphTermToString[x : _RDFLiteral | _RDFString | _String] := rDFLiteralToString[x];
graphTermToString[x_RDFBlankNode] := blankNodeToString[x];
graphTermToString[{}] := "()";
graphTermToString[x_] := graphTermToString[ToRDFLiteral[x] // Replace[_ToRDFLiteral :> fail[]]];

(* [110] *)
clear[expressionToString];
expressionToString[x_] := conditionalOrExpressionToString[x];

(* [111] *)
clear[conditionalOrExpressionToString];
conditionalOrExpressionToString[o_Or] := StringRiffle[conditionalAndExpressionToString /@ List @@ o, " || "];
conditionalOrExpressionToString[x_] := conditionalAndExpressionToString[x];

(* [112] *)
clear[conditionalAndExpressionToString];
conditionalAndExpressionToString[a_And] := StringRiffle[valueLogicalToString /@ List @@ a, " && "];
conditionalAndExpressionToString[x_] := valueLogicalToString[x];

(* [113] *)
clear[valueLogicalToString];
valueLogicalToString[x_] := relationalExpressionToString[x];

(* [114] *)
clear[relationalExpressionToString];
relationalExpressionToString[x : (Equal | Unequal | Less | Greater | LessEqual | GreaterEqual)[Repeated[_, {3, Infinity}]]] := conditionalAndExpressionToString[And @@ Partition[x, 2, 1]];
relationalExpressionToString[x_Equal] := StringRiffle[numericExpressionToString /@ List @@ x, " = "];
relationalExpressionToString[x_Unequal] := StringRiffle[numericExpressionToString /@ List @@ x, " != "];
relationalExpressionToString[x_Less] := StringRiffle[numericExpressionToString /@ List @@ x, " < "];
relationalExpressionToString[x_Greater] := StringRiffle[numericExpressionToString /@ List @@ x, " > "];
relationalExpressionToString[x_LessEqual] := StringRiffle[numericExpressionToString /@ List @@ x, " <= "];
relationalExpressionToString[x_GreaterEqual] := StringRiffle[numericExpressionToString /@ List @@ x, " >= "];
relationalExpressionToString[SPARQLEvaluation[f_String?(StringMatchQ["IN", IgnoreCase -> True])][expr_, l_List]] := StringRiffle[{expressionToString[expr], f, expressionListToString[l]}];
relationalExpressionToString[Not[SPARQLEvaluation[f_String?(StringMatchQ["IN", IgnoreCase -> True])][expr_, l_List]]] := StringRiffle[{expressionToString[expr], "NOT", f, expressionListToString[l]}];
relationalExpressionToString[x_] := numericExpressionToString[x];

(* [115] *)
clear[numericExpressionToString];
numericExpressionToString[x_] := additiveExpressionToString[x];

(* [116] *)
clear[additiveExpressionToString];
additiveExpressionToString[x_Plus] := StringRiffle[multiplicativeExpressionToString /@ List @@ x, " + "];
additiveExpressionToString[x_ - y_] := StringRiffle[multiplicativeExpressionToString /@ {x, y}, " - "];
additiveExpressionToString[x_] := multiplicativeExpressionToString[x];

(* [117] *)
clear[multiplicativeExpressionToString];
multiplicativeExpressionToString[Times[Rational[1, y_], x_]] := unaryExpressionToString[x] <> " / " <> unaryExpressionToString[y];
multiplicativeExpressionToString[x_Times] := StringRiffle[unaryExpressionToString /@ List @@ x, " * "];
multiplicativeExpressionToString[x : _Divide | Inactive[Divide][___]] := StringRiffle[unaryExpressionToString /@ List @@ x, " / "];
multiplicativeExpressionToString[x_ / y_] := StringRiffle[unaryExpressionToString /@ {x, y}, " / "];
multiplicativeExpressionToString[x_] := unaryExpressionToString[x];

(* [118] *)
clear[unaryExpressionToString];
unaryExpressionToString[Not[x_]] := stringRiffle[{"!", primaryExpressionToString[x]}];
unaryExpressionToString[x_] := primaryExpressionToString[x];

(* [119] *)
clear[primaryExpressionToString];
primaryExpressionToString[f : SPARQLEvaluation[_String | _SPARQLDistinct /* _String][___]] := builtInCallToString[f];
primaryExpressionToString[f : SPARQLEvaluation[_IRI][___]] := iriOrFunctionToString[f];
primaryExpressionToString[i_IRI] := iriOrFunctionToString[i];
primaryExpressionToString[x : RDFLiteral[_, xsd["integer"] | xsd["decimal"] | xsd["double"]] | _?NumberQ] := numericLiteralToString[x];
primaryExpressionToString[x : RDFLiteral[_, xsd["boolean"]] | _?BooleanQ] := booleanLiteralToString[x];
primaryExpressionToString[x : _RDFLiteral | _RDFString | _String] := rDFLiteralToString[x];
primaryExpressionToString[var_SPARQLVariable] := varToString[var];
primaryExpressionToString[x_] := With[
	{l = ToRDFLiteral[x]},
	primaryExpressionToString[l] /; ! MatchQ[l, _ToRDFLiteral]
];
primaryExpressionToString[x_] := If[x =!= $oldX,
	Block[
		{$oldX = x},
		brackettedExpressionToString[x]
	],
	fail[]
];

(* [120] *)
clear[brackettedExpressionToString];
brackettedExpressionToString[x_] := "(" <> expressionToString[x] <> ")";

(* [121] *)
clear[builtInCallToString];
builtInCallToString[f : _?SPARQLAggregateFunctionQ[___]] := aggregateToString[f];
builtInCallToString[x : SPARQLEvaluation[_String?(StringMatchQ["EXISTS", IgnoreCase -> True])][___]] := existsFuncToString[x];
builtInCallToString[x : Not[SPARQLEvaluation[_String?(StringMatchQ["EXISTS", IgnoreCase -> True])][___]]] := notExistsFuncToString[x];
builtInCallToString[SPARQLEvaluation[f_String][expr___]] := f <> "(" <> StringRiffle[expressionToString /@ {expr}, ", "] <> ")";

(* [125] *)
clear[existsFuncToString];
existsFuncToString[_[ggp_]] := StringRiffle[{"EXISTS", groupGraphPatternToString[ggp]}];

(* [126] *)
clear[notExistsFuncToString];
notExistsFuncToString[Not[_[ggp_]]] := StringRiffle[{"NOT EXISTS", groupGraphPatternToString[ggp]}];

(* [127] *)
clear[aggregateToString];
aggregateToString[SPARQLEvaluation[f_String | (d : SPARQLDistinct[OptionsPattern[]]) /* f_][expr : Except[OptionsPattern[]] : All, opts : OptionsPattern[]]] := StringJoin[{
	f,
	"(",
	If[{d} === {}, Nothing, "DISTINCT" <> " "],
	If[expr === All,
		"*",
		{
			expressionToString[expr],
			With[
				{sep = OptionValue["Separator" -> " ", {opts}, "Separator"]},
				If[sep === " ", Nothing, "; SEPARATOR = " <> stringToString[sep]]
			]
		}
	],
	")"
}];

(* [128] *)
clear[iriOrFunctionToString];
iriOrFunctionToString[SPARQLEvaluation[i_][args___]] := iriToString[i] <> argListToString[{args}];
iriOrFunctionToString[i_] := iriToString[i];

(* [129] *)
clear[rDFLiteralToString];
rDFLiteralToString[s_String | RDFLiteral[s_String, xsd["string"]]] := stringToString[s];
rDFLiteralToString[RDFString[s_, lang_String]] := stringToString[s] <> "@" <> lang;
rDFLiteralToString[RDFLiteral[s_, dt_]] := stringToString[s] <> "^^" <> iriToString[Replace[dt, i_String :> IRI[i]]];

(* [130] *)
clear[numericLiteralToString];
numericLiteralToString[RDFLiteral[s_String, xsd["integer"] | xsd["decimal"] | xsd["double"]]] := s;
numericLiteralToString[i_Integer] := ToString[i];
numericLiteralToString[r_Real?MachineNumberQ] := With[{s = ToString[r]}, If[StringEndsQ["."][s], s <> "0", s]];
numericLiteralToString[Rational[x_, y_]] := ToString[x] <> " / " <> ToString[y];

(* [134] *)
clear[booleanLiteralToString];
booleanLiteralToString[RDFLiteral[s : "false" | "true", xsd["boolean"]]] := s;
booleanLiteralToString[False] := "false";
booleanLiteralToString[True] := "true";

(* [135] *)
clear[stringToString];
stringToString[s_String] := Which[
	StringFreeQ[s, "\""],
	"\"" <> s <> "\"",
	StringFreeQ[s, "'"],
	"'" <> s <> "'",
	True,
	"\"" <> StringReplace[s, "\"" -> "\\\""] <> "\""
];

(* [136] *)
clear[iriToString];
iriToString[IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]] := "a";
iriToString[IRI[i_String]] := "<" <> i <> ">";
iriToString[IRI[{prefix_String, local_String}]] := prefix <> ":" <> local;

(* [138] *)
clear[blankNodeToString];
blankNodeToString[RDFBlankNode[label_String]] := "_:" <> label;
blankNodeToString[RDFBlankNode[]] := "[]";

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportSPARQLQuery];
Options[iImportSPARQLQuery] = Options[ImportSPARQLQuery];
iImportSPARQLQuery[file_, OptionsPattern[]] := {"Data" -> (GrammarApply1[
	SPARQLGrammar[GrammarToken["QueryUnit"]],
	File[file],
	"WhitespacePattern" -> Whitespace | ("#" ~~ Except["\n"] ...)
] // Replace[_GrammarApply1 | $Failed :> fail[]] //
Replace[Except[SPARQLQuery[___, "Base" -> _, ___], q_] :> RDFExpandIRIs[q, None, ChooseBase[OptionValue["Base"], file]]])};

clear[iImportSPARQLQueryPrefixes];
iImportSPARQLQueryPrefixes[file_, OptionsPattern[]] := {"Prefixes" -> (GrammarApply1[
	SPARQLGrammar[FixedOrder[p : GrammarToken["Prologue"], ___] :> Lookup[{p}, "Prefixes", <||>]],
	File[file],
	"WhitespacePattern" -> Whitespace | ("#" ~~ Except["\n"] ...)
] // Replace[_GrammarApply1 | $Failed :> fail[]])};

clear[iImportSPARQLUpdate];
Options[iImportSPARQLUpdate] = Options[ImportSPARQLUpdate];
iImportSPARQLUpdate[file_, OptionsPattern[]] := {"Data" -> (GrammarApply1[
	SPARQLGrammar[GrammarToken["UpdateUnit"]],
	File[file],
	"WhitespacePattern" -> Whitespace | ("#" ~~ Except["\n"] ...)
] // Replace[_GrammarApply1 | $Failed :> fail[]] //
Replace[Except[SPARQLUpdate[___, "Base" -> _, ___], q_] :> RDFExpandIRIs[q, None, ChooseBase[OptionValue["Base"], file]]])};

clear[iImportSPARQLUpdatePrefixes];
iImportSPARQLUpdatePrefixes[file_, OptionsPattern[]] := {"Prefixes" -> (GrammarApply1[
	SPARQLGrammar[FixedOrder[p : GrammarToken["Prologue"], ___] :> Lookup[{p}, "Prefixes", <||>]],
	File[file],
	"WhitespacePattern" -> Whitespace | ("#" ~~ Except["\n"] ...)
] // Replace[_GrammarApply1 | $Failed :> fail[]])};


(* query syntax restrictions *)
satisfiesQuerySyntaxRestrictionsQ[x_] := And[
	satisfiesAggregateProjectionRestrictionsQ[x],
	satisfiesAssignmentRestrictionsQ[x],
	satisfiesBindRestrictionsQ[x]
];

(* aggregate projection restrictions *)
satisfiesAggregateProjectionRestrictionsQ[x_] := AllTrue[
	Cases[x, _SPARQLAggregate, Infinity],
	allVariablesAggregatedQ
];

allVariablesAggregatedQ[SPARQLAggregate[All, ___]] := False;
allVariablesAggregatedQ[SPARQLAggregate[agg_, rest___]] := AllTrue[
	Flatten[{agg}],
	Replace[{
		(Rule | RuleDelayed)[_, expr_] :> ! FreeQ[expr, _?SPARQLAggregateFunctionQ[___]],
		var_String :> Length[{rest}] >= 1 && MemberQ[Flatten[{First[{rest}]}], SPARQLVariable[var] | (Rule | RuleDelayed)[var, _]],
		_ :> True
	}]
];
allVariablesAggregatedQ[_] := False;

(* assignment restrictions *)
satisfiesAssignmentRestrictionsQ[SPARQLProject[se_]] := DuplicateFreeQ[Replace[Flatten[{se}], (Rule | RuleDelayed)[var_, _] :> var, {1}]];
satisfiesAssignmentRestrictionsQ[RightComposition[inner___, proj : SPARQLProject[se_]]] := And[
	satisfiesAssignmentRestrictionsQ[proj],
	FreeQ[{inner}, Alternatives @@ Cases[Flatten[{se}], (Rule | RuleDelayed)[var_, _] :> var]]
];
satisfiesAssignmentRestrictionsQ[RightComposition[inner___, Except[_SPARQLProject]]] := satisfiesAssignmentRestrictionsQ[RightComposition[inner]];
satisfiesAssignmentRestrictionsQ[x_] := AllTrue[
	Cases[x, _SPARQLProject | _RightComposition, Infinity],
	satisfiesAssignmentRestrictionsQ
];

(* bind restrictions *)
satisfiesBindRestrictionsQ[x_] := FreeQ[
	x,
	{before__, (Rule | RuleDelayed)[var_, _], ___} /; ! FreeQ[{before}, SPARQLVariable[var] | (Rule | RuleDelayed)[var, _]]
];


(* update syntax restrictions *)
satisfiesUpdateSyntaxRestrictionsQ[x_] := And[
	(* https://www.w3.org/TR/sparql11-query/#sparqlGrammar *)
	x // FreeQ[Alternatives[
		(* note 8 *)
		(SPARQLInsertData | SPARQLDeleteData)[_?(Not @* FreeQ[_RDFTriple?(MemberQ[_SPARQLVariable]) | SPARQLGraph[_SPARQLVariable, ___]]), ___],
		(* note 9 *)
		SPARQLDelete[Except[_Rule | _RuleDelayed]?(Not @* FreeQ[RDFBlankNode]) | (Rule | RuleDelayed)[_, _?(Not @* FreeQ[RDFBlankNode])], ___],
		SPARQLDeleteInsert[_?(Not @* FreeQ[RDFBlankNode]), ___],
		SPARQLDeleteData[_?(Not @* FreeQ[RDFBlankNode]), ___]
	]],
	(* https://www.w3.org/TR/sparql11-query/#grammarBNodes *)
	Cases[x, SPARQLInsertData[y_, ___] :> Cases[y, RDFBlankNode[l_] :> l, Infinity], Infinity] // Replace[{
		{} | {_} :> True,
		l_ :> Intersection @@ l === {}
	}]
];


clear[activate];
activate[expr_] := expr /. Except[Inactive[Divide][_, 0 | 0.], i : Inactive[_][___]] :> Activate[i];

clear[toPattern];
toPattern[patt_] := patt //. {
	gp_List?(MemberQ[filter[_]]) :> condition[DeleteCases[gp, filter[_]], And @@ Cases[gp, filter[cond_] :> cond]],
	RDFTriple[s_, Verbatim[PatternSequence][pathseq___], o_] :> SPARQLPropertyPath[s, {pathseq}, o],
	RDFTriple[s_, pathexpr_?PropertyPathExpressionQ, o_] :> SPARQLPropertyPath[s, {pathexpr}, o]
} /. condition -> Condition;

clear[simplifySyntax];
simplifySyntax[expr_] := expr //. {
	SPARQLAggregate[agg_, groupby_, having_, None] :> SPARQLAggregate[agg, groupby, having],
	SPARQLAggregate[agg_, groupby_, True] :> SPARQLAggregate[agg, groupby],
	SPARQLAggregate[agg_, None] :> SPARQLAggregate[agg],
	SPARQLProject[All] :> Identity,
	SPARQLOrderBy[None] :> Identity
};


clear[solutionModifiersToOperator];
solutionModifiersToOperator[{project_, distinct_}, where_, {group_, having_, order_, limit_}] := If[
	Or[
		! FreeQ[project, _?SPARQLAggregateFunctionQ[___]],
		group =!= None,
		having =!= True,
		! FreeQ[order, _?SPARQLAggregateFunctionQ[___]]
	],
	SPARQLSelect[where] /*
	distinct /*
	SPARQLAggregate[
		project,
		group,
		having,
		order
	] /*
	limit,
	With[{b = Cases[project, _Rule | RuleDelayed, {0, 1}]},
		If[b === {} || order === None,
			SPARQLSelect[where] /*
			SPARQLOrderBy[order] /*
			SPARQLProject[project] /*
			distinct /*
			limit,
			SPARQLSelect[{where, Sequence @@ b}] /*
			SPARQLOrderBy[order] /*
			SPARQLProject[Replace[project, (Rule | RuleDelayed)[var_, _] :> var, {0, 1}]] /*
			distinct /*
			limit
		]
	]
];


SPARQLGrammar[token : GrammarToken["QueryUnit" | "UpdateUnit"]] := SPARQLGrammar[
	x : token :> (
		x //
		activate //
		ExpandTriples //
		toPattern //
		If[Switch[token,
			GrammarToken["QueryUnit"], satisfiesQuerySyntaxRestrictionsQ,
			GrammarToken["UpdateUnit"], satisfiesUpdateSyntaxRestrictionsQ
		][#], #, fail[]] & //
		simplifySyntax //
		If[TrueQ[$UseSPARQLSolutionModiferOperatorSyntax],
			Identity,
			SPARQLFromSolutionModifierOperatorSyntax
		] //
		Replace[c_RightComposition :> Switch[token,
			GrammarToken["QueryUnit"], SPARQLQuery,
			GrammarToken["UpdateUnit"], SPARQLUpdate
		][c]]
	)
];
SPARQLGrammar[entry_] := GrammarRules[{
	entry
}, {
	(* [1] *)
	"QueryUnit" -> GrammarToken["Query"],
	(* [2] *)
	"Query" -> FixedOrder[
		p : GrammarToken["Prologue"],
		q : GrammarToken["SelectQuery"] | GrammarToken["ConstructQuery"](* | GrammarToken["DescribeQuery"]*) | GrammarToken["AskQuery"],
		v : GrammarToken["ValuesClause"]
	] :> Replace[
		SPARQLQuery[
			RDFExpandIRIs[
				q /* v // Replace[SPARQLQuery[x_, qopts___] /* y_ :> SPARQLQuery[x /* y, qopts]],
				Lookup[{p}, "Prefixes", <||>],
				Lookup[{p}, "Base", None]
			],
			Sequence @@ FilterRules[{p}, "Base"]
		] /. SPARQLQuery[SPARQLQuery[x_, opts1___], opts2___] :> SPARQLQuery[x, opts1, opts2],
		SPARQLQuery[x_] :> x
	],
	(* [3] *)
	"UpdateUnit" -> GrammarToken["Update"],
	(* [4] *)
	"Prologue" -> decl : (GrammarToken["BaseDecl"] | GrammarToken["PrefixDecl"]) ... :> Sequence @@ Normal[GroupBy[{decl}, First -> Last, Apply[Join]]],
	(* [5] *)
	"BaseDecl" -> FixedOrder["BASE", i : GrammarToken["IRIREF"]] :> "Base" -> i,
	(* [6] *)
	"PrefixDecl" -> FixedOrder["PREFIX", prefix : PNAMENS, i : GrammarToken["IRIREF"]] :> "Prefixes" -> <|StringDrop[prefix, -1] -> i|>,
	(* [7] *)
	"SelectQuery" -> FixedOrder[
		sc : GrammarToken["SelectClause"],
		ds : GrammarToken["DatasetClause1"],
		where : GrammarToken["WhereClause"],
		sm : GrammarToken["SolutionModifier"]
	] :> SPARQLQuery[
		solutionModifiersToOperator[sc, where, sm],
		ds
	],
	(* [8] *)
	"SubSelect" -> FixedOrder[
		sc : GrammarToken["SelectClause"],
		where : GrammarToken["WhereClause"],
		sm : GrammarToken["SolutionModifier"],
		v : GrammarToken["ValuesClause"]
	] :> solutionModifiersToOperator[sc, where, sm] /* v,
	(* [9] *)
	"SelectClause" -> FixedOrder[
		"SELECT",
		d : Repeated[GrammarToken["DISTINCT"] | GrammarToken["REDUCED"], {0, 1}],
		var : (GrammarToken["Var"] | FixedOrder["(", GrammarToken["Expression"], "AS", GrammarToken["Var"], ")"]) .. | Verbatim["*"]
	] :> {
		(Replace[Replace[FixedPoint[Replace[{x___, "(", expr_, _String, v_, ")", y___} :> {x, v -> expr, y}], {var}], {SPARQLVariable[v_] :> v, (SPARQLVariable[v_] -> e_) :> v -> e}, {1}], {x_} :> x] // Replace[
			"*" :> All
		]),
		If[{d} === {}, Identity, d]
	},
	(* [10] *)
	"ConstructQuery" -> {
		FixedOrder[
			"CONSTRUCT",
			t : GrammarToken["ConstructTemplate"],
			ds : GrammarToken["DatasetClause1"],
			w : GrammarToken["WhereClause"],
			sm : GrammarToken["SolutionModifier"]
		] :> SPARQLQuery[
			SPARQLConstruct[
				If[sm[[3]] === None && sm[[4]] === Identity, w, SPARQLSelect[w] /* SPARQLOrderBy[sm[[3]]] /* sm[[4]]] -> t
			],
			ds
		],
		FixedOrder[
			"CONSTRUCT",
			ds : GrammarToken["DatasetClause1"],
			"WHERE",
			"{",
			t : Repeated[GrammarToken["TriplesTemplate"], {0, 1}],
			"}",
			sm : GrammarToken["SolutionModifier"]
		] :> SPARQLQuery[
			SPARQLConstruct[
				If[sm[[3]] === None && sm[[4]] === Identity, {t}, SPARQLSelect[{t}] /* SPARQLOrderBy[sm[[3]]] /* sm[[4]]]
			],
			ds
		]
	},
	(* [12] *)
	"AskQuery" -> FixedOrder[
		"ASK",
		ds : GrammarToken["DatasetClause1"],
		where : GrammarToken["WhereClause"],
		sm : GrammarToken["SolutionModifier"]
	] :> SPARQLQuery[SPARQLAsk[where], ds],
	(* [13] *)
	"DatasetClause" -> FixedOrder["FROM", g : GrammarToken["DefaultGraphClause"] | GrammarToken["NamedGraphClause"]] :> g,
	"DatasetClause1" -> ds : GrammarToken["DatasetClause"] ... :> Sequence @@ {
		Cases[{ds}, {"default", s_} :> s] // Replace[{
			{} :> Nothing,
			l_ :> "From" -> Replace[l, {x_} :> x]
		}],
		Cases[{ds}, {"named", s_} :> s] // Replace[{
			{} :> Nothing,
			l_ :> "FromNamed" -> Replace[l, {x_} :> x]
		}]
	},
	(* [14] *)
	"DefaultGraphClause" -> s : GrammarToken["SourceSelector"] :> {"default", s},
	(* [15] *)
	"NamedGraphClause" -> FixedOrder["NAMED", s : GrammarToken["SourceSelector"]] :> {"named", s},
	(* [16] *)
	"SourceSelector" -> GrammarToken["iri"],
	(* [17] *)
	"WhereClause" -> FixedOrder[Repeated["WHERE", {0, 1}], ggp : GrammarToken["GroupGraphPattern"]] :> ggp,
	(* [18] *)
	"SolutionModifier" -> FixedOrder[
		g : Repeated[GrammarToken["GroupClause"], {0, 1}],
		h : Repeated[GrammarToken["HavingClause"], {0, 1}],
		o : Repeated[GrammarToken["OrderClause"], {0, 1}],
		l : Repeated[GrammarToken["LimitOffsetClauses"], {0, 1}]
	] :> {
		If[{g} === {}, None, g],
		If[{h} === {}, True, h],
		If[{o} === {}, None, o],
		If[{l} === {}, Identity, l]
	},
	(* [19] *)
	"GroupClause" -> FixedOrder["GROUP", "BY", cond : GrammarToken["GroupCondition"] ..] :> Replace[{cond}, {x_} :> x],
	(* [20] *)
	"GroupCondition" -> {
		GrammarToken["BuiltInCall"],
		GrammarToken["FunctionCall"],
		FixedOrder["(", e : GrammarToken["Expression"], as : Repeated[FixedOrder["AS", GrammarToken["Var"]], {0, 1}], ")"] :> Replace[{e, as}, {{x_} :> x, {x_, _, v_} :> First[v] -> x}],
		GrammarToken["Var"]
	},
	(* [21] *)
	"HavingClause" -> FixedOrder["HAVING", cond : GrammarToken["HavingCondition"] ..] :> Replace[{cond}, {x_} :> x],
	(* [22] *)
	"HavingCondition" -> GrammarToken["Constraint"],
	(* [23] *)
	"OrderClause" -> FixedOrder["ORDER", "BY", cond : GrammarToken["OrderCondition"] ..] :> Replace[{cond}, {x_} :> x],
	(* [24] *)
	"OrderCondition" -> {FixedOrder[ad : GrammarToken["ASC"] | GrammarToken["DESC"], e : GrammarToken["BrackettedExpression"]] :> e -> ad, GrammarToken["Constraint"], GrammarToken["Var"]},
	(* [25] *)
	"LimitOffsetClauses" -> {
		FixedOrder[l : GrammarToken["LimitClause"], o : Repeated[GrammarToken["OffsetClause"], {0, 1}]] :> SPARQLLimit[l, o],
		FixedOrder[o : GrammarToken["OffsetClause"], l : Repeated[GrammarToken["LimitClause"], {0, 1}]] :> SPARQLLimit[If[{l} === {}, Infinity, l], o]
	},
	(* [26] *)
	"LimitClause" -> FixedOrder["LIMIT", i : INTEGER] :> Interpreter["Integer"][i],
	(* [27] *)
	"OffsetClause" -> FixedOrder["OFFSET", i : INTEGER] :> Interpreter["Integer"][i],
	(* [28] *)
	"ValuesClause" -> v : Repeated[FixedOrder["VALUES", GrammarToken["DataBlock"]], {0, 1}] :> Replace[{v}, {{_, db_} :> SPARQLValues @@ db, _ :> Sequence[]}],
	(* [29] *)
	"Update" -> FixedOrder[
		p : GrammarToken["Prologue"],
		u : Repeated[FixedOrder[GrammarToken["Update1"], Repeated[FixedOrder[";", GrammarToken["Update"]], {0, 1}]], {0, 1}]
	] :> Replace[
		SPARQLUpdate[
			RDFExpandIRIs[
				Replace[{u}, {
					{} :> Identity,
					{u1_} :> u1,
					{u1_, ";", u2_} :> RightComposition[u1, u2]
				}],
				Lookup[{p}, "Prefixes", <||>],
				Lookup[{p}, "Base", None]
			],
			Sequence @@ FilterRules[{p}, "Base"]
		],
		SPARQLUpdate[x_] :> x
	],
	(* [30] *)
	"Update1" -> {
		GrammarToken["Load"],
		GrammarToken["Clear"],
		GrammarToken["Drop"],
		GrammarToken["Add"],
		GrammarToken["Move"],
		GrammarToken["Copy"],
		GrammarToken["Create"],
		GrammarToken["InsertData"],
		GrammarToken["DeleteData"],
		GrammarToken["DeleteWhere"],
		GrammarToken["Modify"]
	},
	(* [31] *)
	"Load" -> FixedOrder["LOAD", s : GrammarToken["SILENT"], i : GrammarToken["iri"], j : Repeated[FixedOrder["INTO", GrammarToken["GraphRef"]], {0, 1}]] :> SPARQLLoad[
		{i, j} // Replace[{
			{g_} :> g,
			{g_, ___, h_} :> g -> h
		}],
		s
	],
	(* [32] *)
	"Clear" -> FixedOrder["CLEAR", s : GrammarToken["SILENT"], g : GrammarToken["GraphRefAll"]] :> SPARQLClear[g, s],
	(* [33] *)
	"Drop" -> FixedOrder["DROP", s : GrammarToken["SILENT"], g : GrammarToken["GraphRefAll"]] :> SPARQLDrop[g, s],
	(* [34] *)
	"Create" -> FixedOrder["CREATE", s : GrammarToken["SILENT"], g : GrammarToken["GraphRef"]] :> SPARQLCreate[g, s],
	(* [35] *)
	"Add" -> FixedOrder["ADD", s : GrammarToken["SILENT"], from : GrammarToken["GraphOrDefault"], "TO", to : GrammarToken["GraphOrDefault"]] :> SPARQLAdd[from, to, s],
	(* [36] *)
	"Move" -> FixedOrder["MOVE", s : GrammarToken["SILENT"], from : GrammarToken["GraphOrDefault"], "TO", to : GrammarToken["GraphOrDefault"]] :> SPARQLMove[from, to, s],
	(* [37] *)
	"Copy" -> FixedOrder["COPY", s : GrammarToken["SILENT"], from : GrammarToken["GraphOrDefault"], "TO", to : GrammarToken["GraphOrDefault"]] :> SPARQLCopy[from, to, s],
	(* [38] *)
	"InsertData" -> FixedOrder["INSERT", "DATA", qd : GrammarToken["QuadData"]] :> SPARQLInsertData[qd],
	(* [39] *)
	"DeleteData" -> FixedOrder["DELETE", "DATA", qd : GrammarToken["QuadData"]] :> SPARQLDeleteData[qd],
	(* [40] *)
	"DeleteWhere" -> FixedOrder["DELETE", "WHERE", q : GrammarToken["QuadPattern"]] :> SPARQLDelete[q],
	(* [41] *)
	"Modify" -> FixedOrder[
		with : GrammarToken["With"],
		clauses : FixedOrder[GrammarToken["DeleteClause"], Repeated[GrammarToken["InsertClause"], {0, 1}]] | GrammarToken["InsertClause"],
		using : GrammarToken["UsingClause1"],
		"WHERE",
		where : GrammarToken["GroupGraphPattern"]
	] :> Replace[
		RightComposition @@ Through[{clauses}[where, with, using]],
		RightComposition[SPARQLDelete[w_ -> del_, opts___], SPARQLInsert[w_ -> ins_, opts___]] :> SPARQLDeleteInsert[del, ins, w, opts]
	],
	(* [42] *)
	"DeleteClause" -> FixedOrder["DELETE", q : GrammarToken["QuadPattern"]] :> Function[SPARQLDelete[# -> q, ##2]],
	(* [43] *)
	"InsertClause" -> FixedOrder["INSERT", q : GrammarToken["QuadPattern"]] :> Function[SPARQLInsert[# -> q, ##2]],
	(* [44] *)
	"UsingClause" -> FixedOrder["USING", x : GrammarToken["iri"] | FixedOrder["NAMED", GrammarToken["iri"]]] :> Replace[{x}, {
		{i_} :> {"Using", i},
		{_, i_} :> {"UsingNamed", i}
	}],
	"UsingClause1" -> x : GrammarToken["UsingClause"] ... :> Sequence @@ Normal[GroupBy[{x}, First -> Last]],
	(* [45] *)
	"GraphOrDefault" -> {
		"DEFAULT" -> "Default",
		FixedOrder[Repeated["GRAPH", {0, 1}], g : GrammarToken["iri"]] :> g
	},
	(* [46] *)
	"GraphRef" -> FixedOrder["GRAPH", g : GrammarToken["iri"]] :> g,
	(* [47] *)
	"GraphRefAll" -> {
		GrammarToken["GraphRef"],
		"DEFAULT" -> "Default",
		"NAMED" -> "Named",
		"ALL" -> "All"
	},
	(* [48] *)
	"QuadPattern" -> FixedOrder["{", q : GrammarToken["Quads"], "}"] :> {q},
	(* [49] *)
	"QuadData" -> FixedOrder["{", q : GrammarToken["Quads"], "}"] :> {q},
	(* [50] *)
	"Quads" -> x : FixedOrder[
		Repeated[GrammarToken["TriplesTemplate"], {0, 1}],
		FixedOrder[GrammarToken["QuadsNotTriples"], Repeated[".", {0, 1}], Repeated[GrammarToken["TriplesTemplate"], {0, 1}]] ...
	] :> Sequence @@ DeleteCases[{x}, "."],
	(* [51] *)
	"QuadsNotTriples" -> FixedOrder["GRAPH", g : GrammarToken["VarOrIri"], "{", t : Repeated[GrammarToken["TriplesTemplate"], {0, 1}], "}"] :> SPARQLGraph[g, {t}],
	(* [52] *)
	"TriplesTemplate" -> t : FixedOrder[GrammarToken["TriplesSameSubject"], Repeated[FixedOrder[".", Repeated[GrammarToken["TriplesTemplate"], {0, 1}]], {0, 1}]] :> Sequence @@ DeleteCases[{t}, "."],
	(* [53] *)
	"GroupGraphPattern" -> {
		FixedOrder["{", sub : GrammarToken["SubSelect"], "}"] :> sub,
		FixedOrder["{", ggp : GrammarToken["GroupGraphPatternSub"], "}"] :> DeleteCases[{ggp}, "."]
	},
	(* [54] *)
	"GroupGraphPatternSub" -> x : FixedOrder[Repeated[GrammarToken["TriplesBlock"], {0, 1}], FixedOrder[GrammarToken["GraphPatternNotTriples"], Repeated[".", {0, 1}], Repeated[GrammarToken["TriplesBlock"], {0, 1}]] ...] :> x,
	(* [55] *)
	"TriplesBlock" -> x : FixedOrder[GrammarToken["TriplesSameSubjectPath"], Repeated[FixedOrder[".", Repeated[GrammarToken["TriplesBlock"], {0, 1}]], {0, 1}]] :> x,
	(* [56] *)
	"GraphPatternNotTriples" -> {GrammarToken["GroupOrUnionGraphPattern"], GrammarToken["OptionalGraphPattern"], GrammarToken["MinusGraphPattern"], GrammarToken["GraphGraphPattern"], GrammarToken["ServiceGraphPattern"], GrammarToken["Filter"], GrammarToken["Bind"], GrammarToken["InlineData"]},
	(* [57] *)
	"OptionalGraphPattern" -> FixedOrder["OPTIONAL", g : GrammarToken["GroupGraphPattern"]] :> SPARQLOptional[g],
	(* [58] *)
	"GraphGraphPattern" -> FixedOrder["GRAPH", v : GrammarToken["VarOrIri"], g : GrammarToken["GroupGraphPattern"]] :> SPARQLGraph[v, g],
	(* [59] *)
	"ServiceGraphPattern" -> FixedOrder["SERVICE", s : GrammarToken["SILENT"], v : GrammarToken["VarOrIri"], g : GrammarToken["GroupGraphPattern"]] :> SPARQLService[v, g, s],
	(* [60] *)
	"Bind" -> FixedOrder["BIND", "(", e : GrammarToken["Expression"], "AS", v : GrammarToken["Var"], ")"] :> First[v] -> e,
	(* [61] *)
	"InlineData" -> FixedOrder["VALUES", db : GrammarToken["DataBlock"]] :> SPARQLValues @@ db,
	(* [62] *)
	"DataBlock" -> {GrammarToken["InlineDataOneVar"], GrammarToken["InlineDataFull"]},
	(* [63] *)
	"InlineDataOneVar" -> FixedOrder[var : GrammarToken["Var"], "{", data : GrammarToken["DataBlockValue"] ..., "}"] :> {First[var], {data}},
	(* [64] *)
	"InlineDataFull" -> FixedOrder[v : GrammarToken["NIL"] | FixedOrder["(", GrammarToken["Var"] ..., ")"], "{", data : (FixedOrder["(", GrammarToken["DataBlockValue"] ..., ")"] | GrammarToken["NIL"]) ..., "}"] :> Replace[{v}, {
		{"(", vars___, ")"} :> {{vars}[[All, 1]], Partition[{data}, Length[{vars}] + 2][[All, 2 ;; -2]]},
		_ -> {{}, {}}
	}],
	(* [65] *)
	"DataBlockValue" -> {GrammarToken["iri"], GrammarToken["RDFLiteral"], GrammarToken["NumericLiteral"], GrammarToken["BooleanLiteral"], "UNDEF" -> Undefined},
	(* [66] *)
	"MinusGraphPattern" -> FixedOrder["MINUS", g : GrammarToken["GroupGraphPattern"]] :> Except[g],
	(* [67] *)
	"GroupOrUnionGraphPattern" -> u : FixedOrder[GrammarToken["GroupGraphPattern"], FixedOrder["UNION", GrammarToken["GroupGraphPattern"]] ...] :> Replace[{u}, {
		{x_} :> x,
		l_ :> Alternatives @@ l[[;; ;; 2]]
	}],
	(* [68] *)
	"Filter" -> FixedOrder["FILTER", c : GrammarToken["Constraint"]] :> filter[c],
	(* [69] *)
	"Constraint" -> {GrammarToken["BrackettedExpression"], GrammarToken["BuiltInCall"], GrammarToken["FunctionCall"]},
	(* [70] *)
	"FunctionCall" -> FixedOrder[f : GrammarToken["iri"], args : GrammarToken["ArgList"]] :> SPARQLEvaluation[f] @@ args,
	(* [71] *)
	"ArgList" -> {GrammarToken["NIL"], FixedOrder["(", d : Repeated[GrammarToken["DISTINCT"], {0, 1}], e1 : GrammarToken["Expression"], erest : FixedOrder[",", GrammarToken["Expression"]] ..., ")"] :> Join[{e1, erest}[[;; ;; 2]], {d}]},
	(* [72] *)
	"ExpressionList" -> {GrammarToken["NIL"], FixedOrder["(", e1 : GrammarToken["Expression"], erest : FixedOrder[",", GrammarToken["Expression"]] ..., ")"] :> {e1, erest}[[;; ;; 2]]},
	(* [73] *)
	"ConstructTemplate" -> FixedOrder["{", c : Repeated[GrammarToken["ConstructTriples"], {0, 1}], "}"] :> {c},
	(* [74] *)
	"ConstructTriples" -> t : FixedOrder[GrammarToken["TriplesSameSubject"], Repeated[FixedOrder[".", Repeated[GrammarToken["ConstructTriples"], {0, 1}]], {0, 1}]] :> Sequence @@ DeleteCases[{t}, "."],
	(* [75] *)
	"TriplesSameSubject" -> t : FixedOrder[GrammarToken["VarOrTerm"], GrammarToken["PropertyListNotEmpty"]] | FixedOrder[GrammarToken["TriplesNode"], GrammarToken["PropertyList"]] :> RDFTriple[t],
	(* [76] *)
	"PropertyList" -> x : Repeated[GrammarToken["PropertyListNotEmpty"], {0, 1}] :> x,
	(* [77] *)
	"PropertyListNotEmpty" -> pol : FixedOrder[GrammarToken["Verb"], GrammarToken["ObjectList"], FixedOrder[";", Repeated[FixedOrder[GrammarToken["Verb"], GrammarToken["ObjectList"]], {0, 1}]] ...] :> Replace[{pol}, {{p_, o_} :> Sequence[p, o], l_ :> SPARQLPredicateObjectList @@ Partition[l, 2, 3]}],
	(* [78] *)
	"Verb" -> {GrammarToken["VarOrIri"], GrammarToken["a"]},
	(* [79] *)
	"ObjectList" -> ol : FixedOrder[GrammarToken["Object"], FixedOrder[",", GrammarToken["Object"]] ...] :> Replace[{ol}, {{o_} :> o, l_ :> SPARQLObjectList @@ l[[;; ;; 2]]}],
	(* [80] *)
	"Object" -> GrammarToken["GraphNode"],
	(* [81] *)
	"TriplesSameSubjectPath" -> {
		t : FixedOrder[GrammarToken["VarOrTerm"], GrammarToken["PropertyListPathNotEmpty"]] :> RDFTriple[t],
		t : FixedOrder[GrammarToken["TriplesNodePath"], GrammarToken["PropertyListPath"]] :> Replace[
			RDFTriple[t],
			RDFTriple[RDFBlankNode[x_List]] :> Function[RDFTriple[RDFBlankNode[], ##]] @@ x
		]
	},
	(* [82] *)
	"PropertyListPath" -> x : Repeated[GrammarToken["PropertyListPathNotEmpty"], {0, 1}] :> x,
	(* [83] *)
	"PropertyListPathNotEmpty" -> pol : FixedOrder[GrammarToken["VerbPath"] | GrammarToken["VerbSimple"], GrammarToken["ObjectListPath"], FixedOrder[";", Repeated[FixedOrder[GrammarToken["VerbPath"] | GrammarToken["VerbSimple"], GrammarToken["ObjectList"]], {0, 1}]] ...] :> Replace[{pol}, {{p_, o_} :> Sequence[p, o], l_ :> SPARQLPredicateObjectList @@ Partition[l, 2, 3]}],
	(* [84] *)
	"VerbPath" -> GrammarToken["Path"],
	(* [85] *)
	"VerbSimple" -> GrammarToken["Var"],
	(* [86] *)
	"ObjectListPath" -> ol : FixedOrder[GrammarToken["ObjectPath"], FixedOrder[",", GrammarToken["ObjectPath"]] ...] :> Replace[{ol}, {{o_} :> o, l_ :> SPARQLObjectList @@ l[[;; ;; 2]]}],
	(* [87] *)
	"ObjectPath" -> GrammarToken["GraphNodePath"],
	(* [88] *)
	"Path" -> GrammarToken["PathAlternative"],
	(* [89] *)
	"PathAlternative" -> FixedOrder[s : GrammarToken["PathSequence"], srest : FixedOrder["|", GrammarToken["PathSequence"]] ...] :> Replace[{s, srest}, {{x_} :> x, l_ :> Alternatives @@ l[[;; ;; 2]]}],
	(* [90] *)
	"PathSequence" -> FixedOrder[s : GrammarToken["PathEltOrInverse"], srest : FixedOrder["/", GrammarToken["PathEltOrInverse"]] ...] :> Replace[{s, srest}, {{x_} :> x, l_ :> PatternSequence @@ l[[;; ;; 2]]}],
	(* [91] *)
	"PathElt" -> FixedOrder[p : GrammarToken["PathPrimary"], m : Repeated[GrammarToken["PathMod"], {0, 1}]] :> Replace[{p, m}, {{_} -> p, {_, mod_} :> mod[p]}],
	(* [92] *)
	"PathEltOrInverse" -> {GrammarToken["PathElt"], FixedOrder["^", p : GrammarToken["PathElt"]] :> SPARQLInverseProperty[p]},
	(* [93] *)
	"PathMod" -> {"? " -> Function[Repeated[#, {0, 1}]], Verbatim["*"] -> RepeatedNull, "+" -> Repeated},
	(* [94] *)
	"PathPrimary" -> {GrammarToken["iri"], GrammarToken["a"], FixedOrder["!", nps : GrammarToken["PathNegatedPropertySet"]] :> Except[nps], FixedOrder["(", p : GrammarToken["Path"], ")"] :> p},
	(* [95] *)
	"PathNegatedPropertySet" -> {GrammarToken["PathOneInPropertySet"], FixedOrder["(", p : Repeated[FixedOrder[GrammarToken["PathOneInPropertySet"], FixedOrder["|", GrammarToken["PathOneInPropertySet"]] ...], {0, 1}], ")"] :> Replace[{p}, {{x_} :> x, l_ :> Alternatives @@ l[[;; ;; 2]]}]},
	(* [96] *)
	"PathOneInPropertySet" -> {GrammarToken["iri"], GrammarToken["a"], FixedOrder["^", i : GrammarToken["iri"] | GrammarToken["a"]] :> Except[i]},
	(* [98] *)
	"TriplesNode" -> {GrammarToken["Collection"], GrammarToken["BlankNodePropertyList"]},
	(* [99] *)
	"BlankNodePropertyList" -> FixedOrder["[", pol : GrammarToken["PropertyListNotEmpty"], "]"] :> RDFBlankNode[Replace[{pol}, {x_} :> x]],
	(* [100] *)
	"TriplesNodePath" -> {GrammarToken["CollectionPath"], GrammarToken["BlankNodePropertyListPath"]},
	(* [101] *)
	"BlankNodePropertyListPath" -> FixedOrder["[", pol : GrammarToken["PropertyListPathNotEmpty"], "]"] :> RDFBlankNode[Replace[{pol}, {x_} :> x]],
	(* [102] *)
	"Collection" -> FixedOrder["(", gn : GrammarToken["GraphNode"] .., ")"] :> RDFCollection[{gn}],
	(* [103] *)
	"CollectionPath" -> FixedOrder["(", gnp : GrammarToken["GraphNodePath"] .., ")"] :> RDFCollection[{gnp}],
	(* [104] *)
	"GraphNode" -> {GrammarToken["VarOrTerm"], GrammarToken["TriplesNode"]},
	(* [105] *)
	"GraphNodePath" -> {GrammarToken["VarOrTerm"], GrammarToken["TriplesNodePath"]},
	(* [106] *)
	"VarOrTerm" -> {GrammarToken["Var"], GrammarToken["GraphTerm"]},
	(* [107] *)
	"VarOrIri" -> {GrammarToken["Var"], GrammarToken["iri"]},
	(* [108] *)
	"Var" -> v : VAR1 | VAR2 :> SPARQLVariable[StringDrop[v, 1]],
	(* [109] *)
	"GraphTerm" -> {GrammarToken["iri"], GrammarToken["RDFLiteral"], GrammarToken["NumericLiteral"], GrammarToken["BooleanLiteral"], GrammarToken["BlankNode"], GrammarToken["NIL"]},
	(* [110] *)
	"Expression" -> GrammarToken["ConditionalOrExpression"],
	(* [111] *)
	"ConditionalOrExpression" -> o : FixedOrder[GrammarToken["ConditionalAndExpression"], FixedOrder["||", GrammarToken["ConditionalAndExpression"]] ...] :> Or @@ {o}[[;; ;; 2]],
	(* [112] *)
	"ConditionalAndExpression" -> a : FixedOrder[GrammarToken["ValueLogical"], FixedOrder["&&", GrammarToken["ValueLogical"]] ...] :> And @@ {a}[[;; ;; 2]],
	(* [113] *)
	"ValueLogical" -> GrammarToken["RelationalExpression"],
	(* [114] *)
	"RelationalExpression" -> FixedOrder[
		n : GrammarToken["NumericExpression"],
		r : Repeated[
			Alternatives[
				FixedOrder[GrammarToken["Equal"], GrammarToken["NumericExpression"]],
				FixedOrder[GrammarToken["Unequal"], GrammarToken["NumericExpression"]],
				FixedOrder[GrammarToken["Less"], GrammarToken["NumericExpression"]],
				FixedOrder[GrammarToken["Greater"], GrammarToken["NumericExpression"]],
				FixedOrder[GrammarToken["LessEqual"], GrammarToken["NumericExpression"]],
				FixedOrder[GrammarToken["GreaterEqual"], GrammarToken["NumericExpression"]],
				FixedOrder[GrammarToken["IN"], GrammarToken["ExpressionList"]],
				FixedOrder[GrammarToken["NOTIN"], GrammarToken["ExpressionList"]]
			],
			{0, 1}
		]
	] :> Replace[{n, r}, {{x_} :> x, {x_, op_, y_} :> op[x, y]}],
	(* [115] *)
	"NumericExpression" -> GrammarToken["AdditiveExpression"],
	(* [116] *)
	"AdditiveExpression" -> FixedOrder[
		m : GrammarToken["MultiplicativeExpression"],
		r : (FixedOrder[GrammarToken["Plus"], GrammarToken["MultiplicativeExpression"]] | FixedOrder[GrammarToken["Subtract"], GrammarToken["MultiplicativeExpression"]] | FixedOrder[GrammarToken["NumericLiteralPositive"] | GrammarToken["NumericLiteralNegative"], (FixedOrder[GrammarToken["Times"], GrammarToken["UnaryExpression"]] | FixedOrder[GrammarToken["Divide"], GrammarToken["UnaryExpression"]]) ...]) ...
	] :> Replace[{m, r}, {{x_} :> x, _ :> Fold[First[#2][#, Last[#2]] &, m, Partition[{r}, 2]]}],
	(* [117] *)
	"MultiplicativeExpression" -> FixedOrder[
		m : GrammarToken["UnaryExpression"],
		r : (FixedOrder[GrammarToken["Times"], GrammarToken["UnaryExpression"]] | FixedOrder[GrammarToken["Divide"], GrammarToken["UnaryExpression"]]) ...
	] :> Replace[{m, r}, {{x_} :> x, _ :> Fold[First[#2][#, Last[#2]] &, m, Partition[{r}, 2]]}],
	(* [118] *)
	"UnaryExpression" -> {
		FixedOrder["!", e : GrammarToken["PrimaryExpression"]] :> ! e,
		FixedOrder["+", e : GrammarToken["PrimaryExpression"]] :> + e,
		FixedOrder["-", e : GrammarToken["PrimaryExpression"]] :> - e,
		GrammarToken["PrimaryExpression"]
	},
	(* [119] *)
	"PrimaryExpression" -> {GrammarToken["BrackettedExpression"], GrammarToken["BuiltInCall"], GrammarToken["iriOrFunction"], GrammarToken["RDFLiteral"], GrammarToken["NumericLiteral"], GrammarToken["BooleanLiteral"], GrammarToken["Var"]},
	(* [120] *)
	"BrackettedExpression" -> FixedOrder["(", e : GrammarToken["Expression"], ")"] :> e,
	(* [121] *)
	"BuiltInCall" -> a : Alternatives[
		GrammarToken["Aggregate"],
		FixedOrder["STR", "(", GrammarToken["Expression"], ")"],
		FixedOrder["LANG", "(", GrammarToken["Expression"], ")"],
		FixedOrder["LANGMATCHES", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["DATATYPE", "(", GrammarToken["Expression"], ")"],
		FixedOrder["BOUND", "(", GrammarToken["Var"], ")"],
		FixedOrder["IRI", "(", GrammarToken["Expression"], ")"],
		FixedOrder["URI", "(", GrammarToken["Expression"], ")"],
		FixedOrder["BNODE", FixedOrder["(", GrammarToken["Expression"], ")"] | GrammarToken["NIL"]],
		FixedOrder["RAND", GrammarToken["NIL"]],
		FixedOrder["ABS", "(", GrammarToken["Expression"], ")"],
		FixedOrder["CEIL", "(", GrammarToken["Expression"], ")"],
		FixedOrder["FLOOR", "(", GrammarToken["Expression"], ")"],
		FixedOrder["ROUND", "(", GrammarToken["Expression"], ")"],
		FixedOrder["CONCAT", GrammarToken["ExpressionList"]],
		GrammarToken["SubstringExpression"],
		FixedOrder["STRLEN", "(", GrammarToken["Expression"], ")"],
		GrammarToken["StrReplaceExpression"],
		FixedOrder["UCASE", "(", GrammarToken["Expression"], ")"],
		FixedOrder["LCASE", "(", GrammarToken["Expression"], ")"],
		FixedOrder["ENCODE_FOR_URI", "(", GrammarToken["Expression"], ")"],
		FixedOrder["CONTAINS", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["STRSTARTS", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["STRENDS", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["STRBEFORE", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["STRAFTER", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["YEAR", "(", GrammarToken["Expression"], ")"],
		FixedOrder["MONTH", "(", GrammarToken["Expression"], ")"],
		FixedOrder["DAY", "(", GrammarToken["Expression"], ")"],
		FixedOrder["HOURS", "(", GrammarToken["Expression"], ")"],
		FixedOrder["MINUTES", "(", GrammarToken["Expression"], ")"],
		FixedOrder["SECONDS", "(", GrammarToken["Expression"], ")"],
		FixedOrder["TIMEZONE", "(", GrammarToken["Expression"], ")"],
		FixedOrder["TZ", "(", GrammarToken["Expression"], ")"],
		FixedOrder["NOW", GrammarToken["NIL"]],
		FixedOrder["UUID", GrammarToken["NIL"]],
		FixedOrder["STRUUID", GrammarToken["NIL"]],
		FixedOrder["MD5", "(", GrammarToken["Expression"], ")"],
		FixedOrder["SHA1", "(", GrammarToken["Expression"], ")"],
		FixedOrder["SHA256", "(", GrammarToken["Expression"], ")"],
		FixedOrder["SHA384", "(", GrammarToken["Expression"], ")"],
		FixedOrder["SHA512", "(", GrammarToken["Expression"], ")"],
		FixedOrder["COALESCE", GrammarToken["ExpressionList"]],
		FixedOrder["IF", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["STRLANG", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["STRDT", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["sameTerm", "(", GrammarToken["Expression"], ",", GrammarToken["Expression"], ")"],
		FixedOrder["isIRI", "(", GrammarToken["Expression"], ")"],
		FixedOrder["isURI", "(", GrammarToken["Expression"], ")"],
		FixedOrder["isBLANK", "(", GrammarToken["Expression"], ")"],
		FixedOrder["isLITERAL", "(", GrammarToken["Expression"], ")"],
		FixedOrder["isNUMERIC", "(", GrammarToken["Expression"], ")"],
		GrammarToken["RegexExpression"],
		GrammarToken["ExistsFunc"],
		GrammarToken["NotExistsFunc"]
	] :> Replace[{a}, {
		{f_, args_List} :> SPARQLEvaluation[f] @@ args,
		{f_, "(", args___, ")"} :> SPARQLEvaluation[f] @@ {args}[[;; ;; 2]],
		{f_} :> f
	}],
	(* [122] *)
	"RegexExpression" -> FixedOrder["REGEX", "(", e1 : GrammarToken["Expression"], ",", e2 : GrammarToken["Expression"], e3 : Repeated[FixedOrder[",", GrammarToken["Expression"]], {0 , 1}], ")"] :> SPARQLEvaluation["REGEX"][e1, e2, Sequence @@ Last[{e3}, {}]],
	(* [123] *)
	"SubstringExpression" -> FixedOrder["SUBSTR", "(", e1 : GrammarToken["Expression"], ",", e2 : GrammarToken["Expression"], e3 : Repeated[FixedOrder[",", GrammarToken["Expression"]], {0 , 1}], ")"] :> SPARQLEvaluation["SUBSTR"][e1, e2, Sequence @@ Last[{e3}, {}]],
	(* [124] *)
	"StrReplaceExpression" -> FixedOrder["REPLACE", "(", e1 : GrammarToken["Expression"], ",", e2 : GrammarToken["Expression"], ",", e3 : GrammarToken["Expression"], e4 : Repeated[FixedOrder[",", GrammarToken["Expression"]], {0 , 1}], ")"] :> SPARQLEvaluation["REPLACE"][e1, e2, e3, Sequence @@ Last[{e4}, {}]],
	(* [125] *)
	"ExistsFunc" -> FixedOrder["EXISTS", ggp : GrammarToken["GroupGraphPattern"]] :> SPARQLEvaluation["EXISTS"][ggp],
	(* [126] *)
	"NotExistsFunc" -> FixedOrder["NOT", "EXISTS", ggp : GrammarToken["GroupGraphPattern"]] :> Not[SPARQLEvaluation["EXISTS"][ggp]],
	(* [127] *)
	"Aggregate" -> {
		FixedOrder[f : "COUNT", "(", d : Repeated[GrammarToken["DISTINCT"], {0, 1}], arg : GrammarToken["COUNTStar"] | GrammarToken["Expression"], ")"] :> SPARQLEvaluation[d /* f][arg],
		FixedOrder[f : "SUM", "(", d : Repeated[GrammarToken["DISTINCT"], {0, 1}], arg : GrammarToken["Expression"], ")"] :> SPARQLEvaluation[d /* f][arg],
		FixedOrder[f : "MIN", "(", d : Repeated[GrammarToken["DISTINCT"], {0, 1}], arg : GrammarToken["Expression"], ")"] :> SPARQLEvaluation[d /* f][arg],
		FixedOrder[f : "MAX", "(", d : Repeated[GrammarToken["DISTINCT"], {0, 1}], arg : GrammarToken["Expression"], ")"] :> SPARQLEvaluation[d /* f][arg],
		FixedOrder[f : "AVG", "(", d : Repeated[GrammarToken["DISTINCT"], {0, 1}], arg : GrammarToken["Expression"], ")"] :> SPARQLEvaluation[d /* f][arg],
		FixedOrder[f : "SAMPLE", "(", d : Repeated[GrammarToken["DISTINCT"], {0, 1}], arg : GrammarToken["Expression"], ")"] :> SPARQLEvaluation[d /* f][arg],
		FixedOrder[f : "GROUP_CONCAT", "(", d : Repeated[GrammarToken["DISTINCT"], {0, 1}], arg : GrammarToken["Expression"], opt : Repeated[GrammarToken["SEPARATOR"], {0, 1}], ")"] :> SPARQLEvaluation[d /* f][arg, opt]
	},
	(* [128] *)
	"iriOrFunction" -> ia : FixedOrder[GrammarToken["iri"], Repeated[GrammarToken["ArgList"], {0, 1}]] :> Replace[{ia}, {{i_} :> i, {f_, args_} :> SPARQLEvaluation[f] @@ args}],
	(* [129] *)
	"RDFLiteral" -> l : FixedOrder[GrammarToken["String"], Repeated[GrammarToken["LANGTAG"] | FixedOrder["^^", GrammarToken["iri"]], {0, 1}]] :> Replace[{l}, {
		{s_, "^^", dt_} :> FromRDFLiteral[RDFLiteral[s, dt]],
		{s_, lang_String} :> RDFString[s, lang],
		{s_} :> s
	}],
	(* [130] *)
	"NumericLiteral" -> {GrammarToken["NumericLiteralUnsigned"], GrammarToken["NumericLiteralPositive"], GrammarToken["NumericLiteralNegative"]},
	(* [131] *)
	"NumericLiteralUnsigned" -> {i : INTEGER :> Interpreter["Integer"][i], d : DECIMAL | DOUBLE :> Interpreter["Real"][d]},
	(* [132] *)
	"NumericLiteralPositive" -> {i : INTEGERPOSITIVE :> Interpreter["Integer"][i], d : DECIMALPOSITIVE | DOUBLEPOSITIVE :> Interpreter["Real"][d]},
	(* [133] *)
	"NumericLiteralNegative" -> {i : INTEGERNEGATIVE :> Interpreter["Integer"][i], d : DECIMALNEGATIVE | DOUBLENEGATIVE :> Interpreter["Real"][d]},
	(* [134] *)
	"BooleanLiteral" -> {"true" -> True, "false" -> False},
	(* [135] *)
	"String" -> {
		s : STRINGLITERAL1 | STRINGLITERAL2 :> StringDecode[StringTake[s, {2, -2}]],
		s : STRINGLITERALLONG1 | STRINGLITERALLONG2 :> StringDecode[StringTake[s, {4, -4}]]
	},
	(* [136] *)
	"iri" -> {GrammarToken["IRIREF"], GrammarToken["PrefixedName"]},
	(* [137] *)
	"PrefixedName" -> p : PNAMELN | PNAMENS :> IRI[MapAt[StringReplace["\\" ~~ x_ :> x], StringSplit[p, ":", 2], 2]],
	(* [138] *)
	"BlankNode" -> {b : BLANKNODELABEL :> RDFBlankNode[StringDrop[b, 2]], ANON -> RDFBlankNode[]},


	"a" -> CaseSensitive["a"] -> IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
	"ASC" -> "ASC" -> "Ascending",
	"COUNTStar" -> Verbatim["*"] -> Sequence[],
	"DESC" -> "DESC" -> "Descending",
	"DISTINCT" -> "DISTINCT" -> SPARQLDistinct[],
	"Divide" -> "/" -> Inactive[Divide],
	"Equal" -> "=" -> Equal,
	"Greater" -> ">" -> Greater,
	"GreaterEqual" -> ">=" -> GreaterEqual,
	"IN" -> "IN" -> SPARQLEvaluation["IN"],
	"IRIREF" -> i : IRIREF :> IRI[StringTake[i, {2, -2}]],
	"LANGTAG" -> l : LANGTAG :> StringDrop[l, 1],
	"Less" -> "<" -> Less,
	"LessEqual" -> "<=" -> LessEqual,
	"NIL" -> NIL -> {},
	"NOTIN" -> FixedOrder["NOT", in : GrammarToken["IN"]] :> Not @* in,
	"Plus" -> "+" -> Plus,
	"REDUCED" -> "REDUCED" -> SPARQLDistinct[Method -> "Reduced"],
	"SEPARATOR" -> FixedOrder[";", "SEPARATOR", "=", sep : GrammarToken["String"]] :> "Separator" -> sep,
	"SILENT" -> s : Repeated["SILENT", {0, 1}] :> Replace[{s}, {{} :> Sequence[], {_} :> Sequence["Silent" -> True]}],
	"Subtract" -> "-" -> Subtract,
	"Times" -> Verbatim["*"] -> Times,
	"Unequal" -> "!=" -> Unequal,
	"With" -> x : Repeated[FixedOrder["WITH", GrammarToken["iri"]], {0, 1}] :> Replace[{x}, {
		{} :> Sequence[],
		{_, i_} :> Sequence["With" -> i]
	}]
}];


(* terminals *)
(* [139] *) IRIREF := "<" ~~ Except["^" | "<" | ">" | "\"" | "{" | "}" | "|" | "^" | "`" | "\\"] ... ~~ ">";
(* [140] *) PNAMENS := Repeated[PNPREFIX, {0, 1}] ~~ ":";
(* [141] *) PNAMELN := PNAMENS ~~ PNLOCAL;
(* [142] *) BLANKNODELABEL := "_:" ~~ PNCHARSU | DigitCharacter ~~ Repeated[(PNCHARS | ".") ... ~~ PNCHARS, {0, 1}];
(* [143] *) VAR1 := "?" ~~ VARNAME;
(* [144] *) VAR2 := "$" ~~ VARNAME;
(* [145] *) LANGTAG := RegularExpression["@[a-zA-Z]+(-[a-zA-Z0-9]+)*"];
(* [146] *) INTEGER := DigitCharacter ..;
(* [147] *) DECIMAL := DigitCharacter ... ~~ "." ~~ DigitCharacter ..;
(* [148] *) DOUBLE := (DigitCharacter .. ~~ "." ~~ DigitCharacter ... ~~ EXPONENT) | ("." ~~ DigitCharacter .. ~~ EXPONENT) | (DigitCharacter .. ~~ EXPONENT);
(* [149] *) INTEGERPOSITIVE := "+" ~~ INTEGER;
(* [150] *) DECIMALPOSITIVE := "+" ~~ DECIMAL;
(* [151] *) DOUBLEPOSITIVE := "+" ~~ DOUBLE;
(* [152] *) INTEGERNEGATIVE := "-" ~~ INTEGER;
(* [153] *) DECIMALNEGATIVE := "-" ~~ DECIMAL;
(* [154] *) DOUBLENEGATIVE := "-" ~~ DOUBLE;
(* [155] *) EXPONENT := "e" | "E" ~~ Repeated["+" | "-", {0, 1}] ~~ DigitCharacter ..;
(* [156] *) STRINGLITERAL1 := "'" ~~ (Except["'" | "\\" | "\n" | "\r"] | ECHAR) ... ~~ "'";
(* [157] *) STRINGLITERAL2 := "\"" ~~ (Except["\"" | "\\" | "\n" | "\r"] | ECHAR) ... ~~ "\"";
(* [158] *) STRINGLITERALLONG1 := "'''" ~~ (Repeated["'" | "''", {0, 1}] ~~ Except["'"] | ECHAR) ... ~~ "'''";
(* [159] *) STRINGLITERALLONG2 := "\"\"\"" ~~ (Repeated["\"" | "\"\"", {0, 1}] ~~ Except["\""] | ECHAR) ... ~~ "\"\"\"";
(* [160] *) ECHAR := "\\" ~~ "t" | "b" | "n" | "r" | "f" | "\\" | "\"" | "'";
(* [161] *) NIL := "(" ~~ WS ... ~~ ")";
(* [162] *) WS := " " | "\t" | "\r" | "\n";
(* [163] *) ANON := "[" ~~ WS ... ~~ "]";
(* [164] *) PNCHARSBASE := LetterCharacter;
(* [165] *) PNCHARSU := PNCHARSBASE | "_";
(* [166] *) VARNAME := PNCHARSU | DigitCharacter ~~ (PNCHARSU | DigitCharacter | "\[CenterDot]" | Alternatives @@ CharacterRange[FromDigits["0300", 16], FromDigits["036F", 16]] | Alternatives @@ CharacterRange[FromDigits["203F", 16], FromDigits["2040", 16]]) ...;
(* [167] *) PNCHARS := 	PNCHARSU | "-" | DigitCharacter;
(* [168] *) PNPREFIX := PNCHARSBASE ~~ Repeated[(PNCHARS | ".") ... ~~ PNCHARS, {0, 1}];
(* [169] *) PNLOCAL := PNCHARSU | ":" | DigitCharacter | PLX ~~ Repeated[(PNCHARS | "." | ":" | PLX) ... ~~ PNCHARS | ":" | PLX, {0, 1}];
(* [170] *) PLX := PERCENT | PNLOCALESC;
(* [171] *) PERCENT := "%" ~~ Repeated[HEX, {2}];
(* [172] *) HEX := HexadecimalCharacter;
(* [173] *) PNLOCALESC := "\\" ~~ "_" | "~" | "." | "-" | "!" | "$" | "&" | "'" | "(" | ")" | Verbatim["*"] | "+" | "," | ";" | "=" | "/" | "?" | "#" | Verbatim["@"] | "%";

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
