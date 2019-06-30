BeginPackage["GraphStore`Formats`Utilities`", {"GraphStore`"}];

Needs["GraphStore`IRI`"];
Needs["GraphStore`RDF`"];
Needs["GraphStore`SPARQL`"];

ChooseBase;
ExpandTriples;
NumericDecode;
RDFCompactIRIs;
RDFExpandIRIs;
ReservedCharacterEncode;
ReservedCharacterDecode;
SPARQLObjectList;
SPARQLPredicateObjectList;
StringDecode;
StringEncode;

Begin["`Private`"];

ChooseBase[base_, file_] := normalizeBase[If[base === Automatic, normalizeFile[file], base]];

normalizeFile[InputStream[file_String, ___]] := File[file];
normalizeFile[file_String] := File[file];
normalizeFile[file_File] := file;
normalizeFile[_] := None;

normalizeBase[f_File] := FileToIRI[f];
normalizeBase[s_String] := s;
normalizeBase[(IRI | URL)[i_]] := i;
normalizeBase[None] := None;
normalizeBase[_] := None;

ExpandTriples[expr_] := expr //. {
	RDFBlankNode[{p_, o_}] :> RDFBlankNode[SPARQLPredicateObjectList[{p, o}]],
	{x___, t_RDFTriple, y___} :> With[{exp = expand[t], l = {x, Sequence @@ expand[t], y}}, l /; exp =!= t],
	t_RDFTriple :> With[{exp = expand[t]}, exp /; exp =!= t],
	head_[x___, RDFTriple[tx___, c_RDFCollection?(Not @* FreeQ[RDFBlankNode[_SPARQLPredicateObjectList]]), ty___], y___] :> head[
		x,
		With[{
			b = EvaluateSPARQLFunction[SPARQLEvaluation["BNODE"]],
			pos = FirstPosition[c, RDFBlankNode[_SPARQLPredicateObjectList], Null, Infinity]
		},
			If[head === List, Sequence, List][
				RDFTriple[tx, ReplacePart[c, pos -> b], ty],
				RDFTriple[b, First[Extract[c, pos]]]
			]
		],
		y
	]
};


(* predicate-object list *)
expand[RDFTriple[s_, pol : SPARQLPredicateObjectList[{_, _} ...]]] := With[
	{s1 = Replace[s, RDFBlankNode[] :> EvaluateSPARQLFunction[SPARQLEvaluation["BNODE"]]]},
	Function[{p, o},
		RDFTriple[s1, p, o]
	] @@@ List @@ pol
];

(* object list *)
expand[RDFTriple[s_, p_, ol_SPARQLObjectList]] := With[
	{s1 = Replace[s, RDFBlankNode[] :> EvaluateSPARQLFunction[SPARQLEvaluation["BNODE"]]]},
	Function[o,
		RDFTriple[s1, p, o]
	] /@ List @@ ol
];

(* blank node *)
expand[RDFTriple[RDFBlankNode[l : SPARQLPredicateObjectList[{_, _} ..]], p2_, o2_]] := With[
	{b = EvaluateSPARQLFunction[SPARQLEvaluation["BNODE"]]},
	{
		Sequence @@ Function[{p1, o1},
			RDFTriple[b, p1, o1]
		] @@@ l,
		RDFTriple[b, p2, o2]
	}
];
expand[RDFTriple[s_, p_, RDFBlankNode[l : SPARQLPredicateObjectList[{_, _} ..]]]] := With[
	{b = EvaluateSPARQLFunction[SPARQLEvaluation["BNODE"]]},
	{
		RDFTriple[s, p, b],
		Sequence @@ Function[{p2, o2},
			RDFTriple[b, p2, o2]
		] @@@ l
	}
];

expand[x_] := x;

NumericDecode[s_String] := StringReplace[s, {
	"\\u" ~~ code : Repeated[HexadecimalCharacter, {4}] :> FromCharacterCode[FromDigits[code, 16], "Unicode"],
	"\\U" ~~ code : Repeated[HexadecimalCharacter, {8}] :> FromCharacterCode[FromDigits[code, 16], "Unicode"]
}];

RDFCompactIRIs[expr_, Automatic, base_] := RDFCompactIRIs[expr, AssociationMap[RDFPrefixData, RDFPrefixData[]], base];
RDFCompactIRIs[expr_, prefixes_, base_] := With[
	{np = Normal[prefixes // Map[Replace[IRI[i_String] :> i]], Association]},
	expr /. IRI[Except["http://www.w3.org/1999/02/22-rdf-syntax-ns#type", i_String]] :> FirstCase[
		np,
		(pre_String -> ns_String /; StringStartsQ[i, ns]) :> IRI[{
			pre,
			StringReplace[
				StringDrop[i, StringLength[ns]],
				x : "=" :> "\\" <> x
			]
		}],
		IRI[CompactIRI[i, base]]
	]
];

RDFExpandIRIs[expr_, <||>, None] := expr;
RDFExpandIRIs[expr_, prefix_, base_] := expr /. {
	If[prefix === None,
		Nothing,
		With[
			{a = <|prefix|> // Replace[#, (File | IRI | URL)[i_] :> i, {1}] &},
			IRI[{pre_String, local_String}] :> IRI[Lookup[a, pre] <> local]
		]
	],
	If[base === None,
		Nothing,
		IRI[i_String] :> With[{tmp = IRI[ExpandIRI[i, base]]}, tmp /; True]
	]
};

$reservedCharacters = "~" | "." | "-" | "!" | "$" | "&" | "'" | "(" | ")" | "*" | "+" | "," | ";" | "=" | "/" | "?" | "#" | "@" | "%" | "_";

ReservedCharacterEncode[s_] := StringReplace[s, c : $reservedCharacters :> "\\" <> c];

ReservedCharacterDecode[s_String] := StringReplace[s, "\\" ~~ c : $reservedCharacters :> c];

$stringEscapeRules = {
	FromCharacterCode[FromDigits["0009", 16], "Unicode"] -> "\\t",
	FromCharacterCode[FromDigits["0008", 16], "Unicode"] -> "\\b",
	FromCharacterCode[FromDigits["000A", 16], "Unicode"] -> "\\n",
	FromCharacterCode[FromDigits["000D", 16], "Unicode"] -> "\\r",
	FromCharacterCode[FromDigits["000C", 16], "Unicode"] -> "\\f",
	FromCharacterCode[FromDigits["0022", 16], "Unicode"] -> "\\\"",
	FromCharacterCode[FromDigits["0027", 16], "Unicode"] -> "\\'",
	FromCharacterCode[FromDigits["005C", 16], "Unicode"] -> "\\\\"
};

$stringUnescapeRules = Reverse /@ $stringEscapeRules;

StringDecode[s_String] := StringReplace[s, $stringUnescapeRules];

StringEncode[s_String] := StringReplace[s, $stringEscapeRules];

End[];
EndPackage[];
