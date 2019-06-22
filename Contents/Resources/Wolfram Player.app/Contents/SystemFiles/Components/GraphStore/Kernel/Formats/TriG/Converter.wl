(* RDF 1.1 TriG *)
(* https://www.w3.org/TR/trig/ *)

BeginPackage["GraphStore`Formats`TriG`", {"GraphStore`"}];

Needs["GraphStore`Formats`Utilities`"];
Needs["GraphStore`Libraries`SerdLink`"];
Needs["GraphStore`RDF`"];

ExportTriG;
Options[ExportTriG] = {
	"Base" -> None,
	"Indentation" -> "  ",
	"Prefixes" -> <||>
};

ImportTriG;
Options[ImportTriG] = {
	"Base" -> Automatic
};

ImportTriGBase;

ImportTriGPrefixes;

Begin["`Private`"];

ExportTriG[args___] := Catch[iExportTriG[args], $failTag, (Message[Export::fmterr, "TriG"]; #) &];
ImportTriG[file_, opts : OptionsPattern[]] := Catch[iImportTriG[file, FilterRules[{opts}, Options[ImportTriG]]], $failTag, (Message[Import::fmterr, "TriG"]; #) &];
ImportTriGBase[file_, opts : OptionsPattern[]] := Catch[iImportTriGBase[file, FilterRules[{opts}, Options[ImportTriGBase]]], $failTag, (Message[Import::fmterr, "TriG"]; #) &];
ImportTriGPrefixes[file_, opts : OptionsPattern[]] := Catch[iImportTriGPrefixes[file, FilterRules[{opts}, Options[ImportTriGPrefixes]]], $failTag, (Message[Import::fmterr, "TriG"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* export *)

clear[iExportTriG];
Options[iExportTriG] = Options[ExportTriG];
iExportTriG[file_, data_, OptionsPattern[]] := Block[
	{$indentation = OptionValue["Indentation"]},
	Export[
		file,
		trigDocToString[data, OptionValue["Prefixes"], OptionValue["Base"]],
		"Text",
		CharacterEncoding -> "UTF-8"
	]
];


clear[graphToString];
Options[graphToString] = {
	"Indentation" -> ""
};
graphToString[g_List, OptionsPattern[]] := StringJoin[Riffle[
	g //
	CompactRDFCollection //
	(* GroupBy[{First -> Rest, First -> Last}] is very slow: https://bugs.wolfram.com/show?number=355456 *)
	GroupBy[First -> Rest] // Map[GroupBy[First -> Last]] //
	Curry[FixedPoint, 2][toBlankNodePropertyList] //
	KeySort //
	KeyValueMap[Function[{s, pol},
		{
			OptionValue["Indentation"],
			subjectToString[s],
			If[Length[pol] === 1, " ", "\n" <> OptionValue["Indentation"] <> $indentation],
			predicateObjectListToString[pol, "Indentation" -> OptionValue["Indentation"] <> If[Length[pol] === 1, "", $indentation]],
			" ."
		}
	]],
	"\n"
]];

clear[toBlankNodePropertyList];
toBlankNodePropertyList[data_] := Module[
	{res = data},
	Intersection[
		(* blank nodes that are used once in the object position ... *)
		Keys[DeleteCases[Counts[Cases[res, RDFBlankNode[_String], {3}]], Except[1]]],
		(* ... and exist as subject *)
		Keys[res]
	] // Scan[Function[b,
		With[
			{s = res[b]},
			If[! MemberQ[s, RDFBlankNode[_String], {2}],
				res = res /. b -> RDFBlankNode[s];
				KeyDropFrom[res, b];
			]
		];
	]];
	res
];

clear[xsd];
xsd[s_String] := IRI["http://www.w3.org/2001/XMLSchema#" <> s];

clear[normalizeLiterals];
normalizeLiterals[g_List] := Replace[
	g,
	RDFTriple[s_, p_, Except[_IRI | _RDFBlankNode | _RDFLiteral | _RDFString, o_]] :> RDFTriple[s, p, ToRDFLiteral[o]],
	{1}
] /. RDFLiteral[s_, dt_String] :> RDFLiteral[s, IRI[dt]];

clear[compact];
compact[expr_, prefixes_, base_] := Module[
	{tmp},
	RDFCompactIRIs[
		expr /. RDFLiteral[s_, dt : xsd["integer"] | xsd["double"] | xsd["boolean"] | xsd["decimal"] | xsd["string"]] :> RDFLiteral[s, tmp @@ dt],
		prefixes,
		base
	] /. tmp[dt_] :> IRI[dt]
];

clear[collectPrefixes];
SetAttributes[collectPrefixes, HoldRest];
collectPrefixes[expr_, prefixes_Symbol] := (
	prefixes = DeleteDuplicates[Join[prefixes, Cases[expr, IRI[{pre_, _}] :> pre, {0, Infinity}]]];
	expr
);


(* [1g] *)
clear[trigDocToString];
trigDocToString[store_RDFStore, prefixes_, base_] := Module[
	{defaultString, namedString, usedPrefixes = {}},
	defaultString = graphToString[collectPrefixes[compact[normalizeLiterals[store["DefaultGraph"]], prefixes, base], usedPrefixes] /. RDFBlankNode[] :> RDFBlankNode[Unique[b]]] // Replace["" :> Nothing];
	namedString = store["NamedGraphs"] // KeyValueMap[Function[{label, g},
		StringJoin[{
			labelOrSubjectToString[collectPrefixes[compact[label, prefixes, base], usedPrefixes]],
			" ",
			wrappedGraphToString[collectPrefixes[compact[normalizeLiterals[g], prefixes, base], usedPrefixes] /. RDFBlankNode[] :> RDFBlankNode[Unique[b]]]
		}]
	]];
	StringJoin[Riffle[
		Flatten[{
			directiveToString[If[prefixes === Automatic, AssociationMap[RDFPrefixData, Sort[usedPrefixes]], prefixes], base] // Replace["" :> Nothing],
			defaultString,
			namedString
		}],
		"\n\n"
	]]
];

(* [5g] *)
clear[wrappedGraphToString];
wrappedGraphToString[g_] := graphToString[g, "Indentation" -> $indentation] // Replace[{
	"" :> "{ }",
	s_ :> "{\n" <> s <> "\n}"
}];

(* [7g] *)
clear[labelOrSubjectToString];
labelOrSubjectToString[x_IRI] := iriToString[x];
labelOrSubjectToString[x_RDFBlankNode] := blankNodeToString[x];

(* [3] *)
clear[directiveToString];
directiveToString[prefixes_?AssociationQ, base_] := StringRiffle[DeleteCases[Flatten[{
	KeyValueMap[sparqlPrefixToString, prefixes],
	sparqlBaseToString[base]
}], ""], "\n"];

(* [5s] *)
clear[sparqlPrefixToString];
sparqlPrefixToString[pre_String, iri_] := "PREFIX " <> pre <> ": " <> IRIREFToString[Replace[iri, s_String :> IRI[s]]];

(* [6s] *)
clear[sparqlBaseToString];
sparqlBaseToString[None] := "";
sparqlBaseToString[iri_] := "BASE " <> IRIREFToString[Replace[iri, s_String :> IRI[s]]];

(* [7] *)
clear[predicateObjectListToString];
Options[predicateObjectListToString] = {
	"Indentation" -> ""
};
predicateObjectListToString[pol_?AssociationQ, OptionsPattern[]] := StringJoin[Riffle[
	pol // KeySort // KeyValueMap[Function[{p, ol},
		{
			verbToString[p],
			" ",
			objectListToString[ol, "Indentation" -> OptionValue["Indentation"]]
		}
	]],
	" ;\n" <> OptionValue["Indentation"]
]];

(* [8] *)
clear[objectListToString];
objectListToString[ol_List, opts : OptionsPattern[]] := StringJoin[Riffle[objectToString[#, opts] & /@ Sort[ol], " , "]];

(* [9] *)
clear[verbToString];
verbToString[IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]] := "a";
verbToString[x_] := predicateToString[x];

(* [10] *)
clear[subjectToString];
subjectToString[x_IRI] := iriToString[x];
subjectToString[x : _RDFBlankNode | _RDFCollection] := blankToString[x];

(* [11] *)
clear[predicateToString];
predicateToString[x_IRI] := iriToString[x];

(* [12] *)
clear[objectToString];
objectToString[x_IRI, OptionsPattern[]] := iriToString[x];
objectToString[x : RDFBlankNode[_String | _Symbol] | _RDFCollection, OptionsPattern[]] :=  blankToString[x];
objectToString[x : RDFBlankNode[_Association], opts : OptionsPattern[]] := blankNodePropertyListToString[x, opts];
objectToString[x_, OptionsPattern[]] := literalToString[x];

(* [13] *)
clear[literalToString];
literalToString[RDFLiteral[s_String, xsd["integer"] | xsd["double"] | xsd["boolean"]]] := s;
literalToString[RDFLiteral[s_String?(Not @* StringEndsQ["."]), xsd["decimal"]]] := s;
literalToString[RDFLiteral[s_String, xsd["string"]]] := stringToString[s];
literalToString[RDFString[s_String, lang_String]] := stringToString[s] <> "@" <> lang;
literalToString[RDFLiteral[s_String, dt_IRI]] := stringToString[s] <> "^^" <> iriToString[dt];

(* [14] *)
clear[blankToString];
blankToString[x_RDFBlankNode] := blankNodeToString[x];
blankToString[x_RDFCollection] := collectionToString[x];

(* [15] *)
clear[blankNodePropertyListToString];
Options[blankNodePropertyListToString] = {
	"Indentation" -> ""
};
blankNodePropertyListToString[RDFBlankNode[data_Association], OptionsPattern[]] := StringJoin[
	"[\n" <> OptionValue["Indentation"] <> $indentation,
	predicateObjectListToString[data, "Indentation" -> OptionValue["Indentation"] <> $indentation],
	"\n" <> OptionValue["Indentation"] <> "]"
];

(* [16] *)
clear[collectionToString]
collectionToString[RDFCollection[l_List]] := "(" <> StringRiffle[objectToString /@ l] <> ")";

(* [18] *)
clear[stringToString];
stringToString[s_String] := If[StringFreeQ[s, "\"" | "\n" | "\r"],
	"\"" <> # <> "\"",
	"\"\"\"" <> # <> "\"\"\""
] &[StringEncode[s]];

(* [135s] *)
clear[iriToString];
iriToString[IRI[{pre_, local_}]] := prefixedNameToString[pre, local];
iriToString[i_IRI] := IRIREFToString[i];

(* [136s] *)
clear[prefixedNameToString];
prefixedNameToString[pre_String, local_String] := pre <> ":" <> ReservedCharacterEncode[local];

(* [137s] *)
clear[blankNodeToString];
blankNodeToString[RDFBlankNode[l_String]] := "_:" <> l;
blankNodeToString[RDFBlankNode[_Symbol]] := "[]";

(* [19] *)
clear[IRIREFToString];
IRIREFToString[IRI[i_String]] := "<" <> i <> ">";

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportTriG];
Options[iImportTriG] = Options[ImportTriG];
iImportTriG[file_, OptionsPattern[]] := {"Data" -> (Replace[
	SerdImport[file, 4 (* SERD_TRIG *), ChooseBase[OptionValue["Base"], file] // Replace[None :> ""]] /. lit_RDFLiteral :> FromRDFLiteral[lit],
	{
		_?FailureQ :> fail[],
		l_List :> RDFStore[
			Cases[l, _RDFTriple],
			GroupBy[Cases[l, _Rule], First -> Last]
		]
	}
])};

clear[iImportTriGBase];
iImportTriGBase[file_, OptionsPattern[]] := importBasePrefixes[file];

clear[iImportTriGPrefixes];
iImportTriGPrefixes[file_, OptionsPattern[]] := importBasePrefixes[file];

clear[importBasePrefixes];
importBasePrefixes[file_] := Module[
	{base = None, prefixes = <||>, line},
	While[
		line = ReadLine[file];
		And[
			StringQ[line],
			StringReplace[
				line,
				{
					(* [4] prefix *)
					StartOfString ~~ WS ~~ CaseSensitive["@prefix"] ~~ WS ~~ pre : PNAMENS ~~ WS ~~ iri : IRIREF ~~ WS ~~ "." ~~ WS ~~ comment ~~ EndOfString :> (
						AssociateTo[prefixes, StringDrop[pre, -1] -> IRI[StringTake[iri, {2, -2}]]];
						True
					),
					(* [5] base *)
					StartOfString ~~ WS ~~ CaseSensitive["@base"] ~~ WS ~~ iri : IRIREF ~~ WS ~~ "." ~~ WS ~~ comment ~~ EndOfString :> (
						base = IRI[StringTake[iri, {2, -2}]];
						True
					),
					(* [5s] SPARQL prefix *)
					StartOfString ~~ WS ~~ "PREFIX" ~~ WS ~~ pre : PNAMENS ~~ WS ~~ iri : IRIREF ~~ WS ~~ comment ~~ EndOfString :> (
						AssociateTo[prefixes, StringDrop[pre, -1] -> IRI[StringTake[iri, {2, -2}]]];
						True
					),
					(* [6s] SPARQL base *)
					StartOfString ~~ WS ~~ "BASE" ~~ WS ~~ iri : IRIREF ~~ WS ~~ comment ~~ EndOfString :> (
						base = IRI[StringTake[iri, {2, -2}]];
						True
					),
					___ :> False
				},
				1,
				IgnoreCase -> True
			] // Replace[StringExpression[True] :> True]
		]
	];
	{"Base" -> base, "Prefixes" -> prefixes}
];

comment := "" | ("#" ~~ ___);

(* [19] *)   IRIREF := "<" ~~ (Except[Join[Alternatives @@ CharacterRange[0, 20], "<" | ">" | "\"" | "{" | "}" | "|" | "^" | "`" | "\\"]] | UCHAR) ... ~~ ">";
(* [139s] *) PNAMENS := Repeated[PNPREFIX, {0, 1}] ~~ ":";
(* [27] *)   UCHAR := ("\\u" ~~ Repeated[HexadecimalCharacter, {4}]) | ("\\U" ~~ Repeated[HexadecimalCharacter, {8}]);
(* [161s] *) WS := WhitespaceCharacter ...;
(* [163s] *) PNCHARSBASE := LetterCharacter;
(* [164s] *) PNCHARSU := PNCHARSBASE | "_";
(* [166s] *) PNCHARS := PNCHARSU | "-" | DigitCharacter;
(* [167s] *) PNPREFIX := PNCHARSBASE ~~ Repeated[(PNCHARS | ".") ... ~~ PNCHARS, {0, 1}];

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
