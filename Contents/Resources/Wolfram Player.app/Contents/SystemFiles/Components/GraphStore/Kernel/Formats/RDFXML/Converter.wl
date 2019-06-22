(* RDF 1.1 XML Syntax *)
(* https://www.w3.org/TR/rdf-syntax-grammar/ *)

BeginPackage["GraphStore`Formats`RDFXML`", {"GraphStore`"}];

Needs["GraphStore`Formats`Utilities`"];
Needs["GraphStore`IRI`"];
Needs["GraphStore`RDF`"];

ExportRDFXML;
Options[ExportRDFXML] = {
	"Base" -> None,
	"Prefixes" -> <||>
};

ImportRDFXML;
Options[ImportRDFXML] = {
	"Base" -> Automatic
};

ImportRDFXMLBase;
ImportRDFXMLPrefixes;

Begin["`Private`"];

ExportRDFXML[args___] := Catch[iExportRDFXML[args], $failTag, (Message[Export::fmterr, "RDFXML"]; #) &];
ImportRDFXML[file_, opts : OptionsPattern[]] := Catch[iImportRDFXML[file, FilterRules[{opts}, Options[ImportRDFXML]]], $failTag, (Message[Import::fmterr, "RDFXML"]; #) &];
ImportRDFXMLBase[file_, opts : OptionsPattern[]] := Catch[iImportRDFXMLBase[file, FilterRules[{opts}, Options[ImportRDFXMLBase]]], $failTag, (Message[Import::fmterr, "RDFXML"]; #) &];
ImportRDFXMLPrefixes[file_, opts : OptionsPattern[]] := Catch[iImportRDFXMLPrefixes[file, FilterRules[{opts}, Options[ImportRDFXMLPrefixes]]], $failTag, (Message[Import::fmterr, "RDFXML"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* common *)

clear[rdf];
rdf[] := "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
rdf[s_String] := rdf[] <> s;

(* end common *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* export *)

clear[iExportRDFXML];
Options[iExportRDFXML] = Options[ExportRDFXML];
iExportRDFXML[file_, data_, OptionsPattern[]] := Export[
	file,
	dataToXML[
		data,
		OptionValue["Prefixes"] // Map[Replace[IRI[i_String] :> i]],
		OptionValue["Base"]
	],
	"XML"
];

clear[dataToXML];
dataToXML[store_RDFStore, prefixes_?AssociationQ, base_] := XMLObject["Document"][
	{XMLObject["Declaration"]["Version" -> "1.0"]},
	XMLElement[
		{"rdf", "RDF"},
		DeleteDuplicates[Join[
			{{"xmlns", "rdf"} -> rdf[]},
			KeyValueMap[
				Function[{prefix, ns},
					{"xmlns", prefix} -> ns
				],
				prefixes
			],
			If[base =!= None, {{"xml", "base"} -> Replace[base, IRI[i_String] :> i]}, {}]
		]],
		Block[{
			$base = base,
			$prefixes = Normal[prefixes, Association],
			$prefixCounter = 1
		},
			tripleToXML /@ store["DefaultGraph"]
		]
	],
	{}
];

clear[tripleToXML];
tripleToXML[RDFTriple[s_, p_, o_]] := XMLElement[
	{"rdf", "Description"},
	{subjectToXML[s]},
	{Module[
		{tag, pattr, oattr, data},
		{tag, pattr} = toQName[p];
		{oattr, data} = objectToXML[o];
		XMLElement[
			tag,
			Join[pattr, oattr],
			data
		]
	]}
];

clear[subjectToXML];
subjectToXML[IRI[iri_String]] := {"rdf", "about"} -> CompactIRI[iri, $base];
subjectToXML[RDFBlankNode[id_String]] := {"rdf", "nodeID"} -> id;

clear[toQName];
toQName[IRI[i_String]] := FirstCase[
	$prefixes,
	(pre_String -> ns_String /; StringStartsQ[i, ns]) :> {{pre, StringDrop[i, StringLength[ns]]}, {}},
	With[
		{pre = "ns" <> ToString[$prefixCounter++]},
		{{pre, Last[#]}, {{"http://www.w3.org/2000/xmlns/", pre} -> First[#]}} &[First[StringReplace[i, {
			ns : (Longest[__] ~~ "#" | "/") ~~ local__ :> {ns, local}
		}]]]
	]
];

clear[objectToXML];
objectToXML[s_String] := {{}, {s}};
objectToXML[IRI[iri_String]] := {{}, {XMLElement[{"rdf", "Description"}, {{"rdf", "about"} -> CompactIRI[iri, $base]}, {}]}};
objectToXML[RDFBlankNode[id_String]] := {{}, {XMLElement[{"rdf", "Description"}, {{"rdf", "nodeID"} -> id}, {}]}};
objectToXML[RDFLiteral[l_String, dt_String]] := {{{"rdf", "datatype"} -> CompactIRI[dt, $base]}, {l}};
objectToXML[RDFString[s_String, lang_String]] := {{{"http://www.w3.org/XML/1998/namespace", "lang"} -> lang}, {s}};
objectToXML[x_] := objectToXML[ToRDFLiteral[x] // Replace[_ToRDFLiteral :> fail[]]];

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportRDFXML];
Options[iImportRDFXML] = Options[ImportRDFXML];
iImportRDFXML[file_, OptionsPattern[]] := {"Data" -> Block[{
	$base = ChooseBase[OptionValue["Base"], file]
},
	RDFStore[
		First[Last[Reap[
			If[! DuplicateFreeQ[First[Last[Reap[
				parseDoc[expandDefaultNamespace[Quiet[Import[file, "XML"]] // Replace[_?FailureQ :> fail[]]]];,
				$rdfIDTag
			]], {}]],
				fail[]
			];
		]], {}]
	]
]};

clear[iImportRDFXMLBase];
iImportRDFXMLBase[file_, OptionsPattern[]] := {
	"Base" -> FirstCase[
		Import[file, "XML"],
		XMLElement[_, attr_, _] :> With[
			{b = FirstCase[attr, ({"http://www.w3.org/XML/1998/namespace", "base"} -> base_) :> IRI[base]]},
			b /; ! MissingQ[b]
		],
		None,
		{0, Infinity}
	]
};

clear[iImportRDFXMLPrefixes];
iImportRDFXMLPrefixes[file_, OptionsPattern[]] := {
	"Prefixes" -> <|Flatten[Cases[Import[file, "XML"], XMLElement[_, attr_, _] :> Cases[attr, ({"http://www.w3.org/2000/xmlns/", prefix_} -> ns_) :> prefix -> IRI[ns]], {0, Infinity}]]|>
};


clear[expandDefaultNamespace];
expandDefaultNamespace[expr_] := expr /. e : XMLElement[_, KeyValuePattern[{"http://www.w3.org/2000/xmlns/", "xmlns"} -> ns_], _] :> expandNamespace[e, ns];

clear[expandNamespace];
expandNamespace[expr_, ns_String] := expr /. XMLElement[tag_String, attr_, data_] :> XMLElement[{ns, tag}, attr, expandNamespace[data, ns]];


clear[opt];
opt[x_] := Repeated[x, {0, 1}];

clear[set];
set[x___] := {OrderlessPatternSequence[x]};

clear[setBase];
SetAttributes[setBase, HoldRest];
setBase[XMLElement[_, KeyValuePattern[{"http://www.w3.org/XML/1998/namespace", "base"} -> base_], _], expr_] := Block[
	{$base = base},
	expr
];
setBase[_, expr_] := expr;

clear[toLangString];
toLangString[e_, s_String] := With[
	{lang = elementLanguage[e]},
	If[lang === "",
		s
		,
		RDFString[s, lang]
	]
];


(* 5. Global Issues *)

(* 5.3 Resolving IRIs *)
clear[resolve];
resolve[e_XMLElement, i_String] := ExpandIRI[i, $base];


(* 6. Syntax Data Model *)

(* 6.1.2 Element Event *)
clear[elementAttributes];
elementAttributes[XMLElement[_, attr_List, _]] := elementAttributes[attr];
elementAttributes[attr_List] := attr //
DeleteCases[Alternatives[
	{"http://www.w3.org/XML/1998/namespace" | "http://www.w3.org/2000/xmlns/", _},
	{_, _String?(StringStartsQ["xml", IgnoreCase -> True])},
	_String?(StringStartsQ["xml", IgnoreCase -> True])
] -> _];

clear[elementURI];
elementURI[XMLElement[{ns_String, local_String}, _, _]] := IRI[ns <> local];
elementURI[XMLElement[b_RDFBlankNode, _, _]] := b;

clear[elementLanguage];
elementLanguage[XMLElement[_, KeyValuePattern[{"http://www.w3.org/XML/1998/namespace", "lang"} -> lang_], _]] := lang;
elementLanguage[_XMLElement] := "";


(* 7. RDF/XML Grammar *)

(* 7.2.2 *)
clear[coreSyntaxTermsQ];
coreSyntaxTermsQ[{rdf[], "RDF" | "ID" | "about" | "parseType" | "resource" | "nodeID" | "datatype"}] := True;
coreSyntaxTermsQ[_] := False;

(* 7.2.3 *)
clear[syntaxTermsQ];
syntaxTermsQ[_?coreSyntaxTermsQ | {rdf[], "Description" | "li"}] := True;
syntaxTermsQ[_] := False;

(* 7.2.4 *)
clear[oldTermsQ];
oldTermsQ[{rdf[], "aboutEach" | "aboutEachPrefix" | "bagID"}] := True;
oldTermsQ[_] := False;

(* 7.2.5 *)
clear[nodeElementURIsQ];
nodeElementURIsQ[Except[_?coreSyntaxTermsQ | {rdf[], "li"} | _?oldTermsQ]] := True;
nodeElementURIsQ[_] := False;

(* 7.2.6 *)
clear[propertyElementURIsQ];
propertyElementURIsQ[Except[_?coreSyntaxTermsQ | {rdf[], "Description"} | _?oldTermsQ]] := True;
propertyElementURIsQ[_] := False;

(* 7.2.7 *)
clear[propertyAttributeURIsQ];
propertyAttributeURIsQ[Except[_?coreSyntaxTermsQ | {rdf[], "Description" | "li"} | _?oldTermsQ]] := True;
propertyAttributeURIsQ[_] := False;

(* 7.2.8 *)
clear[parseDoc];
parseDoc[XMLObject["Document"][_, x : XMLElement[{rdf[], "RDF"}, ___], __]] := parseRDF[x];
parseDoc[XMLObject["Document"][_, x_XMLElement, __]] := parseNodeElement[x];

(* 7.2.9 *)
clear[parseRDF];
parseRDF[e : XMLElement[{rdf[], "RDF"}, _, l_List]] := setBase[e, parseNodeElementList[l]];

(* 7.2.10 *)
clear[parseNodeElementList];
parseNodeElementList[l_List] := parseNodeElement /@ l;

(* 7.2.11 *)
clear[parseNodeElement];
parseNodeElement[e : XMLElement[
	tag_?nodeElementURIsQ,
	attr_?(elementAttributes /* MatchQ[set[opt[_?idAttrQ | _?nodeIdAttrQ | _?aboutAttrQ], ___?propertyAttrQ]]),
	l_List
]] := setBase[e, Module[
	{subject},
	subject = attr // Replace[{
		KeyValuePattern[{rdf[], "ID"} -> id_] :> IRI[Sow[resolve[e, "#" <> id], $rdfIDTag]],
		KeyValuePattern[{rdf[], "nodeID"} -> b_] :> RDFBlankNode[b],
		KeyValuePattern[{rdf[], "about"} -> about_] :> IRI[resolve[e, about]],
		_ :> (elementURI[e] // Replace[
			Except[_RDFBlankNode] :> RDFBlankNode[CreateUUID["b-"]]
		])
	}];
	If[! MatchQ[tag, _RDFBlankNode] && tag =!= {rdf[], "Description"},
		Sow[RDFTriple[subject, IRI[rdf["type"]], elementURI[e]]];
	];
	attr // Replace[
		KeyValuePattern[{rdf[], "type"} -> type_] :> Sow[RDFTriple[subject, IRI[rdf["type"]], IRI[resolve[e, type]]]]
	];
	elementAttributes[e] // DeleteCases[{rdf[], "type"} -> _] // Select[propertyAttrQ] // Scan[Function[a,
		Sow[RDFTriple[subject, IRI[StringJoin[First[a]]], toLangString[e, Last[a]]]];
	]];
	Block[{
		$parentSubject = subject,
		$parentLiCounter = 1
	},
		parsePropertyEltList[l];
	];
	subject
]];

(* 7.2.12 *)
(* ws *)

(* 7.2.13 *)
clear[parsePropertyEltList];
parsePropertyEltList[l_List] := parsePropertyElt /@ l;

(* 7.2.14 *)
clear[parsePropertyElt];
parsePropertyElt[eIn : XMLElement[tag_?propertyElementURIsQ, attr_List, data_List]] := setBase[eIn, Module[
	{e},
	e = If[tag === {rdf[], "li"},
		XMLElement[{rdf[], "_" <> ToString[$parentLiCounter++]}, attr, data]
		,
		eIn
	];
	Switch[{elementAttributes[e], data},
		{set[opt[_?idAttrQ]], {_XMLElement}}, parseResourcePropertyElt[e],
		{set[opt[_?idAttrQ], opt[_?datatypeAttrQ]], {_String}}, parseLiteralPropertyElt[e],
		{set[opt[_?idAttrQ], _?parseLiteralQ], {_?literalQ}}, parseParseTypeLiteralPropertyElt[e],
		{set[opt[_?idAttrQ], _?parseResourceQ], _}, parseParseTypeResourcePropertyElt[e],
		{set[opt[_?idAttrQ], _?parseCollectionQ], _}, parseParseTypeCollectionPropertyElt[e],
		{set[opt[_?idAttrQ], _?parseOtherQ], _}, parseParseTypeOtherPropertyElt[e],
		{set[opt[_?idAttrQ], opt[_?resourceAttrQ | _?nodeIdAttrQ | _?datatypeAttrQ], ___?propertyAttrQ], {}}, parseEmptyPropertyElt[e],
		_, fail[]
	];
]];

(* 7.2.15 *)
clear[parseResourcePropertyElt];
parseResourcePropertyElt[e : XMLElement[_, _, {ne_}]] := setBase[e, tryReify[e, Sow[RDFTriple[$parentSubject, elementURI[e], parseNodeElement[ne]]]]];

(* 7.2.16 *)
clear[parseLiteralPropertyElt];
parseLiteralPropertyElt[e : XMLElement[_, attr_, {text_}]] := setBase[e, tryReify[e, Sow[RDFTriple[
	$parentSubject,
	elementURI[e],
	attr // Replace[{
		KeyValuePattern[{rdf[], "datatype"} -> dt_] :> FromRDFLiteral[RDFLiteral[text, dt]],
		_ :> toLangString[e, text]
	}]
]]]];

(* 7.2.17 *)
clear[parseParseTypeLiteralPropertyElt];
parseParseTypeLiteralPropertyElt[e : XMLElement[_, _, {l_}]] := setBase[e, tryReify[e, Sow[RDFTriple[$parentSubject, elementURI[e], l]]]];

(* 7.2.18 *)
clear[parseParseTypeResourcePropertyElt];
parseParseTypeResourcePropertyElt[e : XMLElement[_, _, c_]] := setBase[e, Module[
	{n},
	n = RDFBlankNode[CreateUUID["b-"]];
	tryReify[e, Sow[RDFTriple[$parentSubject, elementURI[e], n]]];
	If[c =!= {},
		parseNodeElement[XMLElement[n, {}, c]];
	];
]];

(* 7.2.19 *)
clear[parseParseTypeCollectionPropertyElt];
parseParseTypeCollectionPropertyElt[e : XMLElement[_, _, l_]] := setBase[e, Module[
	{s},
	s = RDFBlankNode[CreateUUID["b-"]] & /@ l;
	tryReify[e, Sow[RDFTriple[$parentSubject, elementURI[e], First[s, IRI[rdf["nil"]]]]]];
	If[s === {},
		Return[];
	];
	{s, l} // MapThread[Function[{n, f},
		Sow[RDFTriple[n, IRI[rdf["first"]], parseNodeElement[f]]];
	]];
	BlockMap[
		Apply[Function[{n, o},
			Sow[RDFTriple[n, IRI[rdf["rest"]], o]];
		]],
		s,
		2,
		1
	];
	Sow[RDFTriple[Last[s], IRI[rdf["rest"]], IRI[rdf["nil"]]]];
]];

(* 7.2.20 *)
clear[parseParseTypeOtherPropertyElt];
parseParseTypeOtherPropertyElt[XMLElement[tag_, attr_, l_]] := parseParseTypeLiteralPropertyElt[XMLElement[
	tag,
	Replace[attr, _?parseOtherQ :> {rdf[], "parseType"} -> "Literal", {1}],
	l
]];

(* 7.2.21 *)
clear[parseEmptyPropertyElt];
parseEmptyPropertyElt[e : XMLElement[_, attr_, _]] := setBase[e, If[MatchQ[elementAttributes[e], set[opt[_?idAttrQ]]],
	tryReify[e, Sow[RDFTriple[$parentSubject, elementURI[e], toLangString[e, ""]]]];
	,
	Module[
		{r},
		r = attr // Replace[{
			KeyValuePattern[{rdf[], "resource"} -> i_] :> IRI[resolve[e, i]],
			KeyValuePattern[{rdf[], "nodeID"} -> i_] :> RDFBlankNode[i],
			_ :> RDFBlankNode[CreateUUID["b-"]]
		}];
		elementAttributes[e] // Select[propertyAttrQ] // Scan[Function[a,
			Sow[RDFTriple[
				r,
				IRI[StringJoin[First[a]]],
				If[First[a] === {rdf[], "type"},
					resolve[e, IRI[First[a]]],
					toLangString[e, Last[a]]
				]
			]];
		]];
		tryReify[e, Sow[RDFTriple[$parentSubject, elementURI[e], r]]];
	];
]];

(* 7.2.22 *)
clear[idAttrQ];
idAttrQ[{rdf[], "ID"} -> _?rdfidQ] := True;
idAttrQ[_] := False;

(* 7.2.23 *)
clear[nodeIdAttrQ];
nodeIdAttrQ[{rdf[], "nodeID"} -> _?rdfidQ] := True;
nodeIdAttrQ[_] := False;

(* 7.2.24 *)
clear[aboutAttrQ];
aboutAttrQ[{rdf[], "about"} -> _?uRIreferenceQ] := True;
aboutAttrQ[_] := False;

(* 7.2.25 *)
clear[propertyAttrQ];
propertyAttrQ[_?propertyAttributeURIsQ -> _String] := True;
propertyAttrQ[_] := False;

(* 7.2.26 *)
clear[resourceAttrQ];
resourceAttrQ[{rdf[], "resource"} -> _?uRIreferenceQ] := True;
resourceAttrQ[_] := False;

(* 7.2.27 *)
clear[datatypeAttrQ];
datatypeAttrQ[{rdf[], "datatype"} -> _?uRIreferenceQ] := True;
datatypeAttrQ[_] := False;

(* 7.2.28 *)
clear[parseLiteralQ];
parseLiteralQ[{rdf[], "parseType"} -> "Literal"] := True;
parseLiteralQ[_] := False;

(* 7.2.29 *)
clear[parseResourceQ];
parseResourceQ[{rdf[], "parseType"} -> "Resource"] := True;
parseResourceQ[_] := False;

(* 7.2.30 *)
clear[parseCollectionQ];
parseCollectionQ[{rdf[], "parseType"} -> "Collection"] := True;
parseCollectionQ[_] := False;

(* 7.2.31 *)
clear[parseOtherQ];
parseOtherQ[{rdf[], "parseType"} -> Except["Resource" | "Literal" | "Collection", _String]] := True;
parseOtherQ[_] := False;

(* 7.2.32 *)
clear[uRIreferenceQ];
uRIreferenceQ[x_String] := True;

(* 7.2.33 *)
clear[literalQ];
literalQ[_] := True;

(* 7.2.34 *)
clear[rdfidQ];
rdfidQ[x_String] := StringMatchQ[x, NCName];


(* 7.3 Reification Rules *)
clear[tryReify];
tryReify[e : XMLElement[_, KeyValuePattern[{rdf[], "ID"} -> id_], _], statement_] := reify[IRI[Sow[resolve[e, "#" <> id], $rdfIDTag]], statement];
tryReify[_, _] := Null;

clear[reify];
reify[r_, RDFTriple[s_, p_, o_]] := (
	Sow[RDFTriple[r, IRI[rdf["subject"]], s]];
	Sow[RDFTriple[r, IRI[rdf["predicate"]], p]];
	Sow[RDFTriple[r, IRI[rdf["object"]], o]];
	Sow[RDFTriple[r, IRI[rdf["type"]], IRI[rdf["Statement"]]]];
);


(* Namespaces in XML 1.0 (Third Edition) *)
(* https://www.w3.org/TR/xml-names/ *)

(* [4] *)  NCName := Name?(StringFreeQ[":"]);


(* Extensible Markup Language (XML) 1.0 (Fifth Edition) *)
(* https://www.w3.org/TR/xml/ *)

clear[fromCode];
fromCode[a_String] := FromCharacterCode[FromDigits[a, 16]];
fromCode[a_String, b_String] := Alternatives @@ CharacterRange[FromDigits[a, 16], FromDigits[b, 16]];

(* [4] *)  NameStartChar := ":" | LetterCharacter | "_" | fromCode["C0", "D6"] | fromCode["D8", "F6"] | fromCode["F8", "2FF"] |
				fromCode["370", "37D"] | fromCode["37F", "1FFF"] | fromCode["200C", "200D"] | fromCode["2070", "218F"] |
				fromCode["2C00", "2FEF"] | fromCode["3001", "D7FF"] | fromCode["F900", "FDCF"] | fromCode["FDF0", "FFFD"](* | fromCode["10000", "EFFFF"]*);
(* [4a] *) NameChar := NameStartChar | "-" | "." | DigitCharacter | fromCode["B7"] | fromCode["0300", "036F"] | fromCode["203F", "2040"];
(* [5] *)  Name := Name = NameStartChar ~~ NameChar ...;

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
