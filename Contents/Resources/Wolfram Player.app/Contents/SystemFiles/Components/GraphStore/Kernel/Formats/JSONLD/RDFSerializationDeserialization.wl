Begin["GraphStore`Formats`JSONLD`Private`"];


(* 10. RDF Serialization/Deserialization Algorithms *)

clear /@ {rdf, xsd};
rdf[s_String] := "http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s;
xsd[s_String] := "http://www.w3.org/2001/XMLSchema#" <> s;

clear[datatypeIRI];
datatypeIRI[x_] := DatatypeIRI[x] // Replace[{
	IRI[i_] :> i,
	_ :> fail[datatypeIRI, {x}]
}];

clear[toIRIOrBlank];
toIRIOrBlank[s_String] := If[blankNodeIdentifierQ[s],
	RDFBlankNode[StringDrop[s, 2]],
	IRI[s]
];

(* 10.1 Deserialize JSON-LD to RDF algorithm *)
clear[deserializeJSONLDToRDF];
deserializeJSONLDToRDF[args___] := Block[
	{$counter = 0, $identifierMap = <||>},
	iDeserializeJSONLDToRDF[args]
];

clear[iDeserializeJSONLDToRDF];
Options[iDeserializeJSONLDToRDF] = {
	"Context" -> None
};
iDeserializeJSONLDToRDF[element_, produceGeneralizedRDF : (_?BooleanQ) : False, OptionsPattern[]] := Module[
	{nodeMap, datasetDefault, datasetNamed, tag1, triples, tag2, tag3, listTriples, listHead, sobj, pobj},
	(* 1, 2 *)
	nodeMap = generateNodeMap[
		expandElement[element, "Context" -> OptionValue["Context"]]
	];
	(* 3 *)
	datasetDefault = {};
	datasetNamed = <||>;
	(* 4 *)
	nodeMap // KeySort // KeyValueMap[Function[{graphName, graph}, Catch[
		(* 4.1 *)
		If[relativeIRIQ[graphName],
			Throw[Null, tag1]
		];
		(* 4.2, 4.3 *)
		triples = reapList[
			graph // KeySort // KeyValueMap[Function[{subject, node}, Catch[
				(* 4.3.1 *)
				If[relativeIRIQ[subject],
					Throw[Null, tag2]
				];
				sobj = toIRIOrBlank[subject];
				(* 4.3.2 *)
				node // KeySort // KeyValueMap[Function[{property, values}, Catch[
					Which[
						(* 4.3.2.1 *)
						property === "@type",
						values // Scan[Function[type,
							Sow[RDFTriple[sobj, IRI[rdf["type"]], toIRIOrBlank[type]]]
						]],
						(* 4.3.2.2 *)
						keywordQ[property],
						Throw[Null, tag3],
						(* 4.3.2.3 *)
						blankNodeIdentifierQ[property] && ! produceGeneralizedRDF,
						Throw[Null, tag3],
						(* 4.3.2.4 *)
						relativeIRIQ[property],
						Throw[Null, tag3],
						(* 4.3.2.5 *)
						AbsoluteIRIQ[property] || blankNodeIdentifierQ[property],
						pobj = toIRIOrBlank[property];
						values // Scan[Function[item,
							If[listObjectQ[item],
								(* 4.3.2.5.1 *)
								listTriples = {};
								listHead = convertListToRDF[item["@list"], listTriples];
								Sow[RDFTriple[sobj, pobj, listHead]];
								Scan[Sow, listTriples];
								,
								(* 4.3.2.5.2 *)
								convertObjectToRDF[item] // Replace[
									Except[Null, x_] :> Sow[RDFTriple[sobj, pobj, x]]
								]
							]
						]]
					],
					tag3
				]]];,
				tag2
			]]];
		];
		If[graphName === "@default",
			(* 4.4 *)
			datasetDefault = triples;,
			(* 4.5 *)
			AssociateTo[datasetNamed, toIRIOrBlank[graphName] -> triples];
		];,
		tag1
	]]];
	(* 5 *)
	RDFStore[datasetDefault, datasetNamed]
];

(* 10.2 Object to RDF Conversion *)
clear[convertObjectToRDF];
convertObjectToRDF[<|___, "@id" -> id_, ___|>?nodeObjectQ] := If[relativeIRIQ[id],
	(* 1 *)
	Null,
	(* 2 *)
	toIRIOrBlank[id]
];
convertObjectToRDF[<|"@value" -> x_|>] := x;
convertObjectToRDF[item_] := Module[
	{value, datatype, literal},
	(* 3 *)
	value = item["@value"];
	(* 4 *)
	datatype = Lookup[item, "@type", Null];
	Which[
		(* 5 *)
		MatchQ[value, True | False],
		value = If[value, "true", "false"];
		If[datatype === Null,
			datatype = xsd["boolean"]
		],
		(* 6 *)
		NumberQ[value] && (Mod[value, 1] != 0 || datatype === xsd["double"]),
		(* ExportString[..., "Real64"]? *)
		(* https://bugs.wolfram.com/show?number=349102 *)
		value = If[value === 0,
			"0.0E0",
			Module[
				{m, e},
				{m, e} = MantissaExponent[value];
				m *= 10;
				e--;
				ToString[m] <> "E" <> ToString[e]
			]
		];
		If[datatype === Null,
			datatype = xsd["double"]
		],
		(* 7 *)
		NumberQ[value] && (Mod[value, 1] == 0 || datatype === xsd["integer"]),
		value = ToString[value];
		If[datatype === Null,
			datatype = xsd["integer"]
		],
		(* 8 *)
		datatype === Null,
		datatype = If[KeyExistsQ[item, "@language"],
			rdf["langString"],
			xsd["string"]
		]
	];
	(* return "native" types *)
	If[! StringQ[value],
		Return[value]
	];
	(* 9 *)
	literal = If[KeyExistsQ[item, "@language"],
		RDFString[value, item["@language"]],
		FromRDFLiteral[RDFLiteral[value, datatype]]
	];
	(* 10 *)
	literal
];

(* 10.3 List to RDF Conversion *)
clear[convertListToRDF];
SetAttributes[convertListToRDF, HoldAll];

(* returns listHead, modifies listTriples *)
(* 1 *)
convertListToRDF[{}, ___] := IRI[rdf["nil"]];
convertListToRDF[list_, listTriples_Symbol] := Module[
	{bnodes, object, rest},
	(* 2 *)
	bnodes = RDFBlankNode[StringDrop[generateBlankNodeIdentifier[Null], 2]] & /@ list;
	(* 3, 4 *)
	listTriples = reapList[
		{bnodes, list, Range[Length[bnodes]]} // MapThread[Function[{subject, item, i},
			(* 4.1 *)
			object = convertObjectToRDF[item];
			(* 4.2 *)
			If[object =!= Null,
				Sow[RDFTriple[subject, IRI[rdf["first"]], object]]
			];
			(* 4.3 *)
			rest = If[i < Length[bnodes],
				bnodes[[i + 1]],
				IRI[rdf["nil"]]
			];
			Sow[RDFTriple[subject, IRI[rdf["rest"]], rest]];
		]];
	];
	(* 5 *)
	First[bnodes, IRI[rdf["nil"]]]
];

(* 10.4 Serialize RDF as JSON-LD Algorithm *)

clear[fromIRIOrBlank];
fromIRIOrBlank[IRI[i_String]] := i;
fromIRIOrBlank[RDFBlankNode[b_String]] := "_:" <> b;
fromIRIOrBlank[x_] := x;

clear[serializeRDFToJSONLD];
Options[serializeRDFToJSONLD] = {
	"UseNativeTypes" -> False,
	"UseRDFType" -> False
};
serializeRDFToJSONLD[store_RDFStore, OptionsPattern[]] := Module[
	{graphMap, tag, tag2},
	(* 1, 2 *)
	graphMap = <|"@default" -> <||>|>;
	(* defaultGraph ~= graphMap["@default"] *)
	(* 3 *)
	store["NamedGraphs"] // KeyMap[fromIRIOrBlank] //
	(* 3.1 *)
	Append["@default" -> store["DefaultGraph"]] //
	KeyValueMap[Function[{name, graph},
		Module[
			{value, ss, ps, os},
			(* 3.2 *)
			keyInitialize[graphMap, name, <||>];
			(* 3.3 *)
			If[name =!= "@default",
				keyInitialize[graphMap["@default"], name, <|"@id" -> name|>];
			];
			(* 3.4 *)
			(* nodeMap ~= graphMap[name] *)
			(* 3.5 *)
			Function[{subject, predicate, object}, Catch[
				{ss, ps, os} = fromIRIOrBlank /@ {subject, predicate, object};
				(* 3.5.1 *)
				keyInitialize[graphMap[name], ss, <|"@id" -> ss|>];
				(* 3.5.2 *)
				(* node ~= graphMap[name, ss]; *)
				(* 3.5.3 *)
				If[MatchQ[object, _IRI | _RDFBlankNode],
					keyInitialize[graphMap[name], os, <|"@id" -> os|>];
				];
				(* 3.5.4 *)
				If[ps === rdf["type"] && ! OptionValue["UseRDFType"] && MatchQ[object, _IRI | _RDFBlankNode],
					graphMap[name, ss, "@type"] = appendIfNotMember[graphMap[name, ss, "@type"] // Replace[_Missing :> {}], os];
					Throw[Null, tag]
				];
				(* 3.5.5 *)
				value = convertRDFToObject[object, OptionValue["UseNativeTypes"]];
				(* 3.5.6 *)
				keyInitialize[graphMap[name, ss], ps, {}];
				(* 3.5.7 *)
				graphMap[name, ss, ps] = appendIfNotMember[graphMap[name, ss, ps], value];
				(* 3.5.8 *)
				If[MatchQ[object, _IRI | _RDFBlankNode],
					(* 3.5.8.1 *)
					keyInitialize[graphMap[name, os], "usages", {}];
					(* 3.5.8.2 *)
					(* usages ~= graphMap[name, os, "usages"] *)
					(* 3.5.8.3 *)
					AppendTo[graphMap[name, os, "usages"], <|
						"node" -> {name, ss},
						"property" -> ps,
						"value" -> value
					|>];
				];,
				tag
			]] @@@ DeleteDuplicates[graph];
		];
	]];
	(* 4 *)
	graphMap // Keys // Scan[Function[name, Catch[
		(* graphObject ~= graphMap[name] *)
		(* 4.1 *)
		If[! KeyExistsQ[graphMap[name], rdf["nil"]],
			Throw[Null, tag]
		];
		(* 4.2, 4.3 *)
		graphMap[name, rdf["nil"]] // Lookup[#, "usages", {}] & // Scan[Function[usage, Catch[
			Module[
				{node, property, head, list, listNodes, nodeUsage},
				(* 4.3.1 *)
				node = graphMap[[##]] & @@ usage["node"];
				property = usage["property"];
				head = usage["value"];
				(* 4.3.2 *)
				list = {};
				listNodes = {};
				(* 4.3.3 *)
				While[
					And[
						property === rdf["rest"],
						(* count usages in all graphs *)
						(*Length[node["usages"]] === 1,*)
						(graphMap // Query[Total, Lookup[#, node["@id"], 0] &, Lookup[#, "usages", {}] &, Length]) === 1,
						MatchQ[node[rdf["first"]], {_}],
						MatchQ[node[rdf["rest"]], {_}],
						ContainsOnly[Keys[node], {"usages", rdf["first"], rdf["rest"], "@type", "@id"}],
						Lookup[node, "@type", {rdf["List"]}] === {rdf["List"]}
					],
					(* 4.3.3.1 *)
					AppendTo[list, First[node[rdf["first"]]]];
					(* 4.3.3.2 *)
					AppendTo[listNodes, node["@id"]];
					(* 4.3.3.3 *)
					nodeUsage = First[node["usages"]];
					(* 4.3.3.4 *)
					node = graphMap[[##]] & @@ nodeUsage["node"];
					property = nodeUsage["property"];
					head = nodeUsage["value"];
					(* 4.3.3.5 *)
					If[iriQ[node["@id"]],
						Break[];
					];
				];
				(* 4.3.4 *)
				If[property === rdf["first"],
					(* 4.3.4.1 *)
					If[node["@id"] === rdf["nil"],
						Throw[Null, tag2]
					];
					(* 4.3.4.2 *)
					(* 4.3.4.3, 4.3.4.4 *)
					head = graphMap[name, head["@id"]];
					(* 4.3.4.5 *)
					head = First[head[rdf["rest"]]];
					(* 4.3.4.6 *)
					list = Most[list];
					listNodes = Most[listNodes];
				];
				(* 4.3.5 *)
				(*KeyDropFrom[head, "@id"];*)
				(* 4.3.6 *)
				list = Reverse[list];
				(* 4.3.7 *)
				head["@list"] = list;
				graphMap = graphMap /. <|"@id" -> head["@id"]|> -> KeyDrop[head, "@id"];
				(* 4.3.8 *)
				KeyDropFrom[graphMap[name], listNodes];
			];,
			tag2
		]]];,
		tag
	]]];
	(* 5, 6, 7 *)
	reapList[
		graphMap["@default"] // Keys // Sort // Map[Function[subject,
			Module[
				{node = graphMap["@default", subject]},
				(* 6.1 *)
				If[KeyExistsQ[graphMap, subject],
					(* 6.1.1 *)
					node["@graph"] = {};
					(* 6.1.2 *)
					graphMap[subject] // KeySort // KeyValueMap[Function[{s, n},
						n // KeyDrop["usages"] // Replace[
							Except[<|"@id" -> _|>, x_] :> AppendTo[node["@graph"], x]
						];
					]];
				];
				(* 6.2 *)
				node // KeyDrop["usages"] // Replace[
					Except[<|"@id" -> _|>, x_] :> Sow[x]
				];
			];
		]]
	]
];

(* 10.5 RDF to Object Conversion *)
clear[convertRDFToObject];
convertRDFToObject[value_, useNativeTypes_?BooleanQ] := (
	(* 1 *)
	If[MatchQ[value, _IRI | _RDFBlankNode],
		Return[<|"@id" -> fromIRIOrBlank[value]|>]
	];
	(* 2 *)
	Module[
		{result, convertedValue, type},
		(* 2.1 *)
		result = <||>;
		(* 2.2 *)
		convertedValue = LexicalForm[value];
		(* 2.3 *)
		type = Null;
		Which[
			(* 2.4 *)
			useNativeTypes &&
			Switch[datatypeIRI[value],
				(* 2.4.1 *)
				xsd["string"],
				True,
				(* 2.4.2 *)
				xsd["boolean"],
				Switch[LexicalForm[value],
					"true", convertedValue = True; True,
					"false", convertedValue = False; True,
					_, False
				],
				(* 2.4.3 *)
				xsd["integer"] | xsd["double"],
				If[NumberQ[value],
					convertedValue = value; True,
					False
				],
				_,
				False
			],
			Null,
			(* 2.5 *)
			MatchQ[value, RDFString[_, _]],
			result["@language"] = Last[value];
			convertedValue = First[value],
			(* 2.6 *)
			True,
			datatypeIRI[value] // Replace[Except[xsd["string"], x_] :> (type = x)]
		];
		(* 2.7 *)
		result["@value"] = convertedValue;
		(* 2.8 *)
		If[type =!= Null,
			result["@type"] = type
		];
		(* 2.9 *)
		result
	]
);


End[];
