Begin["GraphStore`Formats`JSONLD`Private`"];


(* 9. Flattening Algorithms *)

(* 9.1 Flattening Algorithm *)
clear[flattenElement];
Options[flattenElement] = {
	"CompactArrays" -> True
};
flattenElement[expandedDocument_List, context_ : Null, OptionsPattern[]] := Block[
	{$counter = 0, $identifierMap = <||>},
	Module[
		{nodeMap, defaultGraph, flattened},
		(* 1, 2 *)
		nodeMap = generateNodeMap[expandedDocument];
		(* 3 *)
		defaultGraph = Lookup[nodeMap, "@default", <||>];
		(* 4 *)
		nodeMap // KeyDrop["@default"] // KeyValueMap[Function[{graphName, graph},
			(* 4.1 *)
			keyInitialize[defaultGraph, graphName, <|"@id" -> graphName|>];
			(* 4.2 *)
			(* entry ~= defaultGraph[graphName] *)
			(* 4.3 *)
			defaultGraph[graphName, "@graph"] = {};
			(* 4.4 *)
			graph // KeySort // KeyValueMap[Function[{id, node},
				If[! MatchQ[node, <|"@id" -> _|>],
					AppendTo[defaultGraph[graphName, "@graph"], node];
				];
			]];
		]];
		(* 5, 6 *)
		flattened = defaultGraph // KeySort // Values // DeleteCases[<|"@id" -> _|>];
		(* 7 *)
		If[MatchQ[context, None | Null],
			Return[flattened]
		];
		(* 8 *)
		compactElement[flattened, context, OptionValue["CompactArrays"]]
	]
];

(* 9.2 Node Map Generation *)
clear[generateNodeMap];
generateNodeMap[expandedDocument_List] := Module[
	{nodeMap, element},
	nodeMap = <||>;
	(* 1 *)
	expandedDocument // Scan[Function[item,
		(* 1.1 *)
		element = item;
		iGenerateNodeMap[element, nodeMap];
	]];
	nodeMap
];

clear[iGenerateNodeMap];
SetAttributes[iGenerateNodeMap, HoldAll];
Options[iGenerateNodeMap] = {
	"ActiveGraph" -> "@default",
	"ActiveProperty" -> Null,
	"ActiveSubject" -> Null
};

(* returns Null, modifies element, nodeMap and list *)
iGenerateNodeMap[element_, nodeMap_Symbol, list_Symbol : Null, OptionsPattern[]] := Module[
	{activeGraph, activeSubject, activeProperty, id, evar, referencedNode},
	activeGraph = OptionValue["ActiveGraph"];
	activeSubject = OptionValue["ActiveSubject"];
	activeProperty = OptionValue["ActiveProperty"];
	(* 1 *)
	If[ListQ[element],
		(* 1.1 *)
		element // Scan[Function[item,
			evar = item;
			iGenerateNodeMap[evar, nodeMap, list, "ActiveGraph" -> activeGraph, "ActiveSubject" -> activeSubject, "ActiveProperty" -> activeProperty];
		]];
		Return[]
	];
	(* 2 *)
	(* AssociationQ[element] *)
	(* graph ~= nodeMap[activeGraph] *)
	(* node ~= nodeMap[activeGraph, activeSubject] *)
	keyInitialize[nodeMap, activeGraph, <||>];
	(* 3 *)
	If[KeyExistsQ[element, "@type"],
		(* 3.1 *)
		element["@type"] = Replace[
			element["@type"],
			item_?blankNodeIdentifierQ :> generateBlankNodeIdentifier[item],
			{1}
		];
	];
	Which[
		(* 4 *)
		KeyExistsQ[element, "@value"],
		If[list === Null,
			(* 4.1 *)
			nodeMap[activeGraph, activeSubject, activeProperty] = appendIfNotMember[
				nodeMap[activeGraph, activeSubject, activeProperty] // Replace[_Missing :> {}],
				element
			],
			(* 4.2 *)
			AppendTo[list["@list"], element]
		],
		(* 5 *)
		KeyExistsQ[element, "@list"],
		Module[
			{result},
			(* 5.1 *)
			result = <|"@list" -> {}|>;
			(* 5.2 *)
			iGenerateNodeMap[element["@list"], nodeMap, result, "ActiveGraph" -> activeGraph, "ActiveSubject" -> activeSubject, "ActiveProperty" -> activeProperty];
			AppendTo[nodeMap[activeGraph, activeSubject, activeProperty], result];
		],
		(* 6 *)
		(* element is a node object *)
		True,
		If[KeyExistsQ[element, "@id"],
			(* 6.1 *)
			id = element["@id"];
			KeyDropFrom[element, "@id"];
			If[blankNodeIdentifierQ[id],
				id = generateBlankNodeIdentifier[id];
			],
			(* 6.2 *)
			id = generateBlankNodeIdentifier[Null];
		];
		(* 6.3 *)
		keyInitialize[nodeMap[activeGraph], id, <|"@id" -> id|>];
		(* 6.4 *)
		(* node ~= nodeMap[activeGraph, id] *)
		Which[
			(* 6.5 *)
			AssociationQ[activeSubject],
			nodeMap[activeGraph, id, activeProperty] = appendIfNotMember[
				nodeMap[activeGraph, id, activeProperty] // Replace[_Missing :> {}],
				activeSubject
			],
			(* 6.6 *)
			activeProperty =!= Null,
			(* 6.6.1 *)
			If[list === Null,
				(* 6.6.2 *)
				nodeMap[activeGraph, activeSubject, activeProperty] = appendIfNotMember[
					nodeMap[activeGraph, activeSubject, activeProperty] // Replace[_Missing :> {}],
					<|"@id" -> id|>
				],
				(* 6.6.3 *)
				AppendTo[list["@list"], <|"@id" -> id|>]
			]
		];
		(* 6.7 *)
		If[KeyExistsQ[element, "@type"],
			nodeMap[activeGraph, id, "@type"] = DeleteDuplicates[Join[
				nodeMap[activeGraph, id, "@type"] // Replace[_Missing :> {}],
				element["@type"]
			]];
			KeyDropFrom[element, "@type"];
		];
		(* 6.8 *)
		If[KeyExistsQ[element, "@index"],
			If[KeyExistsQ[nodeMap[activeGraph, id], "@index"] && nodeMap[activeGraph, id, "@index"] =!= element["@index"],
				fail["conflicting indexes"]
			];
			nodeMap[activeGraph, id, "@index"] = element["@index"];
			KeyDropFrom[element, "@index"];
		];
		(* 6.9 *)
		If[KeyExistsQ[element, "@reverse"],
			(* 6.9.1 *)
			referencedNode = <|"@id" -> id|>;
			(* 6.9.2, 6.9.3 *)
			element["@reverse"] // KeyValueMap[Function[{property, values},
				(* 6.9.3.1 *)
				values // Scan[Function[value,
					Module[
						{v = value},
						iGenerateNodeMap[v, nodeMap, nodeMap, "ActiveSubject" -> referencedNode, "ActiveProperty" -> property];
					]
				]];
			]];
			(* 6.9.4 *)
			KeyDropFrom[element, "@reverse"]
		];
		(* 6.10 *)
		If[KeyExistsQ[element, "@graph"],
			evar = element["@graph"];
			iGenerateNodeMap[evar, nodeMap, "ActiveGraph" -> id];
			KeyDropFrom[element, "@graph"];
		];
		(* 6.11 *)
		element // KeySort //
		(* 6.11.1 *)
		KeyMap[Replace[property_?blankNodeIdentifierQ :> generateBlankNodeIdentifier[property]]] //
		KeyValueMap[Function[{property, value},
			(* 6.11.2 *)
			keyInitialize[nodeMap[activeGraph, id], property, {}];
			(* 6.11.3 *)
			evar = value;
			iGenerateNodeMap[evar, nodeMap, "ActiveGraph" -> activeGraph, "ActiveSubject" -> id, "ActiveProperty" -> property];
		]];
	];
];

(* 9.3 Generate Blank Node Identifier *)

$counter = 0;
$identifierMap = <||>;

clear[generateBlankNodeIdentifier];
generateBlankNodeIdentifier[identifier_] := (
	(* 1 *)
	If[identifier =!= Null && KeyExistsQ[$identifierMap, identifier],
		Return[$identifierMap[identifier]]
	];
	Module[
		{bni},
		(* 2 *)
		bni = "_:b" <> ToString[$counter];
		(* 3 *)
		$counter++;
		(* 4 *)
		If[identifier =!= Null,
			$identifierMap[identifier] = bni;
		];
		(* 5 *)
		bni
	]
);


End[];
