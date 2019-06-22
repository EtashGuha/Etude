Begin["GraphStore`Formats`JSONLD`Private`"];


(* 8. Compaction Algorithms *)

(* 8.1 Compaction Algorithm *)
clear[compactElement];
compactElement[element_, context_, compactArrays : (_?BooleanQ) : True] := Module[
	{result, activeContext, inverseContext},
	activeContext = processContext[<||>, context];
	inverseContext = createInverseContext[activeContext];
	result = iCompactElement[
		activeContext,
		inverseContext,
		Null,
		expandElement[element],
		compactArrays
	];
	If[ListQ[result],
		result = If[result === {},
			<||>,
			<|
				compactIRI[
					activeContext,
					inverseContext,
					"@graph",
					"Vocab" -> True
				] -> result
			|>
		];
	];
	(* embed blank node objects that are used once *)
	result = Fold[
		Function[{r, i},
			With[{pos = Position[r, Except[i, KeyValuePattern[Normal[i]]], Infinity, 2]},
				If[Length[pos] === 1,
					r // ReplaceAll[i :> With[{tmp = KeyDrop[First[Extract[r, pos]], Keys[i]]}, tmp /; True]] // Delete[pos],
					r
				]
			]
		],
		result,
		Cases[result, <|"@id" -> _String?(StringStartsQ["_:"])|>, Infinity] // Counts // Select[EqualTo[1]] // Keys
	];
	If[! MatchQ[context, None | Null],
		If[MatchQ[context, _Association | {___Association}], context, importContext[context]] // Replace[
			c : <|__|> | {__} :> PrependTo[result, "@context" -> c]
		];
	];
	result
];

clear[iCompactElement];
(* 1 *)
iCompactElement[_, _, _, element_?scalarQ, ___] := element;
(* 2 *)
iCompactElement[activeContext_, inverseContext_, activeProperty_, element_List, compactArrays : (_?BooleanQ) : True] := Module[
	{result},
	(* 2.1, 2.2 *)
	result = reapList[
		element // Scan[Function[item,
			iCompactElement[activeContext, inverseContext, activeProperty, item, compactArrays] // Replace[
				Except[Null, x_] :> Sow[x]
			];
		]];
	];
	(* 2.3 *)
	If[Length[result] === 1 && ! KeyExistsQ[Lookup[activeContext, activeProperty, <||>], containerMapping] && compactArrays,
		result = First[result];
	];
	(* 2.4 *)
	result
];
(* 3 *)
iCompactElement[activeContext_, inverseContext_, activeProperty_, element_?AssociationQ, compactArrays : (_?BooleanQ) : True] := Module[
	{result, insideReverse, compactedValue, alias, tag, itemActiveProperty, container, compactedItem, mapKey},
	(* 4 *)
	If[KeyMemberQ[element, "@value" | "@id"],
		result = compactValue[activeContext, inverseContext, activeProperty, element];
		If[scalarQ[result],
			Return[result]
		];
	];
	(* 5 *)
	insideReverse = activeProperty === "@reverse";
	(* 6 *)
	result = <||>;
	(* 7 *)
	element // KeySort // KeyValueMap[Function[{expandedProperty, expandedValue}, Catch[
		(* 7.1 *)
		If[MatchQ[expandedProperty, "@id" | "@type"],
			If[StringQ[expandedValue],
				(* 7.1.1 *)
				compactedValue = compactIRI[activeContext, inverseContext, expandedValue, "Vocab" -> expandedProperty === "@type"],
				(* 7.1.2 *)
				compactedValue = reapList[
					expandedValue // Scan[Function[expandedType,
						Sow[compactIRI[activeContext, inverseContext, expandedType, "Vocab" -> True]]
					]]
				];
				(* 7.1.2.3 *)
				If[Length[compactedValue] === 1,
					compactedValue = First[compactedValue];
				];
			];
			(* 7.1.3 *)
			alias = compactIRI[activeContext, inverseContext, expandedProperty, "Vocab" -> True];
			(* 7.1.4 *)
			result[alias] = compactedValue;
			Throw[Null, tag];
		];
		(* 7.2 *)
		If[expandedProperty === "@reverse",
			(* 7.2.1 *)
			compactedValue = iCompactElement[activeContext, inverseContext, "@reverse", expandedValue];
			(* 7.2.2 *)
			compactedValue // Keys // Map[Function[property,
				(* value ~= compactedValue[property] *)
				(* 7.2.2.1 *)
				If[TrueQ[activeContext[property, reverseProperty]],
					(* 7.2.2.1.1 *)
					If[activeContext[property, containerMapping] === "@set" || ! compactArrays,
						listify[compactedValue[property]];
					];
					If[! KeyExistsQ[result, property],
						(* 7.2.2.1.2 *)
						result[property] = compactedValue[property],
						(* 7.2.2.1.3 *)
						listify[result[property]];
						appendOrJoinTo[result[property], compactedValue[property]];
					];
					(* 7.2.2.1.4 *)
					KeyDropFrom[compactedValue, property];
				];
			]];
			(* 7.2.3 *)
			If[compactedValue =!= <||>,
				AssociateTo[
					result,
					compactIRI[activeContext, inverseContext, "@reverse", "Vocab" -> True] -> compactedValue
				];
			];
			(* 7.2.4 *)
			Throw[Null, tag];
		];
		(* 7.3 *)
		If[expandedProperty === "@index" && activeContext[activeProperty, containerMapping] === "@index",
			Throw[Null, tag];
		];
		(* 7.4 *)
		If[MatchQ[expandedProperty, "@index" | "@value" | "@language"],
			(* 7.4.1 *)
			alias = compactIRI[activeContext, inverseContext, expandedProperty, "Vocab" -> True];
			(* 7.4.2 *)
			result[alias] = expandedValue;
			Throw[Null, tag];
		];
		(* 7.5 *)
		If[expandedValue === {},
			(* 7.5.1 *)
			itemActiveProperty = compactIRI[activeContext, inverseContext, expandedProperty, "Value" -> expandedValue, "Vocab" -> True, "Reverse" -> insideReverse];
			(* 7.5.2 *)
			If[! KeyExistsQ[result, itemActiveProperty],
				result[itemActiveProperty] = {},
				listify[result[itemActiveProperty]]
			];
		];
		(* 7.6 *)
		(* ListQ[expandedValue] *)
		expandedValue //
		(* "@reverse" -> JSON object *)
		Replace[a_Association :> {a}] //
		Scan[Function[expandedItem,
			(* 7.6.1 *)
			itemActiveProperty = compactIRI[activeContext, inverseContext, expandedProperty, "Value" -> expandedItem, "Vocab" -> True, "Reverse" -> insideReverse];
			(* 7.6.2 *)
			container = activeContext[itemActiveProperty, containerMapping] // Replace[_Missing :> Null];
			(* 7.6.3 *)
			compactedItem = iCompactElement[activeContext, inverseContext, itemActiveProperty, Lookup[expandedItem, "@list", expandedItem]];
			(* 7.6.4 *)
			If[listObjectQ[expandedItem],
				(* 7.6.4.1 *)
				listify[compactedItem];
				If[container =!= "@list",
					(* 7.6.4.2 *)
					(* 7.6.4.2.1 *)
					compactedItem = <|
						compactIRI[activeContext, inverseContext, "@list", "Value" -> compactedItem, "Vocab" -> True] -> compactedItem
					|>;
					(* 7.6.4.2.2 *)
					If[KeyExistsQ[expandedItem, "@index"],
						AssociateTo[
							compactedItem,
							compactIRI[activeContext, inverseContext, "@index", "Value" -> expandedItem["@index"], "Vocab" -> True] -> expandedItem["@index"]
						];
					],
					(* 7.6.4.3 *)
					If[KeyExistsQ[result, itemActiveProperty],
						fail["compaction to list of lists"];
					];
				];
			];
			If[MatchQ[container, "@language" | "@index"],
				(* 7.6.5 *)
				(* 7.6.5.1 *)
				keyInitialize[result, itemActiveProperty, <||>];
				(* mapObject ~= result[itemActiveProperty] *)
				(* 7.6.5.2 *)
				If[container === "@language" && AssociationQ[compactedItem] && KeyExistsQ[compactedItem, "@value"],
					compactedItem = compactedItem["@value"];
				];
				(* 7.6.5.3 *)
				mapKey = expandedItem[container];
				(* 7.6.5.4 *)
				If[! KeyExistsQ[result[itemActiveProperty], mapKey],
					result[itemActiveProperty, mapKey] = compactedItem,
					listify[result[itemActiveProperty, mapKey]];
					AppendTo[result[itemActiveProperty, mapKey], compactedItem];
				],
				(* 7.6.6 *)
				(* 7.6.6.1 *)
				If[! compactArrays || MatchQ[container, "@set" | "@list"] || MatchQ[expandedProperty, "@list" | "@graph"],
					listify[compactedItem];
				];
				If[! KeyExistsQ[result, itemActiveProperty],
					(* 7.6.6.2 *)
					result[itemActiveProperty] = compactedItem,
					(* 7.6.6.3 *)
					listify[result[itemActiveProperty]];
					appendOrJoinTo[result[itemActiveProperty], compactedItem];
				];
			];
		]];,
		tag
	]]];
	(* 8 *)
	result
];

(* 8.2 Inverse Context Creation *)
clear[createInverseContext];
createInverseContext[activeContext_?AssociationQ] := Module[
	{result, defLang, tag, container, iri, language},
	(* 1 *)
	result = <||>;
	(* 2 *)
	defLang = Lookup[activeContext, defaultLanguage, "@none"];
	(* 3 *)
	activeContext // KeySelect[StringQ] // KeySortBy[StringLength] // KeyValueMap[Function[{term, termDefinition}, Catch[
		(* 3.1 *)
		If[termDefinition === Null,
			Throw[Null, tag]
		];
		(* 3.2 *)
		container = Lookup[termDefinition, containerMapping, "@none"];
		(* 3.3 *)
		iri = termDefinition[IRIMapping];
		(* 3.4 *)
		keyInitialize[result, iri, <||>];
		(* 3.5 *)
		(* containerMap ~= result[iri] *)
		(* 3.6 *)
		keyInitialize[result[iri], container, <|
			"@language" -> <||>,
			"@type" -> <||>,
			(* "@any": introduced in "JSON-LD 1.1 Processing Algorithms and API", retrieved 2018-03-23 *)
			"@any" -> <|"@none" -> term|>
		|>];
		(* 3.7 *)
		(* typeLanguageMap ~= result[iri, container] *)
		Which[
			(* 3.8 *)
			TrueQ[termDefinition[reverseProperty]],
			keyInitialize[result[iri, container, "@type"], "@reverse", term],
			(* 3.9 *)
			KeyExistsQ[termDefinition, typeMapping],
			keyInitialize[result[iri, container, "@type"], termDefinition[typeMapping], term],
			(* 3.10 *)
			KeyExistsQ[termDefinition, languageMapping],
			(* 3.10.1 *)
			(* languageMap ~= result[iri, container, "@language"] *)
			(* 3.10.2 *)
			language = termDefinition[languageMapping] // Replace[Null :> "@null"];
			(* 3.10.3 *)
			keyInitialize[result[iri, container, "@language"], language, term],
			(* 3.11 *)
			True,
			(* 3.11.1 *)
			(* languageMap ~= result[iri, container, "@language"] *)
			(* 3.11.2 *)
			keyInitialize[result[iri, container, "@language"], defLang, term];
			(* 3.11.3 *)
			keyInitialize[result[iri, container, "@language"], "@none", term];
			(* 3.11.4 *)
			(* typeMap ~= result[iri, container, "@type"] *)
			(* 3.11.5 *)
			keyInitialize[result[iri, container, "@type"], "@none", term];
		];
		,
		tag
	]]];
	(* 4 *)
	result
];

(* 8.3 IRI Compaction *)
clear[compactIRI];
Options[compactIRI] = {
	"Reverse" -> False,
	"Value" -> Null,
	"Vocab" -> False
};
compactIRI[activeContext_?AssociationQ, inverseContext_?AssociationQ, iri_, OptionsPattern[]] := Module[
	{value, vocab, reverse, defLang, containers, typeLanguage, typeLanguageValue, list, commonType, commonLanguage, itemType, itemLanguage, tag, preferredValues, compIRI, candidate},
	value = OptionValue["Value"];
	vocab = OptionValue["Vocab"];
	reverse = OptionValue["Reverse"];
	(* 1 *)
	If[iri === Null,
		Return[Null]
	];
	(* 2 *)
	If[vocab && KeyExistsQ[inverseContext, iri],
		(* 2.1 *)
		defLang = Lookup[activeContext, defaultLanguage, "@none"];
		(* 2.2 *)
		containers = {};
		(* 2.3 *)
		typeLanguage = "@language";
		typeLanguageValue = "@null";
		(* 2.4 *)
		If[AssociationQ[value] && KeyExistsQ[value, "@index"],
			AppendTo[containers, "@index"]
		];
		Which[
			(* 2.5 *)
			reverse,
			typeLanguage = "@type";
			typeLanguageValue = "@reverse";
			AppendTo[containers, "@set"],
			(* 2.6 *)
			listObjectQ[value],
			(* 2.6.1 *)
			If[! KeyExistsQ[value, "@index"],
				AppendTo[containers, "@list"]
			];
			(* 2.6.2 *)
			list = value["@list"];
			(* 2.6.3 *)
			commonType = Null;
			commonLanguage = Null;
			If[list === {},
				commonLanguage = defLang;
			];
			(* 2.6.4 *)
			Catch[list // Scan[Function[item,
				(* 2.6.4.1 *)
				itemLanguage = "@none";
				itemType = "@none";
				If[KeyExistsQ[item, "@value"],
					(* 2.6.4.2 *)
					Which[
						(* 2.6.4.2.1 *)
						KeyExistsQ[item, "@language"],
						itemLanguage = item["@language"],
						(* 2.6.4.2.2 *)
						KeyExistsQ[item, "@type"],
						itemType = item["@type"],
						(* 2.6.4.2.3 *)
						True,
						itemLanguage = "@null"
					],
					(* 2.6.4.3 *)
					itemType = "@id"
				];
				If[commonLanguage === Null,
					(* 2.6.4.4 *)
					commonLanguage = itemLanguage,
					(* 2.6.4.5 *)
					If[itemLanguage =!= commonLanguage && KeyExistsQ[item, "@value"],
						commonLanguage = "@none";
					]
				];
				If[commonType === Null,
					(* 2.6.4.6 *)
					commonType = itemType,
					(* 2.6.4.7 *)
					If[itemType =!= commonType,
						commonType = "@none";
					]
				];
				(* 2.6.4.8 *)
				If[commonLanguage === "@none" && commonType === "@none",
					Throw[Null, tag];
				]
			]], tag];
			(* 2.6.5 *)
			If[commonLanguage === Null,
				commonLanguage = "@none";
			];
			(* 2.6.6 *)
			If[commonType === Null,
				commonType = "@none";
			];
			If[commonType =!= "@none",
				(* 2.6.7 *)
				typeLanguage = "@type";
				typeLanguageValue = commonType,
				(* 2.6.8 *)
				typeLanguageValue = commonLanguage
			];
			,
			(* 2.7 *)
			True,
			If[valueObjectQ[value],
				(* 2.7.1 *)
				Which[
					(* 2.7.1.1 *)
					KeyExistsQ[value, "@language"] && ! KeyExistsQ[value, "@index"],
					typeLanguageValue = value["@language"];
					AppendTo[containers, "@language"],
					(* 2.7.1.2 *)
					KeyExistsQ[value, "@type"],
					typeLanguageValue = value["@type"];
					typeLanguage = "@type";
				],
				(* 2.7.2 *)
				typeLanguage = "@type";
				typeLanguageValue = "@id";
			];
			(* 2.7.3 *)
			AppendTo[containers, "@set"];
		];
		(* 2.8 *)
		AppendTo[containers, "@none"];
		(* 2.9 *)
		If[typeLanguageValue === Null,
			typeLanguageValue = "@null";
		];
		(* 2.10 *)
		preferredValues = {};
		(* 2.11 *)
		If[typeLanguageValue === "@reverse",
			AppendTo[preferredValues, "@reverse"];
		];
		If[MatchQ[typeLanguageValue, "@id" | "@reverse"] && AssociationQ[value] && KeyExistsQ[value, "@id"],
			(* 2.12 *)
			If[activeContext[compactIRI[activeContext, inverseContext, value["@id"], "Vocab" -> True(*", DocumentRelative" -> True*)], IRIMapping] === value["@id"],
				(* 2.12.1 *)
				preferredValues = Join[preferredValues, {"@vocab", "@id", "@none"}],
				(* 2.12.2 *)
				preferredValues = Join[preferredValues, {"@id", "@vocab", "@none"}]
			],
			(* 2.13 *)
			preferredValues = Join[preferredValues, {typeLanguageValue, "@none"}];
			If[listObjectQ[value] && value["@list"] === {},
				typeLanguage = "@any";
			];
		];
		Module[
			{term},
			(* 2.14 *)
			term = selectTerm[inverseContext, iri, containers, typeLanguage, preferredValues];
			(* 2.15 *)
			If[term =!= Null,
				Return[term];
			];
		];
	];
	(* 3 *)
	If[vocab && KeyExistsQ[activeContext, vocabularyMapping],
		(* 3.1 *)
		If[StringStartsQ[iri, activeContext[vocabularyMapping]] && StringLength[iri] > StringLength[activeContext[vocabularyMapping]],
			Module[
				{suffix},
				suffix = StringDrop[iri, StringLength[activeContext[vocabularyMapping]]];
				If[! KeyExistsQ[activeContext, suffix],
					Return[suffix];
				];
			];
		];
	];
	(* 4 *)
	compIRI = Null;
	(* 5 *)
	activeContext // KeySelect[StringQ] // KeyValueMap[Function[{term, termDefinition}, Catch[
		(* 5.1 *)
		If[StringContainsQ[term, ":"],
			Throw[Null, tag];
		];
		(* 5.2 *)
		If[termDefinition === Null || termDefinition[IRIMapping] === iri || ! StringStartsQ[iri, termDefinition[IRIMapping]],
			Throw[Null, tag];
		];
		(* 5.3 *)
		candidate = term <> ":" <> StringDrop[iri, StringLength[termDefinition[IRIMapping]]];
		(* 5.4 *)
		If[
			And[
				compIRI === Null || (StringLength[candidate] < StringLength[compIRI] || (StringLength[candidate] == compIRI && OrderedQ[{candidate, compIRI}])),
				! KeyExistsQ[activeContext, candidate] || (activeContext[candidate, IRIMapping] === iri && value === Null)
			],
			compIRI = candidate;
		];,
		tag
	]]];
	(* 6 *)
	If[compIRI =!= Null,
		Return[compIRI];
	];
	(* 7 *)
	If[! vocab,
		Return[CompactIRI[iri, Lookup[activeContext, baseIRI, $base]]]
	];
	(* 8 *)
	iri
];

(* 8.4 Term Selection *)
clear[selectTerm];
selectTerm[inverseContext_?AssociationQ, iri_String, containers_List, typeLanguage_String, preferredValues_List] := Module[
	{containerMap, tag1, typeLanguageMap, valueMap, tag2},
	(* 1 *)
	containerMap = inverseContext[iri];
	(* 2 *)
	containers // Scan[Function[container, Catch[
		(* 2.1 *)
		If[! KeyExistsQ[containerMap, container],
			Throw[Null, tag1]
		];
		(* 2.2 *)
		typeLanguageMap = containerMap[container];
		(* 2.3 *)
		valueMap = typeLanguageMap[typeLanguage];
		(* 2.4 *)
		preferredValues // Scan[Function[item, Catch[
			If[! KeyExistsQ[valueMap, item],
				(* 2.4.1 *)
				Throw[Null, tag2],
				(* 2.4.2 *)
				Return[valueMap[item], Module]
			];
			,
			tag2
		]]];,
		tag1
	]]];
	(* 3 *)
	Null
];

(* 8.5 Value Compaction *)
clear[compactValue];
compactValue[activeContext_, inverseContext_, activeProperty_, value_?AssociationQ] := Module[
	{numberMembers},
	(* 1 *)
	numberMembers = Length[value];
	(* 2 *)
	If[KeyExistsQ[value, "@index"] && activeContext[activeProperty, containerMapping] === "@index",
		numberMembers -= 1;
	];
	(* 3 *)
	If[numberMembers > 2,
		Return[value];
	];
	(* 4 *)
	If[KeyExistsQ[value, "@id"],
		Which[
			(* 4.1 *)
			numberMembers === 1 && activeContext[activeProperty, typeMapping] === "@id",
			Return[compactIRI[activeContext, inverseContext, value["@id"]]],
			(* 4.2 *)
			numberMembers === 1 && activeContext[activeProperty, typeMapping] === "@vocab",
			Return[compactIRI[activeContext, inverseContext, value["@id"], "Vocab" -> True]],
			(* 4.3 *)
			True,
			Return[value]
		];
	];
	(* 5 *)
	If[KeyExistsQ[value, "@type"] && value["@type"] === activeContext[activeProperty, typeMapping],
		Return[value["@value"]]
	];
	(* 6 *)
	If[KeyExistsQ[value, "@language"] && (value["@language"] === activeContext[activeProperty, languageMapping] || value["@language"] === activeContext[defaultLanguage]),
		Return[value["@value"]]
	];
	(* 7 *)
	If[numberMembers === 1 && Or[
		! StringQ[value["@value"]],
		! KeyExistsQ[activeContext, defaultLanguage],
		activeContext[activeProperty, languageMapping] === Null
	],
		Return[value["@value"]]
	];
	(* 8 *)
	value
];


End[];
