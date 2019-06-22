Begin["GraphStore`Formats`JSONLD`Private`"];


(* 7. Expansion Algorithms *)

(* 7.1 Expansion Algorithm *)
clear[expandElement];
Options[expandElement] = {
	"Context" -> None
};
expandElement[element_, OptionsPattern[]] := Module[
	{result, activeContext = processContext[<||>, OptionValue["Context"]]},
	result = iExpandElement[activeContext, Null, element];
	Switch[result,
		<|"@graph" -> _|>,
		result = result["@graph"],
		Null,
		result = {}
	];
	listify[result];
	result
];

clear[iExpandElement];
(* 1 *)
iExpandElement[_, _, Null] := Null;
(* 2 *)
iExpandElement[activeContext_, activeProperty_, element_?scalarQ] := If[MatchQ[activeProperty, Null | "@graph"],
	(* 2.1 *)
	Null,
	(* 2.2 *)
	expandValue[activeContext, activeProperty, element]
];
(* 3 *)
iExpandElement[activeContext_, activeProperty_, element_List] := Module[
	{expandedItem},
	reapList[
		element // Scan[Function[item,
			(* 3.2.1 *)
			expandedItem = iExpandElement[activeContext, activeProperty, item];
			(* 3.2.2 *)
			If[activeProperty === "@list" || activeContext[activeProperty, containerMapping] === "@list",
				If[ListQ[expandedItem] || listObjectQ[expandedItem],
					fail["list of lists"]
				];
			];
			(* 3.2.3 *)
			If[ListQ[expandedItem],
				Scan[Sow, expandedItem],
				If[expandedItem =!= Null,
					Sow[expandedItem]
				]
			]
		]];
	]
];
(* 4 *)
iExpandElement[activeContextIn_, activeProperty_, element_?AssociationQ] := Module[
	{result, activeContext, tag, expandedProperty, expandedValue},
	(* 5 *)
	activeContext = If[KeyExistsQ[element, "@context"],
		processContext[activeContextIn, element["@context"]],
		activeContextIn
	];
	(* 6 *)
	result = <||>;
	(* 7 *)
	element // KeySort // KeyValueMap[Function[{key, value}, Catch[
		(* 7.1 *)
		If[key === "@context",
			Throw[Null, tag];
		];
		(* 7.2 *)
		expandedProperty = expandIRI[activeContext, key, "Vocab" ->  True];
		(* 7.3 *)
		If[expandedProperty === Null || (StringQ[expandedProperty] && StringFreeQ[expandedProperty, ":"] && ! keywordQ[expandedProperty]),
			Throw[Null, tag];
		];
		Which[
			(* 7.4 *)
			keywordQ[expandedProperty],
			(* 7.4.1 *)
			If[activeProperty === "@reverse",
				fail["invalid reverse property map"]
			];
			(* 7.4.2 *)
			If[KeyExistsQ[result, expandedProperty],
				fail["colliding keywords"]
			];
			Switch[expandedProperty,
				(* 7.4.3 *)
				"@id",
				If[! StringQ[value],
					fail["invalid @id value"]
				];
				expandedValue = expandIRI[activeContext, value, "DocumentRelative" -> True],
				(* 7.4.4 *)
				"@type",
				If[! MatchQ[value, _String | {___String}],
					fail["invalid type value"]
				];
				expandedValue = Map[
					Function[item,
						expandIRI[activeContext, item, "Vocab" -> True, "DocumentRelative" -> True]
					],
					value,
					{Boole[ListQ[value]]}
				],
				(* 7.4.5 *)
				"@graph",
				expandedValue = iExpandElement[activeContext, "@graph", value],
				(* 7.4.6 *)
				"@value",
				If[! MatchQ[value, _?scalarQ | Null],
					fail["invalid value object value"]
				];
				expandedValue = value;
				If[expandedValue === Null,
					result["@value"] = Null;
					Throw[Null, tag];
				],
				(* 7.4.7 *)
				"@language",
				If[! StringQ[value],
					fail["invalid language-tagged string"]
				];
				expandedValue = ToLowerCase[value],
				(* 7.4.8 *)
				"@index",
				If[! StringQ[value],
					fail["invalid @index value"]
				];
				expandedValue = value,
				(* 7.4.9 *)
				"@list",
				(* 7.4.9.1 *)
				If[MatchQ[activeProperty, Null | "@graph"],
					Throw[Null, tag]
				];
				(* 7.4.9.2 *)
				expandedValue = iExpandElement[activeContext, activeProperty, value];
				(* 7.4.9.3 *)
				If[listObjectQ[expandedValue],
					fail["list of lists"]
				];
				(* ensure that expandedValue is a list *)
				listify[expandedValue],
				(* 7.4.10 *)
				"@set",
				expandedValue = iExpandElement[activeContext, activeProperty, value],
				(* 7.4.11 *)
				"@reverse",
				If[! AssociationQ[value],
					fail["invalid @reverse value"]
				];
				(* 7.4.11.1 *)
				expandedValue = iExpandElement[activeContext, "@reverse", value];
				(* 7.4.11.2 *)
				If[KeyExistsQ[expandedValue, "@reverse"],
					expandedValue["@reverse"] // KeyValueMap[Function[{property, item},
						(* 7.4.11.2.1 *)
						keyInitialize[result, property, {}];
						(* 7.4.11.2.2 *)
						appendOrJoinTo[result[property], item];
					]];
				];
				(* 7.4.11.3 *)
				If[DeleteCases[Keys[expandedValue], "@reverse"] =!= {},
					(* 7.4.11.3.1 *)
					keyInitialize[result, "@reverse", <||>];
					(* 7.4.11.3.2 *)
					(* reverseMap ~= result["@reverse"] *)
					(* 7.4.11.3.3 *)
					expandedValue // KeyDrop["@reverse"] // KeyValueMap[Function[{property, items},
						(* 7.4.11.3.3.1 *)
						items // Scan[Function[item,
							(* 7.4.11.3.3.1.1 *)
							If[valueObjectQ[item] || listObjectQ[item],
								fail["invalid reverse property value"];
							];
							(* 7.4.11.3.3.1.2 *)
							keyInitialize[result["@reverse"], property, {}];
							(* 7.4.11.3.3.1.3 *)
							AppendTo[result["@reverse", property], item];
						]];
					]];
				];
				(* 7.4.11.4 *)
				Throw[Null, tag]
			];
			(* 7.4.12 *)
			If[expandedValue =!= Null,
				result[expandedProperty] = expandedValue;
			];
			(* 7.4.13 *)
			Throw[Null, tag],
			(* 7.5 *)
			activeContext[key, containerMapping] === "@language" && AssociationQ[value],
			(* 7.5.1, 7.5.2 *)
			expandedValue = reapList[
				value // KeySort // KeyValueMap[Function[{language, languageValue},
					(* 7.5.2.1 *)
					Flatten[{languageValue}] //
					(* 7.5.2.2 *)
					Scan[Function[item,
						(* 7.5.2.2.1 *)
						If[! StringQ[item],
							fail["invalid language map value"];
						];
						(* 7.5.2.2.2 *)
						Sow[<|
							"@value" -> item,
							"@language" -> ToLowerCase[language]
						|>];
					]];
				]]
			];,
			(* 7.6 *)
			activeContext[key, containerMapping] === "@index" && AssociationQ[value],
			(* 7.6.1, 7.6.2 *)
			expandedValue = reapList[
				value // KeySort // KeyValueMap[Function[{index, indexValue},
					(* 7.6.2.1 *)
					Flatten[{indexValue}] //
					(* 7.6.2.2 *)
					iExpandElement[activeContext, key, #] & //
					(* 7.6.2.3 *)
					Scan[Function[item,
						Module[
							{i = item},
							(* 7.6.2.3.1 *)
							keyInitialize[i, "@index", index];
							(* 7.6.2.3.2 *)
							Sow[i];
						];
					]];
				]]
			];,
			(* 7.7 *)
			True,
			expandedValue = iExpandElement[activeContext, key, value];
		];
		(* 7.8 *)
		If[expandedValue === Null,
			Throw[Null, tag];
		];
		Which[
			(* 7.9 *)
			activeContext[key, containerMapping] === "@list" && ! listObjectQ[expandedValue],
			listify[expandedValue];
			expandedValue = <|"@list" -> expandedValue|>;,
			(* 7.10 *)
			TrueQ[activeContext[key, reverseProperty]],
			(* 7.10.1 *)
			keyInitialize[result, "@reverse", <||>];
			(* 7.10.2 *)
			(* reverseMap ~= result["@reverse"] *)
			(* 7.10.3 *)
			listify[expandedValue];
			(* 7.10.4 *)
			expandedValue // Scan[Function[item,
				(* 7.10.4.1 *)
				If[valueObjectQ[item] || listObjectQ[item],
					fail["invalid reverse property value"];
				];
				(* 7.10.4.2 *)
				keyInitialize[result["@reverse"], expandedProperty, {}];
				(* 7.10.4.3 *)
				AppendTo[result["@reverse", expandedProperty], item];
			]];
		];
		(* 7.11 *)
		If[! TrueQ[activeContext[key, reverseProperty]],
			(* 7.11.1 *)
			keyInitialize[result, expandedProperty, {}];
			(* 7.11.2 *)
			appendOrJoinTo[result[expandedProperty], expandedValue];
		],
		tag
	]]];
	Which[
		(* 8 *)
		KeyExistsQ[result, "@value"],
		(* 8.1 *)
		If[
			Or[
				! ContainsOnly[Keys[result], {"@value", "@language", "@type", "@index"}],
				ContainsAll[Keys[result], {"@language", "@type"}]
			],
			fail["invalid value object"];
		];
		Which[
			(* 8.2 *)
			result["@value"] === Null,
			result = Null,
			(* 8.3 *)
			! StringQ[result["@value"]] && KeyExistsQ[result, "@language"],
			fail["invalid language-tagged value"],
			(* 8.4 *)
			KeyExistsQ[result, "@type"] && ! StringQ[result["@type"]],
			fail["invalid typed value"]
		],
		(* 9 *)
		KeyExistsQ[result, "@type"],
		listify[result["@type"]],
		(* 10 *)
		KeyExistsQ[result, "@set"] || KeyExistsQ[result, "@list"],
		(* 10.1 *)
		If[! (Length[result] === 1 || (Length[result] === 2 && KeyExistsQ[result, "@index"])),
			fail["invalid set or list object"]
		];
		(* 10.2 *)
		If[KeyExistsQ[result, "@set"],
			result = result["@set"];
		];
	];
	(* 11 *)
	If[MatchQ[result, <|"@language" -> _|>],
		result = Null
	];
	(* 12 *)
	If[MatchQ[activeProperty, Null | "@graph"],
		Which[
			(* 12.1 *)
			MatchQ[result, <||> | KeyValuePattern["@value" | "@list" -> _]],
			result = Null,
			(* 12.2 *)
			MatchQ[result, <|"@id" -> _|>],
			result = Null
		]
	];
	(* 13 *)
	result
];

(* 7.2 Value Expansion *)
clear[expandValue];
expandValue[activeContext_, activeProperty_, value_] := Module[
	{ac, result},
	Switch[activeContext[activeProperty, typeMapping],
		(* 1 *)
		"@id",
		If[StringQ[value],
			ac = activeContext;
			<|"@id" -> expandIRI[ac, value, "DocumentRelative" -> True]|>,
			<|"@value" -> value|>
		],
		(* 2 *)
		"@vocab",
		If[StringQ[value],
			ac = activeContext;
			<|"@id" -> expandIRI[ac, value, "Vocab" -> True, "DocumentRelative" -> True]|>,
			<|"@value" -> value|>
		],
		(* 3 *)
		_,
		result = <|"@value" -> value|>;
		Which[
			(* 4 *)
			! MissingQ[activeContext[activeProperty, typeMapping]],
			result["@type"] = activeContext[activeProperty, typeMapping],
			(* 5 *)
			StringQ[value],
			If[! MissingQ[activeContext[activeProperty, languageMapping]],
				(* 5.1 *)
				activeContext[activeProperty, languageMapping] // Replace[
					lang_String :> (result["@language"] = lang)
				],
				(* 5.2 *)
				activeContext[defaultLanguage] // Replace[
					lang_String :> (result["@language"] = lang)
				]
			]
		];
		(* 6 *)
		result
	]
];


End[];
