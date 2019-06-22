Begin["GraphStore`Formats`JSONLD`Private`"];


(* 6. Context Processing Algorithms *)

(* active context: term definitions, base IRI, vocabulary mapping, default language *)
(* term definition: IRI mapping, reverse property, type mapping, language mapping, container mapping *)

(*
activeContext = <|
	baseIRI -> ...,
	vocabularyMapping -> ...,
	defaultLanguage -> ...,

	(* term definitions *)
	"term1" -> <|
		IRIMapping -> ...,
		reverseProperty -> ...,
		typeMapping -> ...,
		languageMapping -> ...,
		containerMapping -> ...
	|>,
	"term2" -> Null,
	"term3" -> ...,
	...
|>
*)

(* 6.1 Context Processing Algorithm *)
clear[processContext];
processContext[activeContext_?AssociationQ, localContext_] := Module[
	{remoteContexts = {}},
	iProcessContext[activeContext, localContext, remoteContexts]
];

clear[iProcessContext];
SetAttributes[iProcessContext, HoldAll];

(* returns a new activeContext, modifies remoteContexts *)
iProcessContext[activeContext_, localContext_, remoteContexts_Symbol] := Module[
	{result, tag, defined},
	(* 1 *)
	result = activeContext;
	(* 2 *)
	Flatten[{localContext}] //
	(* normalize contexts *)
	Map[Replace[{(URL | IRI)[i_] :> i, f_File :> FileToIRI[f]}]] //
	(* 3 *)
	Scan[Function[context, Catch[
		(* 3.1 *)
		If[MatchQ[context, None | Null],
			result = <||>;
			Throw[Null, tag]
		];
		(* 3.2 *)
		If[StringQ[context],
			(* 3.2.1 *)
			(* to do: Not clear what _value_ refers to. *)
			(* 3.2.2 *)
			If[MemberQ[remoteContexts, context],
				fail["recursive context inclusion"],
				AppendTo[remoteContexts, context]
			];
			Module[
				{importedContext},
				(* 3.2.3 *)
				importedContext = importContext[context];
				(* 3.2.4 *)
				result = iProcessContext[result, importedContext, remoteContexts];
			];
			(* 3.2.5 *)
			Throw[Null, tag];
		];
		(* 3.3 *)
		If[! AssociationQ[context],
			fail["invalid local context"]
		];
		(* 3.4 *)
		If[KeyExistsQ[context, "@base"] && remoteContexts === {},
			(* 3.4.1 *)
			Module[
				{value = context["@base"]},
				Which[
					(* 3.4.2 *)
					value === Null,
					(* KeyDropFrom[result, baseIRI] *)
					result[baseIRI] = Null,
					(* 3.4.3 *)
					AbsoluteIRIQ[value],
					result[baseIRI] = value,
					(* 3.4.4 *)
					relativeIRIQ[value] && Lookup[result, baseIRI, $base] =!= Null,
					result[baseIRI] = ExpandIRI[value, Lookup[result, baseIRI, $base]],
					(* 3.4.5 *)
					True,
					fail["invalid base IRI"]
				];
			];
		];
		(* 3.5 *)
		If[KeyExistsQ[context, "@vocab"],
			(* 3.5.1 *)
			Module[
				{value = context["@vocab"]},
				If[value === Null,
					(* 3.5.2 *)
					KeyDropFrom[result, vocabularyMapping],
					(* 3.5.3 *)
					If[AbsoluteIRIQ[value] || blankNodeIdentifierQ[value],
						result[vocabularyMapping] = value,
						fail["invalid vocab mapping"]
					];
				];
			];
		];
		(* 3.6 *)
		If[KeyExistsQ[context, "@language"],
			(* 3.6.1 *)
			Module[
				{value = context["@language"]},
				If[value === Null,
					(* 3.6.2 *)
					KeyDropFrom[result, defaultLanguage],
					(* 3.6.3 *)
					If[StringQ[value],
						result[defaultLanguage] = ToLowerCase[value],
						fail["invalid default language"]
					];
				];
			];
		];
		(* 3.7 *)
		defined = <||>;
		(* 3.8 *)
		context // KeyDrop[{"@base", "@vocab", "@language"}] // KeyValueMap[Function[{key, value},
			createTermDefinition[result, context, key, defined]
		]],
		tag
	]]];
	(* 4 *)
	result
];

(* 6.2 Create Term Definition *)
clear[createTermDefinition];
SetAttributes[createTermDefinition, HoldAll];

(* returns Null, modifies activeContext and defined *)
createTermDefinition[activeContext_Symbol, localContext_, term_, defined_Symbol] := Module[
	{value, definition},
	(* 1 *)
	If[KeyExistsQ[defined, term],
		If[defined[term],
			Return[],
			fail["cyclic IRI mapping"]
		]
	];
	(* 2 *)
	defined[term] = False;
	(* 3 *)
	If[keywordQ[term],
		fail["keyword redefinition"]
	];
	(* 4 *)
	KeyDropFrom[activeContext, term];
	(* 5 *)
	value = localContext[term];
	Which[
		(* 6 *)
		value === Null || (AssociationQ[value] && value["@id"] === Null),
		activeContext[term] = Null;
		defined[term] = True;
		Return[],
		(* 7 *)
		StringQ[value],
		value = <|"@id" -> value|>,
		(* 8 *)
		! AssociationQ[value],
		fail["invalid term definition"]
	];
	(* 9 *)
	definition = <||>;
	(* 10 *)
	If[KeyExistsQ[value, "@type"],
		(* 10.1 *)
		Module[
			{type = value["@type"]},
			If[! StringQ[type],
				fail["invalid type mapping"]
			];
			(* 10.2 *)
			type = expandIRI[activeContext, type, defined, "Vocab" -> True, "LocalContext" -> localContext];
			If[! MatchQ[type, "@id" | "@vocab" | _?AbsoluteIRIQ],
				fail["invalid type mapping"]
			];
			(* 10.3 *)
			definition[typeMapping] = type;
		]
	];
	(* 11 *)
	If[KeyExistsQ[value, "@reverse"],
		(* 11.1 *)
		If[KeyExistsQ[value, "@id"],
			fail["invalid reverse property"]
		];
		(* 11.2 *)
		If[! StringQ[value["@reverse"]],
			fail["invalid IRI mapping"]
		];
		(* 11.3 *)
		definition[IRIMapping] = expandIRI[activeContext, value["@reverse"], defined, "Vocab" -> True, "LocalContext" -> localContext];
		If[! MatchQ[definition[IRIMapping], _?AbsoluteIRIQ | _?blankNodeIdentifierQ],
			fail["invalid IRI mapping"];
		];
		(* 11.4 *)
		If[KeyExistsQ[value, "@container"],
			definition[containerMapping] = value["@container"] // Replace[
				Except["@set" | "@index" | Null] :> fail["invalid reverse property"]
			];
		];
		(* 11.5 *)
		definition[reverseProperty] = True;
		(* 11.6 *)
		activeContext[term] = definition;
		defined[term] = True;
		Return[];
	];
	(* 12 *)
	definition[reverseProperty] = False;
	Which[
		(* 13 *)
		KeyExistsQ[value, "@id"] && value["@id"] =!= term,
		(* 13.1 *)
		If[! StringQ[value["@id"]],
			fail["invalid IRI mapping"]
		];
		(* 13.2 *)
		definition[IRIMapping] = expandIRI[activeContext, value["@id"], defined, "Vocab" -> True, "LocalContext" -> localContext] // Replace[{
			Except[_?keywordQ | _?AbsoluteIRIQ | _?blankNodeIdentifierQ] :> fail["invalid IRI mapping"],
			"@context" :> fail["invalid keyword alias"]
		}],
		(* 14 *)
		StringContainsQ[term, ":"],
		Module[
			{prefix, suffix},
			{prefix, suffix} = StringSplit[term, ":", 2];
			(* 14.1 *)
			If[KeyExistsQ[localContext, prefix],
				createTermDefinition[activeContext, localContext, prefix, defined];
			];
			If[KeyExistsQ[activeContext, prefix],
				(* 14.2 *)
				definition[IRIMapping] = activeContext[prefix, IRIMapping] <> suffix,
				(* 14.3 *)
				definition[IRIMapping] = term
			];
		],
		(* 15 *)
		True,
		If[KeyExistsQ[activeContext, vocabularyMapping],
			definition[IRIMapping] = activeContext[vocabularyMapping] <> term,
			fail["invalid IRI mapping"]
		]
	];
	(* 16 *)
	If[KeyExistsQ[value, "@container"],
		definition[containerMapping] = value["@container"] // Replace[
			Except["@list" | "@set" | "@index" | "@language"] :> fail["invalid container mapping"]
		];
	];
	(* 17 *)
	If[KeyExistsQ[value, "@language"] && ! KeyExistsQ[value, "@type"],
		definition[languageMapping] = value["@language"] // Replace[
			Except[_String | Null] :> fail["invalid language mapping"]
		] // Replace[
			s_String :> ToLowerCase[s]
		];
	];
	(* 18 *)
	activeContext[term] = definition;
	defined[term] = True;
	Null
];

(* 6.3 IRI Expansion *)
clear[expandIRI];
SetAttributes[expandIRI, HoldAll];
Options[expandIRI] = {
	"DocumentRelative" -> False,
	"Vocab" -> False,
	"LocalContext" -> Null
};

(* returns expanded value, modifies activeContext and defined *)
expandIRI[activeContext_Symbol, value_, defined_Symbol : Null, OptionsPattern[]] := (
	(* 1 *)
	If[keywordQ[value] || value === Null,
		Return[value]
	];
	(* 2 *)
	If[OptionValue["LocalContext"] =!= Null && KeyExistsQ[OptionValue["LocalContext"], value] && ! TrueQ[defined[value]],
		createTermDefinition[activeContext, OptionValue["LocalContext"], value, defined]
	];
	(* 3 *)
	If[OptionValue["Vocab"] && KeyExistsQ[activeContext, value],
		Return[activeContext[value, IRIMapping] // Replace[Except[_String] :> Null]]
	];
	(* 4 *)
	If[StringContainsQ[value, ":"],
		Module[{prefix, suffix},
			(* 4.1 *)
			{prefix, suffix} = StringSplit[value, ":", 2];
			(* 4.2 *)
			If[prefix === "_" || StringStartsQ[suffix, "//"],
				Return[value]
			];
			(* 4.3 *)
			If[OptionValue["LocalContext"] =!= Null && KeyExistsQ[OptionValue["LocalContext"], prefix] && ! TrueQ[defined[prefix]],
				createTermDefinition[activeContext, OptionValue["LocalContext"], prefix, defined]
			];
			(* 4.4 *)
			If[KeyExistsQ[activeContext, prefix],
				Return[activeContext[prefix, IRIMapping] <> suffix]
			];
			(* 4.5 *)
			Return[value]
		];
	];
	(* 5 *)
	If[OptionValue["Vocab"] && KeyExistsQ[activeContext, vocabularyMapping],
		Return[activeContext[vocabularyMapping] <> value]
	];
	(* 6 *)
	If[OptionValue["DocumentRelative"],
		Return[ExpandIRI[value, Lookup[activeContext, baseIRI, $base] // Replace[Null -> None]]];
	];
	(* 7 *)
	value
);


End[];
