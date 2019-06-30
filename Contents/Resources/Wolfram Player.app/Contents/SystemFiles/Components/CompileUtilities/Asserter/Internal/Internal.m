
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]

Assert[StringQ[$Verb]]
Assert[StringQ[$StrategyName]]


curriedForm[HoldPattern[fun : Function[{args__, last_}, body_]]] :=
	curriedForm[ Function[ {args}, Function[{last}, body]]]

curriedForm[HoldPattern[fun : Function[{arg_}, body_]]] := 
	Function[{ arg}, body]



Get["CompileUtilities`Asserter`Internal`Propositions`"]

handleFailure[msg___] := (
	$FailureHandler[msg];
	False
)

emitMessage[tester_, tag_, template_, params0_] :=
	Module[{params},
		params = Join[
			params0,
			If[StringQ[$SubjectName],
				<| "propositionName" -> "\"" <> ToString[$SubjectName] <> "\"" |>,
				<| "propositionName" -> "<:" <> ToString[params0["actualValue"]] <> ":>" |>
			],
			<|
				"verb" -> $Verb,
				"subject" -> params0["actualValue"],
				"object" -> params0["expectedValue"]
			|>
		];
		params = Map[
			Function[{param},
				Which[
					ReferenceQ[param] || ObjectInstanceQ[param],
						param["toString"],
					VectorQ[param, ReferenceQ] || VectorQ[param, ObjectInstanceQ],
						#["toString"]& /@ param,
					True,
						ToString[param]
				]
			],
			params
		];
		handleFailure[
			$FailureSymbol[tag,
				Join[
					<|
						"MessageTemplate" -> StringTemplate[template], 
			  			"MessageParameters" -> params,
						"ActualValue" -> params0["actualValue"],
						"ExpectedValue" -> params0["expectedValue"]
			  		|>,
			  		If[StringQ[$PropositionDescription],
						<| "Description" -> $PropositionDescription |>,
						<||>
					]
			  	]
			]
		]
	]

badKey[key_, args___] := handleFailure[
	$FailureSymbol["Invalid" <> $StrategyName <> "Key",
		<|
			"MessageTemplate" -> StringTemplate["The `StrategyName` proposition \"`Proposition`\" provided is not known."],
			"MessageParameters" -> <|
				"Proposition" -> key,
				"StrategyName" -> $StrategyName,
				"Arguments" -> {args}
			|>,
			"Proposition" -> key,
			"Arguments" -> {args}
		|>
	]
]

$Dispatcher["describe", description_] := (
	$PropositionDescription = description;
	dispatch
)

$Dispatcher[description_String /; description =!= "describe", arg_] := (
	$PropositionDescription = description;
	dispatch[arg]
)
$Dispatcher[arg_] := (
	ClearAll[$PropositionDescription];
	dispatch[arg]
)

	
dispatch[subject_][key_String, args__:None] := (
	Which[
		!AssociationQ[$Propositions],
			handleFailure[
				$FailureSymbol["Invalid" <> $StrategyName <> "Propositions", <|
					"MessageTemplate" -> StringTemplate["The `StrategyName` propositions table is not valid."], 
			  		"MessageParameters" -> <|
			  			"StrategyName" -> $StrategyName
			  		|>,
					"Propositions" -> $Propositions
			  	|>]
			],
		KeyExistsQ[$Propositions, key],
			$Propositions[key][dispatch][subject][args],
		True,
			badKey[key, args]
	]
)

(*

dispatch[subject_] :=
	Module[{res = $Propositions["fails"][dispatch][subject][None]},
		retObj[res; False]
	]

retObj[any___] := False

*)

Get["CompileUtilities`Asserter`Internal`Formatting`"]
