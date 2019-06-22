Package["Databases`Entity`"]

PackageImport["EntityFramework`"]

PackageImport["Databases`"] (* DatabaseReference, RelationalDatabase*)
PackageImport["Databases`Database`"] (* DatabaseStore *)
PackageImport["Databases`Schema`"] (* DBInspect, DBConnectionValidQ *)
PackageImport["Databases`Common`"]  (* DBError *)
PackageImport["Databases`SQL`"]  (* DBPrefixedField *)

PackageScope["iEntityToDatabaseStore"]
PackageScope[$mappings]

$directFMapping = {
	dbf_DatabaseFunction :> RuleCondition[
		Lookup[$CompiledEntityFunctions, dbf, DBRaise[EntityStore, "no_conversion_known", {dbf}]]
	],
	pf_DBPrefixedField :> RuleCondition[
		toEntityProperty[pf]
	]
}

$reverseFMapping = {
	ef_EntityFunction :> RuleCondition[DBCompileEntityFunction[None, ef]],
	ep_EntityProperty :> RuleCondition[toPrefixedField[ep]]
}

$mappings = {
	<|"Level" -> {{}}, "Keys" -> {"Types" -> "Models"}, "Values" -> {{}, {}}|>,
	<|
		"Level" -> {"Types", All},
		"Keys" -> {
			"Properties" -> "Fields",
			"EntityTypeExtractor" -> "ModelExtractor",
			"CanonicalNameProperties" -> "CanonicalFields"
		},
		"Values" -> {
			{dbq_?DBUncompiledQueryQ :> RuleCondition[
				Lookup[$CompiledEntityQueries, dbq, DBRaise[EntityStore, "no_conversion_known", {dbq}]]
				]},
			{q_?Internal`PossibleEntityListQ :> RuleCondition @ Switch[
				q,
				{}, {},
				{__Entity}, DBRaise[EntityStore, "list_of_entities_cannot_be_aliased", {q}],
				_, DBToDatabaseQuery[q]
			]}
		}
	|>,
	<|
		"Level" -> {"Types", All, "Properties", All},
		"Keys" -> {
			"DestinationEntityType" -> "DestinationModel",
			"EntityTypeMapping" -> "ModelMapping"
		},
		"Values" -> {
			$directFMapping,
			$reverseFMapping
		}
	|>
}

$dbstorehandler = DBCreateMessageHandler[
	Function[code, Quiet[code, DatabaseStore::nopk], HoldAll],
	False,
	False,
	Replace[
		Hold[Message[DatabaseStore::nopk, table_], ___] :>
     		Message[EntityStore::nocanon, table]
	]
];

keyReplace[a_, lor_] := KeyMap[
	Replace[#, lor]&,
	a
]

replaceMappings[assoc_, reverse_:False] := Fold[
	MapAt[
		Function[
			inner,
			keyReplace[
				Replace[
					inner,
					If[reverse, First, Last][#2["Values"]],
					{1}
				],
				If[reverse, Map[Reverse], Identity][#2["Keys"]]
			]
		],
		#1,
		#2["Level"]
	]&,
	assoc,
	MapAt[
		If[
			reverse,
			Replace[#, Join @@ $mappings[[All, "Keys"]], {1}]&,
			Identity
		],
		ReverseSortBy[$mappings, "Level"],
		{All, "Level"}
	]
]

DatabaseStore[EntityStore[meta_, schema : _RelationalDatabase]] := With[{
	res = DBCheckAndSetCache[
		iEntityToDatabaseStore[meta, schema],
		DBHandleError[] @ DatabaseStore[
				If[
					AssociationQ[meta] && KeyExistsQ[meta, "Types"],
					replaceMappings[meta],
					ReplaceAll[
						meta,
						Join @@ $mappings[[All, "Values", -1]]
					]
				],
				schema
		]
	]},
	If[
		FailureQ[res],
		Throw[res, DBError],
		res
	]
]

DatabaseStore[EntityStore[schema : _RelationalDatabase]] := With[{
	res = DBCheckAndSetCache[
		iEntityToDatabaseStore[Automatic, schema],
		DBHandleError[] @ DatabaseStore[schema]
	]},
	If[
		FailureQ[res],
		DBRaise[res],
		res
	]
]


DatabaseStore /: EntityStore[dbs: DatabaseStore[meta_, schema_]] := DBCheckAndSetCache[
	iDatabaseToEntityStore[meta, schema],
	EntityStore[
		replaceMappings[meta, True],
		schema
	]
]


(*                  Generation of the first argument of EntityStore                *)


DatabaseReference /: EntityStore[conn : _DatabaseReference?DBConnectionValidQ] :=
	EntityStore[RelationalDatabase[conn]]

DatabaseReference /:
    EntityStore[tables: $DBTablePattern, conn : _DatabaseReference?DBConnectionValidQ] :=
		EntityStore[RelationalDatabase[tables, conn]]

RelationalDatabase /: es: EntityStore[schema : _RelationalDatabase] /;
	!MatchQ[Internal`CheckCache[iEntityToDatabaseStore[Automatic, schema]], _Failure] := With[{
	res = Catch[
		EntityStore[$dbstorehandler[DatabaseStore[Unevaluated[es]]]],
		DBError
	]},
	res /; !FailureQ[res]
]

RelationalDatabase /: es: EntityStore[meta_, schema : _RelationalDatabase] /; With[
	{cache = Internal`CheckCache[iEntityToDatabaseStore[meta, schema]]},
		And[
			Not[AssociationQ[meta] && KeyExistsQ[meta, "Types"]],
			!MatchQ[cache, _Failure]
		]
	] := 
    With[{res = DBHandleError[] @ EntityStore[
            $dbstorehandler[DatabaseStore[Unevaluated[es]]]
        ]},
        res /; !FailureQ[res]
    ]

RelationalDatabase /: es: EntityStore[meta_, schema : _RelationalDatabase] /;
	MatchQ[Internal`CheckCache[iEntityToDatabaseStore[meta, schema]], _Failure] :=
	(
		Message[EntityStore::invent];
		es /; False
	)

RelationalDatabase /: es: EntityStore[schema : _RelationalDatabase] /;
	MatchQ[Internal`CheckCache[iEntityToDatabaseStore[Automatic, schema]], _Failure] :=
	(
		Message[EntityStore::invent];
		es /; False
	)


RelationalDatabase /: EntityFramework`EntityStoreSubValueHandler[
	es_,
	a_,
	schema: _RelationalDatabase,
	opts___,
	{}
] :=
	Keys[a["Types"]]

RelationalDatabase /: EntityFramework`EntityStoreSubValueHandler[
	es_,
	a_,
	schema : _RelationalDatabase,
	opts___,
	{MakeBoxes, fmt_}
] :=
	makeEntityStoreBoxes[es, fmt]

RelationalDatabase /: EntityFramework`EntityStoreSubValueHandler[
	es_,
	a_,
	schema : _RelationalDatabase,
	opts___,
	{Entity[ent_], "DisableLabeledFormatting"}
] :=
	True


RelationalDatabase /: EntityFramework`EntityStoreSubValueHandler[
	es_,
	a_,
	schema : _RelationalDatabase,
	opts___,
	{args___}
] :=
	entityValue[es][args]

