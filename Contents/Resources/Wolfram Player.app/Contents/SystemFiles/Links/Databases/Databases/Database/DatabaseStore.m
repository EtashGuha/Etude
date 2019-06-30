Package["Databases`Database`"]

PackageImport["Databases`"] (* DatabaseReference, SQLDatabaseObject *)
PackageImport["Databases`Common`"] (* DBRaise, DBError *)
PackageImport["Databases`Schema`"] (* DBInspect *)
PackageImport["Databases`SQL`"] (* DBGetPrefixedSelectedFields, DBPrimaryKey *)
PackageImport["EntityFramework`"]


addIndex[assoc: <|(_ -> _?StringQ)...|>] := AssociationThread[
	Keys[assoc],
	SortBy[
		Join @@ Map[
			Function[
				same,
				MapIndexed[
					{First[#1] <> "_" <> ToString[First[#2]], Last[#1]} &,
					same
				]
			],
			GroupBy[MapIndexed[List, Values[assoc]], First]
		],
		Last
	][[All, 1]]
]

fixPrefixedFields[pfs: {__DBPrefixedField}] := Module[{parts, uniquenames, res},
	parts = AssociationMap[Last @* DBPrefixedFieldParts, pfs];
	uniquenames = DeleteDuplicates[parts];
	parts = addIndex[KeyComplement[{parts, uniquenames}]];
	Normal @ Join[DBKeyValueReverse[uniquenames], DBKeyValueReverse[parts]]
]


canonicalFieldsRename[fieldsAssoc_, canonicalField_?StringQ] := FirstPosition[
	fieldsAssoc,
	a_?AssociationQ /; a["ColumnName"] === canonicalField
][[1, 1]]

canonicalFieldsRename[fieldsAssoc_, canonicalField_DBPrefixedField] := FirstPosition[
	fieldsAssoc,
	a_?AssociationQ /; DBPrefixedField[a["ColumnPrefix"], a["ColumnName"]] === canonicalField
][[1, 1]]

canonicalFieldsRename[fieldsAssoc_, canonicalFields_List] := Map[
	canonicalFieldsRename[fieldsAssoc, #]&,
	canonicalFields
]


canonicalFieldsMessage[table_][{} | None] := (Message[DatabaseStore::nopk, table]; {})

canonicalFieldsMessage[_][any_] := any

extractHighLevelInfo["Tables", tables: {(_?StringQ -> _)...}, schema_] :=
	Association @ Replace[
		tables,
		{
			(alias_?StringQ -> s_?StringQ) :> alias -> <|
				"ModelExtractor" -> If[
					schema[s, "Exists"],
					s,
					DBRaise[extractHighLevelInfo, "missing_table", {s}]
				],
				"Fields" -> extractHighLevelInfo["Columns", s, schema[s, "Columns"], schema],
				"CanonicalFields" -> canonicalFieldsMessage[s] @ schema[s, "PrimaryKey", "Columns"]
			|>,
			(s_?StringQ -> l_List) :> With[
					{fields = extractHighLevelInfo["Columns", s, l, schema]},
					s -> <|
					"ModelExtractor" -> If[
						schema[s, "Exists"],
						s,
						DBRaise[extractHighLevelInfo, "missing_table", {s}]
					],
					"Fields" -> fields,
					"CanonicalFields" -> canonicalFieldsRename[
						fields,
						canonicalFieldsMessage[s] @ schema[s, "PrimaryKey", "Columns"]
					]
				|>
			],
			(alias_?StringQ -> dbq_?DBUncompiledQueryQ) :> With[
				{qbo = DBQueryToQueryBuilderObject[dbq, DatabaseStore[schema, "AddRelations" -> False]]},
				With[
					{fixed = fixPrefixedFields[
						Cases[
							qbo @ DBGetPrefixedSelectedFields[],
							Apply[Alternatives] @ qbo @ DBGetFieldsWithDependencies[True]
							(* DBGetFieldsWithDependencies can never return rules because
							   the DatabaseStore is trivial *)
						]
					]},
					With[{fields = extractHighLevelInfo[
						"Columns",
						dbq,
						fixed,
						qbo
					]},
					alias -> <|
						"ModelExtractor" -> dbq,
						"Fields" -> fields,
						"CanonicalFields" -> canonicalFieldsRename[
							fields,
							canonicalFieldsMessage[s] @ qbo @ DBPrimaryKey[]
						]
					|>]
				]
			]
		},
		{1}
	]

extractHighLevelInfo["Tables", tableNames_, schema_] :=
	extractHighLevelInfo[
		"Tables",
		Replace[
			tableNames,
			{
				s_?StringQ :> s -> s,
				r: (Rule | RuleDelayed)[_?StringQ, _] :> Rule @@ r,
				any_ :> DBRaise[extractHighLevelInfo, "invalid_model_spec", {any}]
			},
			{1}
		],
		schema
	]

extractHighLevelInfo["Columns", tableName_, columnNames: {(_?StringQ -> _)..}, schema_] := 	Association @ MapAt[
	Replace[
		Replace[
			#,
			{
				s_?StringQ :> If[
					MatchQ[schema, $DBStrictSchemaPattern] && schema[tableName, s, "Exists"],
					DBPrefixedField[tableName -> s],
					DBRaise[extractHighLevelInfo, "missing_column", {tableName, s}]
				],
				pf_DBPrefixedField :> pf,
				dbf_DatabaseFunction :> dbf,
				any_ :> DBRaise[extractHighLevelInfo, "invalid_field_spec", {any}]
			}
		],
		{
			f_DBPrefixedField :> With[{parts = DBPrefixedFieldParts[f]},
				<|
					"ColumnPrefix" -> First[parts],
					"ColumnName" -> Last[parts]
				|>
			],
			func_DatabaseFunction :> <|"Function" -> func|>
		}
	]&,
	columnNames,
	{All, 2}
]

extractHighLevelInfo["Columns", tableName_, columnNames_, schema_] :=
	extractHighLevelInfo[
		"Columns",
		tableName,
		Replace[
			columnNames,
			{
				s_?StringQ :> s -> s,
				r: (Rule | RuleDelayed)[_?StringQ, _] :> Rule @@ r,
				any_ :> DBRaise[extractHighLevelInfo, "invalid_field_spec", {any}]
			},
			{1}
		],
		schema
	]

removeMaybe[lor: {___Rule}] := With[{keys = Keys[lor]},
	AssociationThread[
		MapIndexed[
			With[{dropped = Replace[
					#,
					"Maybe"[_] -> Nothing,
					{1}
				]},
				StringRiffle[
					If[
						MemberQ[Take[keys, First[#2] - 1], dropped],
						Replace[
							#,
							"Maybe"[a_] :> a,
							{1}
						],
						dropped
					],
					"-"
				]
			]&,
			keys
		],
		Values[lor]
	]
]

recursiveNamingScheme[vals_, canonical_] := removeMaybe @ Flatten @ recursiveNamingScheme[
	vals,
	{
		{Key["DestinationModel"], True},
		{
			Replace[
				{
					Thread[canonical -> _] :> "reverse",
					lor: {__Rule} :> StringRiffle[lor[[All, 1]], "-"],
					list_ :> StringRiffle[list, "-"],
					_ -> ""
				}
			] @* Key["ModelMapping"],
			False
		},
		{
			Replace[
				{
					lor: {__Rule} :> StringRiffle[lor[[All, 2]], "-"],
					list_ :> StringRiffle[list, "-"],
					_ -> ""
				}
			] @* Key["ModelMapping"],
			False
		},
		{ToString[Position[vals, #][[1]]]&, True}
	},
	{}
]

recursiveNamingScheme[vals_, {{first_, keep_}, rest___}, name_] := Values @ GroupBy[
	vals,
	first,
	If[
		Length[#] === 1,
		Append[name, first[First[#]]] -> First[#],
		recursiveNamingScheme[#, {rest}, Append[name, If[keep, Identity, "Maybe"][first[First[#]]]]]
	]&
]

generateAndNameRelations[store_] := Module[{direct = generateDirectRelations[store]},
	Association @ KeyValueMap[
		Function[{key, val},
			key -> recursiveNamingScheme[val, store[key, "CanonicalFields"]]
		],
		Merge[
			{direct, generateReverseRelations[direct]},
			Apply[Join]
		]
	]
]

addRelations[store: DatabaseStore[meta_, schema_]] := With[{rels = generateAndNameRelations[store]},
	DatabaseStore[
		<|
			"Models" -> AssociationMap[
				Function[
					model,
					Append[
						meta["Models", model],
						"Fields" -> Join[
							meta["Models", model, "Fields"],
							Replace[
								Intersection[store[model, "Fields"], Keys[rels[model]]],
								{
									{} -> rels[model],
									nonEmpty_List :> Association @ KeyValueMap[
										If[
											MemberQ[nonEmpty, #],
											# <> "-" <> If[
												singleRelationType[store, model, #2] === "ManyToOne",
												"Entity",
												"EntityClass"
											],
											#
										] -> #2&,
										rels[model]
									]
								}
							]
						]
					]
				],
				store["Models"]
			]
		|>,
		schema
	]
]


PackageScope["storeCheck"]

storeCheck[schema_DatabaseStore?databaseStoreQ, _: Automatic] := schema
storeCheck[_, msg: _String | Automatic : Automatic] :=
	With[{message = If[msg === Automatic, "failed_retrieving_current_store", msg]},
		DBRaise[storeCheck, message, {}]
	]

databaseStoreQ[dbs: DatabaseStore[meta_, schema_]] /; System`Private`HoldNoEntryQ[dbs] := True
databaseStoreQ[dbs: DatabaseStore[meta_, schema_]] /; !System`Private`HoldNoEntryQ[dbs] := (
	If[
		!KeyExistsQ[meta, "Models"],
		DBRaise[databaseStoreQ, "missing_models", {meta}]
	];
	KeyValueMap[
		Function[
			If[
				!KeyExistsQ[#2, "Fields"],
				DBRaise[databaseStoreQ, "missing_fields", {#1, #2}]
			];
			If[
				!KeyExistsQ[#2, "ModelExtractor"],
				DBRaise[databaseStoreQ, "missing_table_name", {#1, #2}]
			];
			If[
				!KeyExistsQ[#2, "CanonicalFields"],
				DBRaise[databaseStoreQ, "missing_canonical_fields", {#1, #2}],
				Replace[
					Complement[
						#2["CanonicalFields"],
						Keys[#2["Fields"]]
					],
					l: Except[{}] :> DBRaise[databaseStoreQ, "undefined_fields_in_canonical_fields", l]
				]
			]
		],
		meta[["Models"]]
	];
	KeyValueMap[
		Function[
			{model, assoc},
			KeyValueMap[
				Function[
					{field, inner},
					If[
						!Or[
							KeyExistsQ[inner, "ColumnName"] && KeyExistsQ[inner, "ColumnPrefix"],
							KeyExistsQ[inner, "Function"],
							KeyExistsQ[inner, "DestinationModel"] && KeyExistsQ[inner, "ModelMapping"]
						],
						DBRaise[databaseStoreQ, "missing_field_extractor", {model, field, inner}]
					]
				],
				assoc
			]
		],
		meta[["Models", All, "Fields"]]
	];
	True
)

populate[schema_] := populate[schema["Tables"], schema]

populate[table: (_?StringQ | _Rule), schema_] := populate[{table}, schema]

populate[
	tables_List,
	schema_
] := <|"Models" -> extractHighLevelInfo["Tables", tables, schema]|>

populate[args_, schema_] := DBRaise[populate, "wrong_args", {args}]


(* constructors *)

Options[DatabaseStore] := {"AddRelations" -> True}

DatabaseStore[conn: $DBDatabaseReferencePattern, opt: OptionsPattern[]] :=
	With[{schema = DBInspect[conn]},
		DatabaseStore[schema, opt] /; schema =!= $Failed
	]

DatabaseStore[schema: $DBStrictSchemaPattern, opt: OptionsPattern[]] :=
	If[TrueQ[OptionValue["AddRelations"]], addRelations, Identity][
		DatabaseStore[populate[schema], schema]
	]

(dbs: DatabaseStore[
    meta_Association?AssociationQ, schema : $DBStrictSchemaPattern, opt: OptionsPattern[]]
) /; And[
    KeyExistsQ[meta, "Models"],
    System`Private`HoldEntryQ[dbs]
] := If[
	databaseStoreQ[Unevaluated[dbs]],
	System`Private`HoldSetNoEntry[dbs]
]

(dbs: DatabaseStore[meta_, schema : $DBStrictSchemaPattern, opt: OptionsPattern[]]) /;
	System`Private`HoldEntryQ[dbs] :=
		System`Private`SetNoEntry[
			If[
				And[
					TrueQ[OptionValue["AddRelations"]],
					MatchQ[Normal[meta], {((_?StringQ -> Except[_List]) | _?StringQ)..}]
				],
				addRelations,
				Identity
			] @ DatabaseStore[populate[Normal[meta], schema], schema]
		]

dbs: DatabaseStore[a___, inv_RelationalDatabase] /; System`Private`HoldEntryQ[dbs] :=
    DBRaise[RelationalDatabase, "badargs", {HoldForm[a], HoldForm[inv]}]

(*properties*)

(dbs: DatabaseStore[meta_, schema_])["Properties"] /; System`Private`NoEntryQ[dbs] := Union[
	Keys[meta], {"Properties", "Schema", "EntityStore", "AliasAssociation"}]

(dbs: DatabaseStore[meta_, schema_])["Schema"] /; System`Private`NoEntryQ[dbs] := schema

(dbs: DatabaseStore[meta_, schema_])["Models"] /; System`Private`NoEntryQ[dbs] :=
    Keys[Lookup[meta, "Models", <||>]]

dbs_DatabaseStore["AliasAssociation"] /; System`Private`NoEntryQ[dbs] := AssociationMap[
	dbs[#, "ModelExtractor"]&,
	Select[dbs["Models"], StringQ[dbs[#, "ModelExtractor"]]&]
]


dbs_DatabaseStore[model_, "Exists"] /; System`Private`NoEntryQ[dbs] :=
    MemberQ[dbs["Models"], model]

(dbs: DatabaseStore[meta_, schema_])[model_, "Properties"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, "Exists"],
	Union[
		Keys[meta[["Models", model]]],
		{"Exists", "Name", "Properties", "Fields", "AliasAssociation", "ConcreteModelQ"},
		{If[
			dbs[model, "ConcreteModelQ"],
			"TableName",
			"ModelQuery"
		]}
	],
	DBRaise[DatabaseStore, "non_existent_model", {model}]
]

dbs_DatabaseStore[model_, "Name"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, "Exists"],
	model,
	DBRaise[DatabaseStore, "non_existent_model", {model}]
]

dbs_DatabaseStore[model_, "ConcreteModelQ"] /; System`Private`NoEntryQ[dbs] :=
	StringQ[dbs[model, "ModelExtractor"]]

dbs_DatabaseStore[model_, "TableName"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, "Exists"] && dbs[model, "ConcreteModelQ"],
	dbs[model, "ModelExtractor"],
	DBRaise[DatabaseStore, "non_existent_model", {model}]
]

dbs_DatabaseStore[model_, "ModelFunction"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, "Exists"] && !dbs[model, "ConcreteModelQ"],
	dbs[model, "ModelExtractor"],
	DBRaise[DatabaseStore, "non_existent_model", {model}]
]

dbs_DatabaseStore[model_, "AliasAssociation"] /; System`Private`NoEntryQ[dbs] := AssociationMap[
	DBPrefixedField[dbs[model, #, "ColumnPrefix"], dbs[model, #, "ColumnName"]]&,
	Select[dbs[model, "Fields"], dbs[model, #, "FieldType"] === "Column"&]
]


(dbs: DatabaseStore[meta_, schema_])[model_, "Fields"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, "Exists"],
	Keys[Lookup[meta[["Models", model]], "Fields", <||>]],
	DBRaise[DatabaseStore, "non_existent_model", {model}]
]

(dbs: DatabaseStore[meta_, schema_])[model_, prop_?StringQ] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, "Exists"],
	Lookup[
		meta[["Models", model]],
		prop,
		DBRaise[DatabaseStore, "non_existent_model_prop", {model, prop}]
	],
	DBRaise[DatabaseStore, "non_existent_model", {model}]
]

dbs_DatabaseStore[model_, field_, "Exists"] /; System`Private`NoEntryQ[dbs] :=
    MemberQ[dbs[model, "Fields"], field]

(dbs: DatabaseStore[meta_, schema_])[model_, field_, "Properties"] /;
    System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"],
		Union[
			Keys[meta[["Models", model, "Fields", field]]],
			{"Exists", "Name", "Properties", "FieldType", "FieldExtractor"},
			Switch[
				dbs[model, field, "FieldType"],
				"Column", {"ColumnName"},
				"Function", {"Function"},
				"Relation", {"DestinationModel", "ModelMapping", "RelationType"}
			]
		],
		DBRaise[DatabaseStore, "non_existent_field", {model, field}]
	]

dbs_DatabaseStore[model_, field_, "Name"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"],
	field,
	DBRaise[DatabaseStore, "non_existent_field", {model, field}]
]

(dbs: DatabaseStore[meta_, schema_])[model_, field_, "FieldExtractor"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"],
	meta["Models", model, "Fields", field],
	DBRaise[DatabaseStore, "non_existent_field", {model, field}]
]

dbs_DatabaseStore[model_, field_, "FieldType"] /; System`Private`NoEntryQ[dbs] := With[
	{fe = dbs[model, field, "FieldExtractor"]},
	Which[
		KeyExistsQ[fe, "ColumnName"] && KeyExistsQ[fe, "ColumnPrefix"],
			"Column",
		KeyExistsQ[fe, "Function"],
			"Function",
		KeyExistsQ[fe, "DestinationModel"] && KeyExistsQ[fe, "ModelMapping"],
			"Relation"
	]
]

dbs_DatabaseStore[model_, field_, "ColumnName"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"] && dbs[model, field, "FieldType"] === "Column",
	dbs[model, field, "FieldExtractor"]["ColumnName"],
	DBRaise[DatabaseStore, "non_existent_field", {model, field}]
]

dbs_DatabaseStore[model_, field_, "Function"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"] && dbs[model, field, "FieldType"] === "Function",
	dbs[model, field, "FieldExtractor"]["Function"],
	DBRaise[DatabaseStore, "non_existent_field", {model, field}]
]

dbs_DatabaseStore[model_, field_, "DestinationModel"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"] && dbs[model, field, "FieldType"] === "Relation",
	dbs[model, field, "FieldExtractor"]["DestinationModel"],
	DBRaise[DatabaseStore, "non_existent_field", {model, field}]
]

dbs_DatabaseStore[model_, field_, "ModelMapping"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"] && dbs[model, field, "FieldType"] === "Relation",
	dbs[model, field, "FieldExtractor"]["ModelMapping"],
	DBRaise[DatabaseStore, "non_existent_field", {model, field}]
]

dbs_DatabaseStore[model_, field_, "RelationType"] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"] && dbs[model, field, "FieldType"] === "Relation",
	singleRelationType[dbs, model, dbs[model, field, "FieldExtractor"]],
	DBRaise[DatabaseStore, "non_existent_field", {model, field}]
]



(dbs: DatabaseStore[meta_, schema_])[model_, field_,  prop_?StringQ] /; System`Private`NoEntryQ[dbs] := If[
	dbs[model, field, "Exists"],
	Lookup[
		meta[["Models", model, "Fields", field]],
		prop,
		DBRaise[DatabaseStore, "non_existent_field_prop", {model, field, prop}]
	],
	DBRaise[DatabaseStore, "non_existent_field", {model, field}]
]

dbs_DatabaseStore[path: Repeated[_, {0, 2}], All] /; System`Private`NoEntryQ[dbs] := If[
	dbs[path, "Exists"],
	dbs[path, dbs[path, "Properties"]],
	DBRaise[DatabaseStore, "non_existent_path", {{path}}]
]

dbs_DatabaseStore[path: Repeated[_, {0, 2}], props_List] /; System`Private`NoEntryQ[dbs] := If[
	dbs[path, "Exists"],
	Map[
		dbs[path, #] &,
		props
	],
	DBRaise[DatabaseStore, "non_existent_path", {{path}}]
]

(*delegate to RelationalDatabase*)

DatabaseStore[meta_, schema_]["Schema", args___] /; System`Private`NoEntryQ[dbs] := With[
	{res = schema[args]},
	res /; !MatchQ[Head[res], $DBStrictSchemaPattern] && ! MissingQ[res]
]

dbs_DatabaseStore[path___] /; System`Private`NoEntryQ[dbs] :=
    DBRaise[DatabaseStore, "non_existent_path", {{path}}]