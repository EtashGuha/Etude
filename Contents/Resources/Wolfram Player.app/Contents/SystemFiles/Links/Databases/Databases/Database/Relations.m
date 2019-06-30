Package["Databases`Database`"]


PackageImport["Databases`"] (* DatabaseView *)
PackageImport["Databases`SQL`"] (* DBQueryBuilderObject *)
PackageImport["Databases`Common`"] (* DBKeyValueReverse *)

PackageExport["DBCompileJoinRules"]

PackageScope["generateDirectRelations"]
PackageScope["generateReverseRelations"]
PackageScope["aliasedForeignKeys"]
PackageScope["conditionSimplify"]
PackageScope["singleRelationType"]
PackageScope["compileJoinSpec"]
PackageScope["getFields"]
PackageScope["DBRelationFunction"]

mapAt[f_, {}, pos_] := {}

mapAt[f_, l_, {first: Except[_List], pos_List, rest___}] := Fold[
	mapAt[
		f,
		#1	,
		{first, #2, rest}
	]&,
	l,
	pos
]

mapAt[f_, l_, pos_] := MapAt[f, l, pos]

conditionSimplify = If[
	#FromColumns === #ToColumns,
	#ToColumns,
	Thread[#FromColumns -> #ToColumns]
]&

generateDirectRelation = <|
	"DestinationModel" -> #ToTable,
	"ModelMapping" -> conditionSimplify[#]
|>&

generateDirectRelations[store_] :=
    generateDirectRelations[store, store["Models"]]

generateDirectRelations[store_, models: (_List | _Rule)] := With[
	{compiledModels = Replace[
		List @@ models,
		{s_?StringQ :> s, any_ :> DBQueryToQueryBuilderObject[any, store]},
		{1}
	]},
	Map[
		generateDirectRelation,
		AssociationThread[
			List @@ models,
			MapIndexed[
				filterForeignKeys[
					store,
					aliasedForeignKeys[store, #1],
					If[
						MatchQ[models, _Rule],
						Delete[compiledModels, #2],
						compiledModels
					]
				]&,
				List @@ models
			]
		],
		{2}
	]
]

filterForeignKeys[store_, fks_, queries_List] := With[
	{checkingFuncs = Replace[
		queries,
		{
			s_?StringQ :> Function[s === #1 && store[#1, #2, "Exists"]],
			qbo_ :> Function[
				qbo @
					DBResolveTopLevelField[DBPrefixedField[#1 -> #2], False] =!= None
			]
		},
		{1}
	]},
	Select[
		fks,
		AllTrue[
			#ToColumns,
			Function[
				toCol,
				AnyTrue[
					checkingFuncs,
					Function[checkingFunc, checkingFunc[#ToTable, toCol]]
				]
			]
		]&
	]
]



getOriginalModel[store_, DatabaseModel[s_?StringQ] | s_?StringQ, field_] := If[
	store[s, field, "Exists"],
	s,
	DBRaise[getOriginalModel, "model_not_found", {field}]
]

getOriginalModel[store_, query_?DBInertQueryQ, field_] :=
	getOriginalModel[store, DBQueryToQueryBuilderObject[query, store], field]

getOriginalModel[store_, query_?DBQueryBuilderObjectQ, field_] := Replace[
   query @ DBGetPrefixedFields[field],
	{
		$Failed :> DBRaise[getOriginalModel, "model_not_found", {field}],
		f_DBPrefixedField ? DBReallyPrefixedFieldQ :> Replace[
			With[{
				parts = DBPrefixedFieldPartsList[f]},
				Scan[
					If[
						Catch[
							getOriginalModel[store, DatabaseModel[#], Last[parts]],
							DBError
						] === #,
						Return[#]
					]&,
					Reverse[Most[parts]]
				]
			],
			Null :> DBRaise[getOriginalModel, "model_not_found", {field}]
		]
	}
]

DBDefError @ getOriginalModel


getOriginalTable[field_DBPrefixedField ? DBReallyPrefixedFieldQ] :=
	Replace[First[DBPrefixedFieldPartsList[field]], Except[_?StringQ] -> None]

getOriginalTable[any_] := None

groupedByTable[store_, m_] := KeyDrop[None] @ GroupBy[
	If[
		StringQ[m] && store[m, "Exists"],
		store[m, "AliasAssociation"],
		Module[{assoc, uniquenames, nonunique},
			assoc = AssociationMap[
				Last @* DBPrefixedFieldParts,
				Replace[
					m,
					model: Except[_?DBQueryBuilderObjectQ] :>
						DBQueryToQueryBuilderObject[model, store]
				] @ DBGetPrefixedSelectedFields[]
			];
			uniquenames = DeleteDuplicates[assoc];
			nonunique = Keys[KeyComplement[{assoc, uniquenames}]];
			DBKeyValueReverse @ Join[uniquenames, AssociationThread[nonunique, nonunique]]
		]
	],
	getOriginalTable
]

explodeFromColumns[aliases_, store_] := Map[
	Function[
		fromCols,
		If[
			store[#ToTable, "Exists"],
			Append[
				#,
				"FromColumns" -> fromCols
			],
			Nothing
		]
	],
	Transpose[
		Map[
			ReplaceList[
				Map[
					Reverse,
					Normal[
						Last @* DBPrefixedFieldParts /@ aliases
					]
				]
			],
			#FromColumns
		]
	]
]&

aliasForeignKey[store_] := Function[
	Map[
		Function[
			model,
			Append[#, "ToTable" -> model]
		],
		Select[
			store["Models"],
			Function[
				model,
				With[{cf = store[model, "CanonicalFields"]},
					And[
						Length[cf] == Length[#ToColumns],
						AllTrue[
							cf,
							Function[
								field,
								And[
									#ToTable === store[model, field, "ColumnPrefix"],
									MemberQ[#ToColumns, store[model, field, "ColumnName"]]
								]
							]
						]
					]
				]
			]
		]
	]
]

aliasedForeignKeys[store_, m_] := Flatten @ KeyValueMap[
	Function[
		{table, aliases},
		Map[
			explodeFromColumns[aliases, store],
			Join @@ Map[
				aliasForeignKey[store],
				store["Schema"][table, "ForeignKeys"]
			]
		]
	],
	groupedByTable[store, m]
]

reverseCondition[lor: {__Rule}] := Reverse /@ lor

reverseCondition[los: {__?StringQ}] := los

reverseCondition[DatabaseFunction[{sym1_, sym2_}, body_]] := DatabaseFunction[{sym2, sym1}, body]

DBDefError @ reverseCondition

generateReverseRelations[directRels_] := GroupBy[
	Map[
		#DestinationModel -> <|
			"DestinationModel" -> #FromTable,
			"ModelMapping" -> reverseCondition[#ModelMapping]
		|>&,
		Join @@ KeyValueMap[
			Function[{key, val},
				Map[
					Append[#, "FromTable" -> key]&,
					val
				]
			],
			directRels
		]
	],
	First -> Last
]

(*TODO ToTable might be a query*)

singleRelationType[store_, model_, assoc_?AssociationQ] := With[
	{fk = aliasedForeignKeys[store, model]},
	If[
		And[
			fk =!= {},
			MemberQ[fk[[All, "ToTable"]], assoc["DestinationModel"]],
			Select[
				Cases[fk, KeyValuePattern["ToTable" -> assoc["DestinationModel"]]],
				conditionSimplify[#] === assoc["ModelMapping"]&
			] =!= {}
		],
		"ManyToOne",
		"OneToMany"
	]
]

relationTargetAndType[store_, model_, rel: Except[_List]] := relationType[store, model, {rel}]

relationTargetAndType[store_, model_, l_List] := Fold[
	With[
		{relAssoc = Replace[
			#2,
			name_?StringQ :> store[First[#1], name, "FieldExtractor"]
		]},
		{
			relAssoc["DestinationModel"],
			Switch[
				{Last[#1], singleRelationType[store, First[#1], relAssoc]},
				{"OneToMany" | None, "OneToMany"}, "OneToMany",
				{"ManyToOne" | None, "ManyToOne"}, "ManyToOne",
				_, "ManyToMany"
			]
		}
	]&,
	{model, None},
	l
]

getCanonicalFields[store_, s_?StringQ] /; store[s, "Exists"] := store[s, "CanonicalFields"]

getCanonicalFields[store_, q_?DBUncompiledQueryQ] :=
	DBQueryToQueryBuilderObject[q, store] @ DBPrimaryKey[]

DBDefError @ getCanonicalFields

toList[l_List] := l
toList[expr_] := {expr}


getAutomaticRelation[store_, DatabaseModel[left_], right_] :=
    getAutomaticRelation[store, left, right]

getAutomaticRelation[store_, left_, DatabaseModel[right_]] :=
	getAutomaticRelation[store, left, right]

getAutomaticRelation[store_, left_, right_] := With[
	{direct = generateDirectRelations[store, left -> right]},
	Replace[
		Merge[
			{direct, generateReverseRelations[direct]},
			Apply[Join]
		][left],
		{
			{a_} :> a["ModelMapping"],
			{} :> DBRaise[getAutomaticRelation, "no_relation_found", {left, right}],
			rels_ :> DBRaise[getAutomaticRelation, "more_than_one_relation_found", {rels}]
		}
	]
]

$fieldSpec = _DBPrefixedField | _?StringQ

DBCompileJoinRules[store_, left_, right_, Automatic] :=
    DBCompileJoinRules[store, left, right, getAutomaticRelation[store, left, right]]

DBCompileJoinRules[store_, left_, right_, field: $fieldSpec] :=
	DBCompileJoinRules[store, left, right, {field}]

DBCompileJoinRules[store_, left_, right_, Automatic -> l_] :=
	DBCompileJoinRules[
		store, left, right, getCanonicalFields[store, left] -> toList[l]
	]

DBCompileJoinRules[store_, left_, right_, l_ -> Automatic] :=
	DBCompileJoinRules[
		store, left, right, toList[l] -> getCanonicalFields[store, right]
	]

DBCompileJoinRules[store_, left_, right_, (None -> _) | (_ -> None)] :=
	DBRaise[
		DBCompileJoinRules,
		"one_or_both_joined_queries_have_no_pk",
		{left, right}
	]

DBCompileJoinRules[
	store_,
	left_,
	right_,
	leftFields: {$fieldSpec..} -> rightFields: {$fieldSpec..}
] :=
	If[Length[leftFields] === Length[rightFields],
		Thread[leftFields -> rightFields],
	(* else *)
		DBRaise[
			DBCompileJoinRules,
			"incompatible_primary_foreign_keys_for_join",
			{leftFields, rightFields}
		]
	]

DBCompileJoinRules[store_, left_, right_, l: $fieldSpec -> r: $fieldSpec] := {l -> r}

DBCompileJoinRules[store_, left_, right_, list: {($fieldSpec | ($fieldSpec -> $fieldSpec))..}] :=
	Replace[list, x: Except[_Rule] :> x -> x, {1}]

DBCompileJoinRules[args___] := DBRaise[DBCompileJoinRules, "invalid_join_spec", {args}]

compileJoinSpec[context_, left_, right_, Automatic] :=
	compileJoinSpec[
        context, left, right, getAutomaticRelation[context["Store"], left, right]
    ]

compileJoinSpec[context_, left_, right_, any_] :=
    compileJoinSpec[
        context, left, right, DBCompileJoinRules[context["Store"], left, right, any]
    ]

errorHandler[side_] := 	DBHandleError[
	compileJoinSpec,
	Replace[
		#2,
		Failure["no_field_found", assoc_] :>
			DBRaise[
				compleJoinSpec,
				"no_field_found_join",
				{side, assoc[["FailingFunctionArgs", 1]]}
			]
	]&
]


qboFromCompilationContext[context_, slot_DBSQLSlot] :=
    Replace[
        context["SlotQueryMap"][slot],
        _Missing :> DBRaise[qboFromCompilationContext, "unbound_slot", {slot}]
    ]
    
qboFromCompilationContext[context_, query_] := compileQuery[query, context]
    
DefError @ qboFromCompilationContext    


compileJoinSpec[context_, left_, right_, lor: {($fieldSpec -> $fieldSpec)..}] := With[{
	l = DBUniqueTemporary["left"],
	r = DBUniqueTemporary["right"],
	leftqbo = qboFromCompilationContext[context, left],
	rightqbo = qboFromCompilationContext[context, right]
    },
	errorHandler[left] @ Map[
		leftqbo @ DBResolveTopLevelSelectedField[#]&,
		lor[[All, 1]]
	];
	errorHandler[right] @ Map[
		rightqbo @ DBResolveTopLevelSelectedField[#]&,
		lor[[All, 2]]
	];
	With[{
		body = And @@ (l[#1] == r[#2]& @@@ lor)},
		DatabaseFunction[{l, r}, body]
	]
]	

compileJoinSpec[___, dbf_DatabaseFunction] := dbf

compileJoinSpec[args___] := DBRaise[compileJoinSpec, "invalid_join_spec", {args}]

getFields[l: Except[_DatabaseFunction], False] := DBCompileJoinRules[None, None, None, l]

getFields[l: Except[_DatabaseFunction], True] := DBCompileJoinRules[None, None, None, l][[All, 1]]

getFields[dbf_DatabaseFunction, True] := getFieldsFromDBF[dbf]

(* the case where there is a DBF as condition cannot occur when 2nd arg is True, because we only care about
ManyToOne relations *)

DBDefError @ getFields


SetAttributes[getFieldsFromDBF, HoldFirst]

getFieldsFromDBF[DatabaseFunction[slots: Except[_List], body_]] := getFieldsFromDBF[DatabaseFunction[{slots}, body]]

getFieldsFromDBF[DatabaseFunction[{left_, rest___}, body_]] :=
    Replace[
        Reap[getFieldsFromDBF[left, Hold[body]], getFieldsFromDBF],
        {
            {_, {}} -> {},
            {_, {x_}} :> x
        }
    ]        

getFieldsFromDBF[slot_, body_Hold] := ReplaceAll[
	body,
	{
		HoldPattern[slot][field_] :> RuleCondition[Sow[field, getFieldsFromDBF]],
		DatabaseFunction[{left_, _}, b_][HoldPattern[slot], _] :> getFieldsFromDBF[left, Hold[b]],
		DatabaseFunction[{_, right_}, b_][_, HoldPattern[slot]] :> getFieldsFromDBF[right, Hold[b]],
		DatabaseFunction[{only_} | only_, b_][HoldPattern[slot]] :> getFieldsFromDBF[only, Hold[b]]
	}
]

DBDefError @ getFieldsFromDBF

DBRelationFunction[spec_?StringQ][context_][left_] :=
    DBRelationType[
        context, 
        context["Store"][getOriginalModel[left, spec], spec, "FieldExtractor"]
    ][left]

DBRelationFunction[spec_?AssociationQ][context_][left_] :=
    DBRelationFunction[
        spec, 
        singleRelationType[context["Store"], left, spec]
    ][context][left]

DBRelationFunction[spec_?AssociationQ, "ManyToOne"][context_][left_] := With[
	{assoc = Association @ MapAt[
		left[#]&,
		Reverse[
			DBCompileJoinRules[
                context["Store"], left, spec["DestinationModel"], spec["ModelMapping"]
            ],
			2
		],
		{All, 2}
	]},
	If[
		AnyTrue[assoc, MissingQ],
		Missing["NotAvailable"],
		DatabaseModelInstance[
			spec["DestinationModel"],
			assoc
		]
	]
]

DBRelationFunction[spec_?AssociationQ, type_][context_][left_] := 
    With[{right = DBUniqueTemporary["right"]},
        DatabaseWhere[
			spec["DestinationModel"],
			DatabaseFunction @@ Prepend[
				DBBodyReplace[
					{left, right},
					compileJoinSpec[
						context,
						left,
						spec["DestinationModel"],
						spec["ModelMapping"]
					]
				],
				right
			]
		]
	]
