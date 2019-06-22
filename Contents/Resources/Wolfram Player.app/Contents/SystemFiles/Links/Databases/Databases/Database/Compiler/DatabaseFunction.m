Package["Databases`Database`"]

PackageImport["Databases`"] (* DatabaseFunction *)
PackageImport["Databases`Common`"] (* DBUnevaluatedPatternCheck, DBBlockExtend, DBRaise *)
PackageImport["Databases`SQL`"] (* DBExprToAST *)
PackageImport["Databases`Schema`"] (* DBSchemaGet *)


PackageExport["DBCompileDatabaseFunction"]
PackageExport["$DBUseOldDBFCompilationScheme"]
PackageExport["DBBodyReplace"]
PackageExport["DBShieldAliases"]

PackageExport["DBSQLSlot"]


(* TODO: temporary, to have both branches working during the transition*)
$DBUseOldDBFCompilationScheme = False


SetAttributes[DBSQLSlot, HoldAll]


DatabaseFunction[var_Symbol, body_][arg_] :=
    DatabaseFunction[{var}, body][arg]

(dbf: DatabaseFunction[{_Symbol}, _])[
    arg_?DBUncompiledQueryQ
] := With[{res = Catch[
		DBQueryToQueryBuilderObject @ DatabaseSelectFields[
			DatabaseAnnotate[
				arg,
				"dummy_field" -> dbf
			],
			{"dummy_field"}
		],
		DBError
	]},
	If[
		FailureQ[res],
		First @* First,
		First
	] @ DBRunQuery[
		If[
			FailureQ[res],
			DBQueryToQueryBuilderObject @ DatabaseSelectFields[
				DatabaseAggregate[
					arg,
					"dummy_field" -> dbf
				],
				{"dummy_field"}
			],
			res
		],
		"Columns"
	]
]


DBCompileDatabaseFunction[compilationContext_Association?AssociationQ] :=
    Function[dbfun, DBCompileDatabaseFunction[dbfun, compilationContext]]

DBCompileDatabaseFunction[
    DatabaseFunction[slot_Symbol, body_],
    rest___
] :=
    DBCompileDatabaseFunction[
        DatabaseFunction[{slot}, body],
        rest
    ]

DBCompileDatabaseFunction[
    dbf_DatabaseFunction, 
    compilationContext_Association?AssociationQ
] /; TrueQ[$DBQueryLightWeightCompilation] := 
    DBTyped[
        DBUncompiledExpression[ToString[dbf]],
        DBType[DBTypeUnion[]]
    ]


DBCompileDatabaseFunction[
   dbf: DatabaseFunction[
        slots: {__Symbol},
        body_
    ],
	compilationContext_Association?AssociationQ
] :=
    With[{spec = makeCompilationSpec[dbf, compilationContext]},
		Composition[
			aliasResolve[spec["SlotQueryMap"], spec["IsJoin"]],
			spec["TypeAwareCompiler"],
			spec["Compiler"]
		] @ spec["HeldBody"]
	]

DBCompileDatabaseFunction[f_DatabaseFunction] :=
    DBRaise[DBCompileDatabaseFunction, "missing_compilation_context", {}]

DBCompileDatabaseFunction[f_DatabaseFunction, _Association?AssociationQ] :=
    DBRaise[DBCompileDatabaseFunction, "invalid_DatabaseFunction", { f }]

DBDefError @ DBCompileDatabaseFunction


makeCompilationSpec[
	dbf: DatabaseFunction[dbfslots: {__Symbol},body_],
	compilationContext_Association?AssociationQ
] :=
	Module[
        {queries, isAggregate, slots, compiler, typeAwareCompiler, queryMap, newCompilationContext}
        ,
        queries = checkQueries @ compilationContext["QueryBuilderObjects"];
        isAggregate = checkBoolean @ compilationContext["Aggregate"];
		slots = Map[DBSQLSlot, Unevaluated[dbfslots]];
		queryMap = Join[
			checkSlotMap @ compilationContext["SlotQueryMap"],
			AssociationThread[slots, queries]
		];
		newCompilationContext = Append[
			KeyDrop[compilationContext, {"QueryBuilderObjects", "Aggregate"}],
			"SlotQueryMap" -> queryMap
		];
		compiler = DBApplyToHeldArguments[compileWLExpression[
			newCompilationContext, compileQueryWithRelations
		]];
		typeAwareCompiler = If[TrueQ[$DBUseOldDBFCompilationScheme],
            ReplaceAll[astSymbol -> DBSQLSymbol],
            (* else*)
            DBTypeAwareCompiler[newCompilationContext, isAggregate]
        ];
		<|
			"HeldBody" -> With[{s = slots}, DBBodyReplace[s, dbf]],
			"IsJoin" -> Length[Unevaluated[dbfslots]] === 2,
			"SlotQueryMap" -> queryMap,
			"Compiler" -> compiler,
			"TypeAwareCompiler" -> typeAwareCompiler
		|>
	]

DBDefError @ makeCompilationSpec


(*
** Resolves aliases for DBSQLSlot[]s in compiled SQL expression,
** when dealing with a JOIN we need to get the current alias from the prefix trie
*)
aliasResolve[aliasAssoc_Association?AssociationQ, isJoin_] :=
   ReplaceAll[
        {
			(qbo_?DBQueryBuilderObjectQ)[prop: _String | _Rule | _DBPrefixedField | _DBRawFieldName] /;
       			AnyTrue[
                    Values[Take[aliasAssoc, If[TrueQ[isJoin], -2, -1]]],
                    qbo @ DBGetName[] === # @ DBGetName[] &
                ] :>
                    RuleCondition[
						DBSQLField[
							If[
								TrueQ[isJoin],
								DBPrefixedField[
									Replace[
										qbo @ DBGetCurrentAlias[],
										{
											None :> #,
											any_ :> Replace[
												#,
												Except[
													f_DBPrefixedField /;
                 										DBPrefixedFieldPartsList[f][[-2]] === any
												] :> any -> #
											]
										}
									]
								]&,
								Identity
							][prop]
						]
					]
        }
    ]

(*
** Returns held body with aliases temporarily replaced with unique string 
** tokens, and the inverse rules to recover them back. The DBAliased is an inert 
** symbolic helper token
*)
SetAttributes[DBShieldAliases, HoldFirst]
DBShieldAliases[body_] :=
    Replace[
        Replace[
            Hold[body],
            DatabaseQueryMakeAlias[e_, a_] :> DatabaseQueryMakeAlias[e, DBAliased[a]],
            {0, Infinity},
            Heads -> True
        ],
        Hold[expr_] :> DBShieldExpression[
            expr,
            {DBAliased[a_] :> a}
        ]
    ]


SetAttributes[DBBodyReplace, HoldFirst]
DBBodyReplace[{val__}, DatabaseFunction[slot_, body_]] :=
    With[{shielded = DBShieldAliases[body]},
        With[{
            b = ReplaceAll[
                shielded["HeldExpression"],
                {
                    DatabaseFunction -> Function,
                    Association -> fakeAssoc
                }
            ],
            pos = Position[
                shielded["HeldExpression"], Function, {0, Infinity}, Heads -> True
            ]
            },
            ReplacePart[
                ReplaceAll[
                    ReplaceAll[ (* Recover back the aliases after function is applied *)
                        Apply[Function, Hold[slot, b, HoldAll]][val],
                        shielded["BackRules"]
                    ],
                    {
                        Function -> DatabaseFunction,
                        fakeAssoc -> Association
                    }
                ],
                pos -> Function
            ]
        ]
    ]



compileWLExpression[
    compilationContext_Association?AssociationQ,
	queryCompiler_: compileQuery,
	queryPredicate_: DBInertQueryQ
] :=
    DBExprToASTAnnotated @ DBBlockExtend[DBExprToAST, {
            DBExprToAST[q_?(DBUnevaluatedPatternCheck[queryPredicate])] :=
                queryCompiler[q, compilationContext]
            ,
            DBExprToAST[q_[prop_?(DBUnevaluatedPatternCheck[dbPropertyExtractorQ])]] :=
                DBExprToAST[q][prop]
            ,
			DBExprToAST[(dbf: DatabaseFunction[slots__ | {slots__}, body_])[s__]] /;
				Length[Unevaluated[{slots}]] == Length[Unevaluated[{s}]] := With[
					{compiled = DBExprToAST /@ Unevaluated[{s}]},
					DBExprToAST @@ DBBodyReplace[compiled, dbf]
				]
			,
			DBExprToAST[s_[dbf: DatabaseFunction[slot_ | {slot_}, body_]]] := With[
				{compiled = DBExprToAST[s]},
       			DBExprToAST @@ DBBodyReplace[compiled, dbf]
			]
            ,
            DBExprToAST[slot: DBSQLSlot[_Symbol]] := slot
			,
			DBExprToAST[
                qbo_?(DBUnevaluatedPatternCheck[DBQueryBuilderObjectQ])
            ] := qbo
        },
        False
    ]

DBDefError @ compileWLExpression


compileQueryWithRelations[args___] := DBBlockExtend[
    compileQuery,
    {
        compileQuery[slot_DBSQLSlot[prop_], compilationContext_] :=
            Module[{qbo, field, deserializer},
                (* Get the QBO that corresponds to this slot *)
                qbo = compilationContext["SlotQueryMap"][slot];
                If[MissingQ[qbo],
                    DBRaise[
                        compileQueryWithRelations, 
                        "unbound_slot",
                        {slot},
                        <| "CompilationContext" -> compilationContext |>
                    ]
                ];
                (* Make sure the field / property exists *)
                field = qbo @ DBResolveTopLevelField[prop, False];
                If[field === None,
                    DBRaise[compileQueryWithRelations, "no_field_found", {prop}]
                ];
                (* Extract deserializer and dynamically replace the compilation context *)
                deserializer = Replace[
                    DBType[field]["Deserializer"],
                    (dbr_DBRelationFunction)[oldContext_] :> dbr[compilationContext] 
                ];
                (* Make sure the field / property is a relation *)
                If[MissingQ[deserializer],
                    DBRaise[
                        compileQueryWithRelations, 
                        "property_is_not_a_relation",
                        {prop},
                        <| "Slot" -> slot[prop], "CompilationContext" -> compilationContext|>
                    ]
                ];
                (* 
                ** Effectively replace slot[prop] with (prop-relation-function)[slot],
                ** thereby expanding the relation, and compile resulting query.
                *)
                compileQuery[deserializer[slot], compilationContext]
            ]
    }
][args]

