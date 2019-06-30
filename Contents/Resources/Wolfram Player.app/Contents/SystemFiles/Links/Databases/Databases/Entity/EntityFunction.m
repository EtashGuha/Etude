Package["Databases`Entity`"]


PackageImport["EntityFramework`"]

PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`Database`"]


PackageExport["DBCompileEntityFunction"]

PackageScope["$CompiledEntityFunctions"]
PackageScope["$DBInEntityFunction"]

$entityHeads = Alternatives[
	_Entity,
	_EntityClass,
	_FilteredEntityClass,
	_SortedEntityClass,
	_ExtendedEntityClass,
	_AggregatedEntityClass,
	_CombinedEntityClass,
	_SampledEntityClass,
	_EntityValue
]


SetAttributes[customHold, HoldAll]

$DBInEntityFunction = False


SetAttributes[$extended, HoldRest]

$extended[store_, any_] := With[{
	dbq = Block[{$DBInEntityFunction = True}, DBBlockExtend[
		DBToDatabaseQuery,
		{
            DBToDatabaseQuery[es_, s_holdSlot] := s,
            DBToDatabaseQuery[es_, s_holdSlot[prop_]] := s[prop],
            DBToDatabaseQuery[es_, arg_] := DBRaise[
                DBToDatabaseQuery,
                "uncompileable_expression",
                {HoldForm[arg]}
            ]
        }, 
        False
	][store, any]]},
	AppendTo[
		$CompiledEntityQueries,
		With[{
			res = ReplaceAll[
				HoldPattern[dbq],
				dbf: DatabaseFunction[l_List, _] :> RuleCondition[
					With[{
						slots = Table[
							Pattern[Evaluate[DBUniqueTemporary["a"]], Blank[]],
							Length[Unevaluated[l]]
						]},
						DatabaseFunction @@ Join[
							Hold[slots],
							DBBodyReplace[
								slots,
								dbf
							]
						]
					]
				]
			]},
			ReplaceAll[
				res :> any,
				holdSlot -> DBSQLSlot
			]
		]
	];
	dbq
]


SetAttributes[holdSlot, HoldFirst]

$CompiledEntityFunctions = <||>

$maxEFIterations = 30


aliasShielded[f_][rules_List] := 
    Composition[
        Function[ReplaceAll[#["HeldExpression"], #["BackRules"]]]
        ,
        MapAt[
            f @ Join[
                {noop: _Databases`SQL`DBPrefixedField | _EntityProperty :> noop},
                rules
            ],
            {Key["HeldExpression"]}
        ]
        ,
        Replace[Hold[e_] :> DBShieldAliases[e]]
    ]
    

aliasShielded[f_][rule:_Rule | _RuleDelayed] := aliasShielded[f][{rule}]

aliasShielded[f_][args___] := DBRaise[aliasShielded, "invalid_rules", {f, {args}}]


replace0Inf[rules_] := Function[ex, Replace[ex, rules, {0, Infinity}]]


DBCompileEntityFunction[es_, ef_EntityFunction] /; TrueQ[$DBQueryLightWeightCompilation] :=
	DatabaseFunction[ToString[ef]]

DBCompileEntityFunction[es: _EntityStore | None, EntityFunction[slot: (_Symbol | _holdSlot), expr_]] :=
	DBCompileEntityFunction[es, EntityFunction[{slot}, expr]]

DBCompileEntityFunction[
	es: _EntityStore | None,
	ef: EntityFunction[slots: {(_Symbol | _holdSlot)..}, expr_]
] := With[{dbf = Apply[DatabaseFunction] @ Join[
		Hold[slots]
		,
        Composition[
            FixedPoint[
                Composition[ (* Note that the order of replacements is important *)		
                    DBIterationLimiter[$maxEFIterations, DBCompileEntityFunction]
                    ,
                    (* Replace all simple EntityProperty-s with their Database counterparts *)
					replace0Inf[
                        ep: EntityProperty[_, _?StringQ] :> RuleCondition[toPrefixedField @ ep]
                    ]
                    ,
                    aliasShielded[replace0Inf][  
                        DatabaseSelectFields[i_DatabaseModelInstance, f_] :> i[f]
                    ]
                ],
                #
            ]&
            ,
    		FixedPoint[
    			Composition[ (* Note that the order of replacements is important *)
    				
                    DBIterationLimiter[$maxEFIterations, DBCompileEntityFunction]
                    ,
                    (* Recover bound variables at the end *)
    				ReplaceAll[holdSlot[s_] /; MemberQ[Unevaluated[slots], Unevaluated[s]] :> s]
                    ,
                    (* Compile inner queries in EV calls *)
    				aliasShielded[ReplaceAll][	
    					q: $entityHeads :> RuleCondition[$extended[es, q]]
    				]
                    ,
                    (* Rewrite all EV calls as class[props] inert expressions *)
                    aliasShielded[replace0Inf][
                        HoldPattern[EntityValue[ent_, prop_]] :> ent[prop]
                    ]
                    ,
                    (* Dress bare string types into EntityClass in inner EV calls *)
                    aliasShielded[replace0Inf][
                        HoldPattern[EntityValue[class_String, prop_]] :> EntityValue[
                            EntityClass[class,{}], 
                            prop
                        ]
                    ],
    				aliasShielded[replace0Inf][
    					HoldPattern[q_["EntityCount"] | EntityValue[q_, "EntityCount"]] :>
             				Length[q]
             		]
                    ,
                    Composition[ (* Just to stress that the two operations below should always be together *)
                        (* Clean up the temp. holding wrappers *)
                        aliasShielded[replace0Inf][customHold[x_] :> x]
                        ,
                        (* 
                        ** Generate necessary synthetic properties for EntityFunction - 
                        ** valued properties in inner EntityValue calls
                        *)
                        aliasShielded[replace0Inf][
                            HoldPattern[EntityValue[x_, props_]] :>
                                With[{processed = processProperties[Unevaluated @ x, props]},
                                    With[{replaced = Replace[
                                            processed,
                                            {Hold[q_], p_} :> customHold[EntityValue[q, p]]
                                        ]},
                                        replaced /; True
                                    ]/; processed =!= {Hold[x], props}
                                ] 
                        ]
                    ]
                    ,
                    (* 
                    ** Dress all complex types in EntityProperty expressions in 
                    ** DBHeldAlias (this prevents evaluation leaks)
                    *)
                    aliasShielded[replace0Inf][
                        HoldPattern[EntityProperty[t:Except[_String], p_]] :>
                            EntityProperty[DBHeldAlias[t], p]
                    ]
                    ,
                    (* Convert all forms to EntityValue calls *)
                    aliasShielded[replace0Inf][{
        				(ep: _EntityProperty | _EntityFunction)[q_] :> EntityValue[q, ep],
                        (q: $entityHeads | _holdSlot)[props_] :> EntityValue[q, props]
                    }]
                    ,
                    (* Dress EF's bound variables*)
    				ReplaceAll[
    					s_Symbol /; MemberQ[Unevaluated[slots], Unevaluated[s]] :>
    					holdSlot[s]
    				]
    			],
    			#
    		] & 
        ] @ Hold[expr]
	]},
	AppendTo[$CompiledEntityFunctions, dbf -> ef];
	dbf
]
