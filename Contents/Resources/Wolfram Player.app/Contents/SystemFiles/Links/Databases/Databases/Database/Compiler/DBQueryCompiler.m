Package["Databases`Database`"]

(*
    DatabaseStore, DatabaseView
    ,
    DatabaseSQLDistinct, DatabaseSQLAscending, DatabaseSQLDescending,
    DatabaseQueryMakeAlias, DatabaseWhere, DatabaseAnnotate, DatabaseAggregate,
    DatabaseOrderBy, DatabaseGroupBy, DatabaseSelectFields, DatabaseExcludeFields,
    DatabaseJoin, DatabaseLimit, DatabaseOffset
*)
PackageImport["Databases`"]

(* DBUniqueTemporary *)
PackageImport["Databases`Common`"]

(*
    DBQueryBuilderObjectQ, DBQueryBuilderObject, DBSQLField, table, DBWhere, DBAnnotate, DBAggregate, DBOrderBy,
    DBGroupBy, DBValues, DBRemoveSeveralFieldsFromSelected, DBJoin, DBDistinct, DBAsSQL
*)
PackageImport["Databases`SQL`"]
PackageImport["Databases`Schema`"]



PackageScope["compileQuery"]
PackageScope["$orderByFieldPattern"]
PackageScope["$queryOperationHead"]

$syntheticFieldPattern = $DBTopLevelFieldNamePattern | (_String -> _DatabaseFunction) | _DatabaseFunction

$orderByFieldPattern = Alternatives[
    $syntheticFieldPattern,
    (DatabaseSQLAscending | DatabaseSQLDescending)[$syntheticFieldPattern]
]


(*
** Main query compiler, inner function. Requries explicit store
*)

$queryOperationHead = Alternatives[
    DatabaseQueryMakeAlias,
    DatabaseWhere,
    DatabaseAnnotate,
    DatabaseAggregate,
    DatabaseOrderBy,
    DatabaseGroupBy,
    DatabaseSelectFields,
    DatabaseExcludeFields,
    DatabaseJoin,
    DatabaseSQLDistinct,
    DatabaseLimit,
    DatabaseOffset
    DatabaseModel,
    DatabaseModelInstance
]


compileQuery[qbo_?DBQueryBuilderObjectQ, _] := qbo

(* Short-cut query-aliasing syntax *)
compileQuery[(h: $queryOperationHead)[queryAlias_ -> query_, rest___], compilationContext_] :=
    compileQuery[h[DatabaseQueryMakeAlias[query, queryAlias], rest], compilationContext]

compileQuery[DatabaseJoin[q1_, queryAlias_ -> q2_, rest___], compilationContext_] :=
    compileQuery[DatabaseJoin[q1, DatabaseQueryMakeAlias[q2, queryAlias], rest], compilationContext]


compileQuery[tableName_?StringQ, compilationContext_] :=
    compileQuery[DatabaseModel[tableName], compilationContext]

compileQuery[dbm: DatabaseModel[name_String], compilationContext_] := With[
    {store = compilationContext["Store"]},
    With[{
            fields = GroupBy[
                store[name, "Fields"], 
                store[name, #, "FieldType"]&
            ]
        },
		If[
			KeyExistsQ[fields, "Relation"]
			,
			Function[inner,
				inner[DBAnnotate[
					fullProcess["Annotation", compilationContext, inner] @ AssociationMap[
						store[name, #, "FieldExtractor"]&,
						fields["Relation"]
					],
                    True
				]]
			]
			,
			Identity
		] @ If[
            KeyExistsQ[fields, "Function"]
            ,
            Function[inner,
                inner[DBAnnotate[
                    fullProcess["Annotation", compilationContext, inner] @ AssociationMap[
                        store[name, #, "Function"]&,
                        fields["Function"]
                    ],
                    True
                ]]
            ]
            ,
            Identity
        ] @ If[
                store[name, "ConcreteModelQ"],
                If[
                    store[name, "TableName"] =!= name,
                    Function[
                        inner,
                        inner @ DBAlias[name, True]
                    ],
                    Identity
                ] @ checkQuery[DBQueryBuilderObject @ DBCreateTable[
					store[name, "TableName"],
                    store[name, "TableName"],
                    Map[Last @* DBPrefixedFieldParts, store[name, "AliasAssociation"]],
                    store["Schema"]
                ]],
                compileQuery[
                    store[name, "ModelExtractor"],
                    Append[compilationContext, "Store" -> DatabaseStore[store["Schema"], "AddRelations" -> False]]
              ] @ DBAlias[name, True]
        ]
    ]
]


instanceConditionFromPK[<||>] := With[{x = DBUniqueTemporary["dbmi"]},
    DatabaseFunction[x, True]
]

instanceConditionFromPK[pk_Association?AssociationQ] :=
    With[{x = DBUniqueTemporary["dbmi"]},
        Replace[
            Apply[Join] @ KeyValueMap[Function[{k,v}, Hold[x[k] == v]], pk],
            Hold[conds__] :> DatabaseFunction[x, And[conds]]
        ]
    ]


compileQuery[dbm: DatabaseModelInstance[q_, pk_], compilationContext_] :=
    compileQuery[
        DatabaseWhere[q, instanceConditionFromPK[pk]],
        compilationContext
    ] @ DBSetSingleRow[True]
    
compileQuery[DatabaseQueryMakeAlias[inner_, name_, sticky_: False], compilationContext_] :=
	checkQuery[compileQuery[inner, compilationContext] @ DBAlias[name, sticky]]

compileQuery[DatabaseQueryMakeAlias[name_, sticky_: False][q_], compilationContext_] :=
    compileQuery[DatabaseQueryMakeAlias[q, name, sticky], compilationContext]  (* Operator form *)



compileQuery[DatabaseWhere[inner_, cond_: DBTyped[True, DBType["Boolean"]]], compilationContext_] :=
    With[{compiledInner = checkQuery[compileQuery[inner, compilationContext]]},
        compiledInner @ DBWhere[
            Composition[
                stripExpressionType,
                checkExpressionType[DBType["Boolean"]],
                fullProcess["Expression", compilationContext, compiledInner]
            ][cond]
        ]
    ]

compileQuery[DatabaseWhere[cond_][q_], compilationContext_] :=
    compileQuery[DatabaseWhere[q, cond], compilationContext] (* Operator form *)



compileQuery[DatabaseAnnotate[inner_, annotation: _Rule | {___Rule} | _?AssociationQ], compilationContext_] :=
    With[{compiledInner = checkQuery[compileQuery[inner, compilationContext]]},
        compiledInner @ DBAnnotate[
            fullProcess["Annotation", compilationContext, compiledInner] @ annotation
        ]
    ]

compileQuery[DatabaseAnnotate[annotation_][q_], compilationContext_] :=
    compileQuery[DatabaseAnnotate[q, annotation], compilationContext]; (* Operator form *)



compileQuery[
    DatabaseAggregate[inner_, aggregation: _Rule | {___Rule} | _?AssociationQ],
    compilationContext_
] :=
    With[{compiledInner = checkQuery[compileQuery[inner, compilationContext]]},
        compiledInner @ DBAggregate[
            fullProcess["Annotation", compilationContext, compiledInner, True] @ aggregation
        ]
    ]


compileQuery[DatabaseAggregate[aggregation_][q_], compilationContext_] :=
    compileQuery[DatabaseAggregate[q, aggregation], compilationContext] (* Operator form *)



compileQuery[DatabaseOrderBy[q_, spec: $orderByFieldPattern], compilationContext_] :=
    compileQuery[DatabaseOrderBy[q, {spec}], compilationContext]

compileQuery[DatabaseOrderBy[inner_, spec: {$orderByFieldPattern..}], compilationContext_] :=
    With[{syntheticAnnotationsSpec = synthesizeAnnotations[spec, True, True]},
        compileQuery[
            syntheticFieldsDecorator[DatabaseOrderBy, syntheticAnnotationsSpec] @ inner,
            compilationContext
        ] /; syntheticAnnotationsSpec["Fields"] =!= spec
    ]

compileQuery[DatabaseOrderBy[inner_, spec: {$orderByFieldPattern..}], compilationContext_] :=
	checkQuery[compileQuery[inner, compilationContext]] @ DBOrderBy[Map[processOrderingField, spec]]

compileQuery[DatabaseOrderBy[spec_][q_], compilationContext_] :=
    compileQuery[DatabaseOrderBy[q, spec], compilationContext] (* Operator form *)



compileQuery[
    DatabaseGroupBy[q_, field: $syntheticFieldPattern, aggregation_],
    compilationContext_
] :=
    compileQuery[DatabaseGroupBy[q, {field}, aggregation], compilationContext]

compileQuery[
	DatabaseGroupBy[
		inner_,
		{},
		aggregation: _Rule | {___Rule} | _?AssociationQ
	],
	compilationContext_
] := compileQuery[DatabaseAggregate[inner, aggregation], compilationContext]

compileQuery[
    DatabaseGroupBy[
	   inner_,
       fields: {$syntheticFieldPattern..},
       aggregation: _Rule | {___Rule} | _?AssociationQ
    ],
    compilationContext_
] :=
    With[{
        (* NOTE: We do not allow anonymous / synthetic fields here, since there seems
        ** to be no scenario where they can be used constructively
        *)
        syntheticAnnotationsSpec = synthesizeAnnotations[fields, False]
        },
        compileQuery[
            syntheticFieldsDecorator[
                DatabaseGroupBy[#, aggregation]&, syntheticAnnotationsSpec
            ] @ inner,
            compilationContext
        ] /; syntheticAnnotationsSpec["Fields"] =!= fields
    ]

compileQuery[
    DatabaseGroupBy[
	   inner_,
       fields: {$syntheticFieldPattern..},
       aggregation: _Rule | {___Rule} | _?AssociationQ
    ],
    compilationContext_
] :=
    With[{compiledInner = checkQuery[compileQuery[inner, compilationContext]]},
        compiledInner @ DBGroupBy[
            Map[processField, fields],
            fullProcess["Annotation", compilationContext, compiledInner, True] @ aggregation
        ]
    ]

compileQuery[DatabaseGroupBy[fields_, aggregation_][q_], compilationContext_] :=
    compileQuery[DatabaseGroupBy[q, fields, aggregation], compilationContext] (* Operator form *)

compileQuery[
    (h: DatabaseSelectFields | DatabaseExcludeFields)[
        inner_,
        field: Except[_List]
    ],
	compilationContext_
] := compileQuery[h[inner, {field}], compilationContext]

compileQuery[
    DatabaseSelectFields[
        inner_,
        fields: {$syntheticFieldPattern...}
    ],
    compilationContext_
] :=
    With[{
        (* NOTE: We do not allow anonymous / synthetic fields here, since there seems
        ** to be no scenario where they can be used constructively
        *)
        syntheticAnnotationsSpec = synthesizeAnnotations[fields, False]
        },
        compileQuery[
            syntheticFieldsDecorator[
                DatabaseSelectFields, syntheticAnnotationsSpec
            ] @ inner,
            compilationContext
        ] /; syntheticAnnotationsSpec["Fields"] =!= fields
]

compileQuery[
    DatabaseSelectFields[
        inner_, fields: {___?(MatchQ[#, $DBTopLevelFieldNamePattern] &)}
    ],
    compilationContext_
] :=
	checkQuery[compileQuery[inner, compilationContext]] @ DBValues[Map[processField, fields]]

compileQuery[DatabaseSelectFields[fields_][q_], compilationContext_] :=
    compileQuery[DatabaseSelectFields[q, fields], compilationContext] (* Operator form *)



compileQuery[
    DatabaseExcludeFields[
        inner_, fields: {___?(MatchQ[#, $DBTopLevelFieldNamePattern] &)}
    ],
    compilationContext_
] :=
	checkQuery[compileQuery[inner, compilationContext]] @ DBRemoveSeveralFieldsFromSelected[
        Map[processField, fields]
    ]

compileQuery[DatabaseExcludeFields[fields_][q_], compilationContext_] :=
    compileQuery[DatabaseExcludeFields[q, fields], compilationContext] (* Operator form *)


compileQuery[
    DatabaseJoin[
        inner_,
        other_,
        cond: Except[_DatabaseFunction] : Automatic,
		type: "Inner" | "Left" | "Outer" | "Right" : "Inner"
    ], compilationContext_
] := 
    compileQuery[
        DatabaseJoin[inner, other, compileJoinSpec[compilationContext, inner, other, cond], type],
        compilationContext
    ]

compileQuery[DatabaseJoin[
	inner_,
	other_,
	DatabaseFunction[{left_, right_}, body_],
	"Right"
], compilationContext_] := compileQuery[
    DatabaseJoin[
        other,
        inner,
        DatabaseFunction[{right, left}, body],
        "Left"
    ],
    compilationContext
]

compileQuery[DatabaseJoin[
    inner_,
    other_,
	condition: _DatabaseFunction,
    type: "Inner" | "Left" | "Outer": "Inner"
], compilationContext_] :=
    With[{compiledInner = checkQuery[compileQuery[inner, compilationContext]]},
        With[{
            compiledOther = checkQuery[compileQuery[other, compilationContext]] @ DBEnsureAlias[
                compiledInner,
                True
            ]
            },
            compiledInner @ DBJoin[
                compiledOther
                ,
                Replace[type, {
                    "Inner" -> Inner,
                    "Left" -> Left,
                    "Outer" -> Outer
                }]
                ,
                Composition[
                    stripExpressionType,
                    checkExpressionType[DBType["Boolean"]],
                    fullProcess[
                       "Expression",
                        compilationContext,
                        {compiledInner, compiledOther}
                    ]
                ][condition]
            ]
        ]
    ]

(* Operator form *)
compileQuery[DatabaseJoin[
    other_,
    condition: _DatabaseFunction | _Rule | _List | Automatic : Automatic,
	type: "Inner" | "Left" | "Right" | "Outer": "Inner"
][q_], compilationContext_] :=
    compileQuery[DatabaseJoin[q, other, type, condition], compilationContext];



compileQuery[DatabaseSQLDistinct[inner_], compilationContext_] :=
    checkQuery[compileQuery[inner, compilationContext]] @  DBDistinct[];


compileQuery[DatabaseLimit[inner_, lim_], schema_] := With[
    {compiledInner = checkQuery[compileQuery[inner, schema]]},
    compiledInner @ DBLimit[
        fullProcess["Expression", schema, compiledInner][lim]
    ]
]

compileQuery[DatabaseOffset[inner_, off_], schema_] := With[
	{compiledInner = checkQuery[compileQuery[inner, schema]]},
	compiledInner @ DBOffset[
		fullProcess["Expression", schema, compiledInner][off]
    ]
]
DBDefError @ compileQuery
