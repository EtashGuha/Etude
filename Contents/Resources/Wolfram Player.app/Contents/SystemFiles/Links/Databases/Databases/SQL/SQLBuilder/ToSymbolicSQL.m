Package["Databases`SQL`"]

PackageImport["Databases`"]
PackageImport["Databases`Common`"] (* DBRaise, DBKeyValueReverse *)
PackageImport["Databases`Schema`"]


PackageExport["DBAsSQL"]
PackageExport["DBGetPostProcessAssoc"]
PackageExport["DBResolveActualFields"]
PackageExport["DBGetFieldPostProcess"]
PackageExport["DBGetFieldsWithDependencies"]

PackageScope["asSQLWithFieldMap"]


(* =============              Ssymbolic SQL generation           ============ *)

(* TODO: resolve all fully-qualified fields (see a note in join[], currently the
** f.q.fields from the second table are not checked, and so e.g. wrong field names
** there are not caught until python execution )
*)

getOuterFields[stuff_, alias_] := Union @ Cases[
	stuff,
	DBSQLField[alias, f_] | Verbatim[Rule][DBSQLField[alias, f_], _] :> f,
	{0, Infinity},
	Heads -> True
]

simplifySQL[symsql_] := FixedPoint[
	ReplaceAll[
		{
			DBSQLSelect[fields_, DBSQLSelect[innerFields_, innerRest___] -> alias_, rest___] :>
				With[{outerFields = getOuterFields[{fields, rest}, alias]},
					With[{deleted = DeleteCases[
						innerFields,
						DBSQLField[_, f_] | Verbatim[Rule][_, f_] /; !MemberQ[outerFields, f]
					]},
						DBSQLSelect[fields, DBSQLSelect[deleted, innerRest] -> alias, rest] /;
							deleted =!= innerFields
					]
				],
			DBSQLSelect[outerFields: {___SQLF},
				DBSQLSelect[
					innerFields: {___SQLF} | {(_DBSQLField -> _)...},
					body_ -> innerAlias_
				] -> outerAlias_
			] /; ContainsExactly[outerFields[[All, 2]], innerFields[[All, 2]]] :>
       			DBSQLSelect[innerFields, body -> innerAlias],
			DBSQLSelect[
				fields_,
				DBSQLJoin[
					DBSQLSelect[innerFields_, innerRest___] -> alias_,
					lists__List
				],
				rest___
			] :> Module[{outerFields, deleted},
					outerFields = getOuterFields[
						{fields, {lists}[[All, -1]], rest},
						Append[
							Alternatives @@ {lists}[[All, 2, -1]],
							alias
						]
					];
					deleted = DeleteCases[
						Prepend[{lists}[[All, 2, 1, 1]], innerFields],
						DBSQLField[_, f_] | Verbatim[Rule][DBSQLField[_, _], f_] /; !MemberQ[outerFields, f],
						{2}
					];
					DBSQLSelect[
						fields,
						DBSQLJoin[
							DBSQLSelect[deleted[[1]], innerRest] -> alias,
							Sequence @@ ReplacePart[
								{lists},
								MapIndexed[
									{First[#2], 2, 1, 1} -> #1&,
									Rest[deleted]
								]
							]
						],
						rest
					] /; deleted =!= Prepend[{lists}[[All, 2, 1, 1]], innerFields]
				],
			DBSQLSelect[
				outerFields_,
				DBSQLSelect[innerFields: {(_DBSQLField -> _)...}, table_?StringQ -> alias_?StringQ] -> outerAlias_,
				rest___
			] /; innerFields[[All, 2]] === innerFields[[All, 1, 2]]
				:> DBSQLSelect[outerFields, table -> outerAlias, rest]
		}
	],
	symsql
]

oldQ_DBQueryBuilderObject @ asSQLRec[outermost: True | False] :=
    Module[{properFields, wh, table, getTableSQL, ord,
		processSubqueries, grby, distinct, q, selectList, offlim
        },
        If[oldQ @ getType[] === "NativeTable",
            Return[oldQ @ get @ "TableName"],
			q = oldQ @ DBResolveActualFields[False, False] @ optimizeQuery[];
        ];

        processSubqueries = Function[
            ReplaceRepeated[#, query_DBQueryBuilderObject :> query @  asSQLRec[False]]
        ];

        getTableSQL = Function[t, t @ asSQLRec[False] -> t @ DBGetName[]];
        
        properFields = q @ get["ProperFields"];
        
        selectList = q @ resolveExpression[
            Replace[
                q @ getSelectedFieldNames[],
                {
                    field_ /; KeyExistsQ[properFields, field] :> 
                        Function[#Expression -> #Alias] @ properFields[field],
                    field_ :> DBSQLField[field]
                },
                {1}
            ],
            <| "ResolveRawFieldsInExpression" -> True|>
        ];
        
        If[!outermost && selectList === {},
            selectList = {0 -> createFieldName["dummy" -> "Integer"]}
        ];

        wh = Composition[
            Replace[{
                DBSQLSymbol["And"][] :> Sequence @@ {},
                DBSQLSymbol["And"][x_] :> DBSQLWhere @ x,
                y_ :> DBSQLWhere @ y
            }],
            Apply[DBSQLSymbol["And"]]
        ] @ q @ get["Where"];

        Which[
            Length[q @ getTables[]] == 0,
                DBRaise[asSQLRec, "query_without_tables", {q}],
            Length[q @ getTables[]] == 1,
                table = getTableSQL @  First @ q @ getTables[],
            q @ get["Joins"] === {},
                DBRaise[asSQLRec, "orphan_tables_detected", {q}],
            True,
                With[{ftable = First @ First[q @ get["Joins"]]["JoinedTables"]},
                    table = DBSQLJoin[
                        getTableSQL @ q @ getInnerTable[ftable],
                        Sequence @@ Map[
                            {
                                #["JoinType"],
                                getTableSQL @ q @ getInnerTable @ Last @ #["JoinedTables"],
                                #["On"]
                            }&,
                            q @ get["Joins"]
                        ]
                    ]
                ]
        ];
        ord = If[q @ get["OrderBy"] === None,
            Sequence @@ {},
            (* else *)
            DBSQLOrderBy @ Replace[
                q @ get["OrderBy"],
                (field_ -> asc_) :> Replace[asc, {True -> DBSQLAsc, False -> DBSQLDesc}][field],
                {1}
            ]
        ];
        grby = If[q @ get["GroupBy"] === None,
            Sequence @@ {},
            (* else *)
            DBSQLGroupBy[#Fields, Replace[#Having, {} -> None]]& @ q @ get["GroupBy"]
        ];
        offlim = Replace[
            {q @ get["Offset"], q @ get["Limit"]},
            {
                {None, None} -> Sequence[],
                {off_, lim_} :> DBSQLOffsetLimit[off, lim]
            }
        ];
        distinct = If[TrueQ @ q @ get @ "Distinct", DBSQLDistinct, Identity];
        processSubqueries  @ DBSQLSelect[
            distinct @ selectList,
            table,
            wh,
            grby,
            ord,
            offlim
        ]
    ]

(*
**  Converts raw field DBRawFieldName[name, uuid] to a string name, given integer index.
*)
stringFieldName[DBRawFieldName[name_, __], 1] := name

stringFieldName[DBRawFieldName[name_, __], index_Integer] :=
	StringJoin[name, "_", ToString[index - 1]]

(*
**  Given symbolic SQL, generated with raw fields of the form DBRawFieldName[name, uuid],
**  creates a list of rules to replace those with unique srting names.
*)
createFieldNamingRules[expr_] :=
	Flatten @ MapIndexed[
		Function[{item, pos}, item -> stringFieldName[item, Last @ pos]],
		GatherBy[
			DeleteDuplicates @ Cases[
				expr,
				_DBRawFieldName,
				{0, Infinity},
				Heads->True
			],
			Replace[DBRawFieldName[name_, __] :> name]
		],
		{2}
	]

(*
**  Generates symbolic SQL, and returns symbolic SQL together with the field map that
**  maps raw fields to generated string field names.
*)
q_DBQueryBuilderObject @ asSQLWithFieldMap[] :=
    Module[{qResolved, symsql, fieldMap},
		qResolved = q @ DBResolveActualFields[];
		symsql = qResolved @ asSQLRec[True];
        fieldMap = Association @ createFieldNamingRules @ symsql;
        <|
            (* TODO: Fix the field ordering bug for symbolic SQL - level optimizer, 
            then we can set this back to True *)
            "SymbolicSQL" -> If[TrueQ[False], simplifySQL, Identity][
                ReplaceAll[symsql, fieldMap]
            ],
            "FieldMap" -> fieldMap,
            "ReverseFieldMap" -> DBKeyValueReverse @ fieldMap,
			"RawToPrefixedFieldsMap" -> qResolved @ getRawToPrefixedFieldsMap[]
        |>
    ]

(*
**  Generates and returns just the symbolic SQL
*)
q_DBQueryBuilderObject @ DBAsSQL[] := Lookup[q @ asSQLWithFieldMap[], "SymbolicSQL"]

getDependencies[annots_List]:= Map[getDependencies, annots]
getDependencies[f: DBRawFieldName[_, _, type_DBType]] :=
	With[{deps = type["Dependencies"]},
		If[MissingQ[deps], Nothing, f -> deps]
	]

q_DBQueryBuilderObject @ DBGetFieldsWithDependencies[
    removeDependingFields: True | False : True
]:=
	With[{selectedRaw = q @ getSelectedFieldNames[]},
		With[{
			dependencies = getDependencies[selectedRaw]},
			DeleteDuplicates @ Join[
                If[TrueQ[removeDependingFields],
                    DeleteCases[Alternatives @@ (q @ DBGetPrefixedFields[Keys[dependencies]])],
                    Identity
                ][q @ DBGetPrefixedFields[selectedRaw]],
				Flatten @ Values[dependencies]
			]
		]
	]

getDeserializer[q_, type_] :=
    DeleteCases[
        Composition[
            type["CursorProcessor", (q @ getSchema[])["Connection"]["Backend"]],
            Replace[type["Deserializer"], _Missing -> Identity]
        ],
        Identity
    ]

addValues[qbo_?DBQueryBuilderObjectQ] := ReplaceAll[
	qbo,
	(q_?DBQueryBuilderObjectQ)[field_] :> RuleCondition @ addValues[q] @ DBValues[field]
]


q_DBQueryBuilderObject @ DBResolveActualFields[
    keepDependingFields : True | False : False,
	resolveRecursively: True | False : True
] := With[{deps = q @ DBGetFieldsWithDependencies[!TrueQ[keepDependingFields]]},
	If[
		resolveRecursively,
		Module[{res = q, names, expr, toAdd, rules},
			names = res @ getSelectedFieldNames[];
			expr = (res @ get["ProperFields"])[[All, "Expression"]];
			rules = getDependencies[names];
			toAdd = Join @@ Map[
				Function[
					dep,
					Map[
						Function[
							f,
							DBPrefixedFieldParts[f][[-1]] -> If[
								res @ DBResolveTopLevelField[f, False] =!= None,
								DBTyped[
									DBSQLField[f],
									res @ DBGetFieldType[f]
								],
           						DBTyped[
									expr[dep[[1]]] @ DBValues[f],
									expr[dep[[1]]] @ DBGetFieldType[f]
								]
							]
						],
						dep[[2]]
					]
				],
				rules
			];
			Fold[#1 @ DBAnnotate[#2]&, res, toAdd] @ DBValues[
				DeleteDuplicates @ Join[
					DeleteCases[deps, Alternatives @@ Flatten[rules[[All, 2]]]],
					toAdd[[All, 1]]
				]
			]
		],
		q @ DBValues[deps]
	]
]
        
q_DBQueryBuilderObject @ DBGetPostProcessAssoc[] := Association @ Block[{changedFields = {}},
	Replace[
		Apply[
			Function[
				{f, type},
				(f -> Replace[
					getDeserializer[q, type],
					{
						Identity -> Identity,
						func_ :>
							func @* With[
								{x = DBUniqueTemporary["x"]},
								Replace[type["Dependencies"],
									{
										_?MissingQ :> Function[row, row[f]],
										r : {__Rule} :> fakeAssocsFunc[
											x,
											Map[
												x,
												AssociationThread[
													r[[All, 2, 2]],
													DBPrefixedField /@ r[[All, 1]]
												]
											]
										],
										any_ :> fakeAssocsFunc[
											x,
											Map[
												x,
												AssociationMap[
													DBPrefixedField,
													Map[
														Last[AppendTo[
															changedFields,
															Last[DBPrefixedFieldParts[#]]
														]]&,
														any
													]
												]
											]
										]
									}
								]
							]
					}
				]
				)
			],
			Map[# -> q @ DBGetFieldType[#] &, q @ DBGetPrefixedSelectedFields[]],
			{1}
		],
		{
			(Verbatim[Rule][f_, Identity] /; MemberQ[changedFields, Last[DBPrefixedFieldParts[f]]]) :>
				f -> With[{changed = DBPrefixedField[Last[DBPrefixedFieldParts[f]]]},
					Function[x, x[changed]]
				],
			(f_ -> Identity) :> (f -> Function[x, x[f]])
		},
		{1}
	]
]


SetAttributes[myHold, HoldFirst]	

q_DBQueryBuilderObject @ DBGetFieldPostProcess[] := With[{
	funcAssoc = q @ DBGetPostProcessAssoc[],
	row = DBUniqueTemporary["row"]},
	If[
		MatchQ[funcAssoc, <|(_ -> HoldPattern[Function[x_, x_[_]]])...|>],
		Identity,
        fakeAssocsFunc[
			row,
			Replace[
				funcAssoc,
				func_ :> myHold[func[row]],
				{1}
			]
		]
	]
]

fakeAssocsFunc[slot_, body_] := Replace[
	Function @@ {
		slot,
		ReplaceAll[
			body,
			{Association -> fakeAssoc}
		]
	},
	{
		myHold[a_] :> a,
		fakeAssoc -> Association
	},
	{0, Infinity},
	Heads -> True
]