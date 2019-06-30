Package["Databases`SQL`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"] (* DBRaise, DBHandleError, DBIterationLimiter *)


PackageExport["DBResolveTopLevelField"]
PackageScope["resolveField"]
PackageScope["resolveFieldEnsureUnique"]
PackageScope["resolveFieldName"]
PackageScope["resolveExpression"]



(* Global resolver settings *)
$defaultFieldResolverSettings =
    <|
        "SearchSelectedFields" -> True,
        "SearchInnerFields" -> True,
        "ErrorOnFieldNotFound" -> True,
        "StrictFieldResolutionMode" -> False,
        "UseMainTableNameForField" -> False,
        "ResolveRawFieldsInExpression" -> False (* relevant only for resolveExression[] *) 
    |>

getResolverSettings[resolverSettings_Association?AssociationQ] :=
    Join[$defaultFieldResolverSettings, resolverSettings]


(*
**  Resolves a prefixed or string field to low-level raw field, based mostly on the
**  query's prefix trie, rather than the query structure. Optionally, can retrict the
**  fields to only those that are currently query-resolvable. Has a "soft mode", in
**  which None is returned for unresolved field, and "hard mode", when an uresolved
**  field raises a Failure.
**
**  Note that this function does not resolve the full name of the field, which is
**  DBSQLField[table-name, raw-field] - which is what resolveField / resolveFieldEnsureUnique
**  do.
*)
q_DBQueryBuilderObject @ DBResolveTopLevelField[
    field: $DBTopLevelFieldNamePattern,
    errorOnFieldNotFound_: True,
    strictFieldResolutionMode: (True | False ) : False,
    restrictToResolvableFieldsOnly: (True | False) : True
] :=
    Module[{resolved},
        resolved = q @ getFieldPrefixTrie[] @ "fieldResolve"[
            field, strictFieldResolutionMode
        ];
        If[
            And[
                TrueQ[restrictToResolvableFieldsOnly],
                !DBMemberOf[q @ getResolvableFieldNames[]][resolved]
            ],
            resolved = None
        ];
        If[resolved === None && TrueQ[errorOnFieldNotFound],
            DBRaise[DBResolveTopLevelField, "no_field_found", {field, q}]
        ];
        resolved
    ]


(*
** Resolves the full name of a field by its short name.
** NOTE: This function does not guarantee either uniqueness or existence of the field
** being searched.
**
** High-level branch of the method: resolving high-level field
*)
q_DBQueryBuilderObject @ resolveField[
    name : $DBTopLevelFieldNamePattern,
    table: _String | All : All,
    resolverSettings: _?AssociationQ : <||>
] :=
    q @ resolveField[
        q @ DBResolveTopLevelField[
            name,
            TrueQ @ getResolverSettings[resolverSettings]["ErrorOnFieldNotFound"],
            TrueQ @ getResolverSettings[resolverSettings]["StrictFieldResolutionMode"]
        ],
        table,
        resolverSettings
    ]

(*
** Lower-level part of the method, resolving raw fields
**
** Takes: raw field name (of the form _DBRawFieldName or a string), optionally a
** specific table name, optionally resolver serrtings.
**
** Returns: the resolved field in the form {table \[Rule] field, ...}
**
** The procedure is as follows:
**  1. If searching among selected fields is enabled, search inner selected fields
**    first (note that those are guaranteed to have unique short names at all times).
**    If the field with this short name is found, return it.
**
**    2. If searching among inner fields is enabled, search inner fields for a given
**    field name, and return the result (can be empty list, one field / table, or
**    several tables). Else, return an empty list.
**
*)
q_DBQueryBuilderObject @ resolveField[None, ___] := {}
q_DBQueryBuilderObject @ resolveField[
    name: $rawFieldNamePattern,
    table: _String | All : All,
    resolverSettings: _?AssociationQ : <||>
    ] :=
    Module[{searched, settings, excludeNativeTables},
        (*
        ** TODO: convert this to a hard error. We should never face such a situation
        ** at all.
        **
        ** We use that native tables have string-named fields. Native tables must be
        ** excluded from the search, because in the new prefixing scheme no actual
        ** fields can be resolved to them. All fields *must* resolve to either {}
        ** (no fields), or a list of the form {tableName -> _DBRawFieldName, ...}.
        *)
        excludeNativeTables = Select[Not @* MatchQ[{___String}]];

        settings = getResolverSettings[resolverSettings];

        If[TrueQ[settings["SearchSelectedFields"]],
            searched = q @ searchRawField[name, table];
            If[Length[searched] == 1,
                Return[searched]
            ]
        ];

        If[TrueQ[settings["SearchInnerFields"]],
            Return[q @ searchRawField[name, table, True]]
        ];

        Return[{}]
    ]


(*
**  Similar to resolveField, but ensures that we get exactly one field out. If exactly
**  one field is found, returns tableName \[Rule] fieldName. Otherwise, raises an error
*)
q_DBQueryBuilderObject @ resolveFieldEnsureUnique[
    table_String -> name: $DBTopLevelFieldNamePattern | $rawFieldNamePattern,
    resolverSettings: _?AssociationQ : <||>
] :=
    q @ resolveFieldEnsureUnique[name, table, resolverSettings];

q_DBQueryBuilderObject @ resolveFieldEnsureUnique[
    name: $DBTopLevelFieldNamePattern | $rawFieldNamePattern,
    table: _String | All : All,
    resolverSettings: _?AssociationQ : <||>
] :=
    With[{resolved = q @ resolveField[name, table, resolverSettings]},
        Which[
            Length[resolved] == 0,
                If[TrueQ[getResolverSettings[resolverSettings]["ErrorOnFieldNotFound"]],
                    DBRaise[resolveFieldEnsureUnique, "no_field_found", {q, name}],
                    (* else *)
                    None
                ],
            Length[resolved] > 1,
                DBRaise[resolveFieldEnsureUnique, "multiple_fields_found", {q, name, resolved}],
            True,
                If[TrueQ[getResolverSettings[resolverSettings]["UseMainTableNameForField"]],
                    Replace[(_ -> field_) -> (q @ DBGetName[] -> field)],
                    (* else *)
                    Identity
                ] @ First @ resolved
        ]
    ]


(*
** TODO: make sure that this method is still neded, given that now DBResolveTopLevelField
** takes an optional argument to restrict fields to only resolvable ones.
*)
(*
** A handy shortcut. NOTE that this is essentially the same as DBResolveTopLevelField,
** except this one also checks that the field is indeed available in the query, after
** resolving the raw field name.
**
** Resolves the field (make sure it exists in the query), and returns back its short
** name. For the string 'name', it will then be returned back (if found), while for
** the prefixed field, the real low-level field name will be returned.
*)
q_DBQueryBuilderObject @ resolveFieldName[
    name: $DBTopLevelFieldNamePattern,
    table: _String | All : All,
    resolverSettings: _?AssociationQ : <||>
] := getFieldName @ q @ resolveFieldEnsureUnique[name, table, resolverSettings]


(*
** Resolves a subquery <sub> against the parent query <q>.
*)
q_DBQueryBuilderObject @ resolveSubquery[
    sub_DBQueryBuilderObject,
    resolverSettings_Association?AssociationQ,
    subqueryLevel: _Integer?NonNegative
] :=
    Fold[
        # @ transform[
            #2,
            (* Resolve only non - fully - qualified or generated fields,
            ** since we are resolving against parent query - all other fields
            ** should have been resolved earlier.
             *)
            Function[
                val,
                q @ resolveExpression[val, False, True, resolverSettings, subqueryLevel + 1]
            ]
        ]&,
        sub,
        {"ProperFields", "Tables", "Joins", "Where", "GroupBy", "OrderBy", "Limit", "Offset"}
    ]


(*
**  Safety net for expression resolver against infinite iteration.
*)
$exprssionResolverIterationLimit = 30

(*
**  Resolves all fields of the form DBSQLField[fieldName] to the full names of the form
**  DBSQLField[table, fieldName], repeatedly until all short field names are resolved to
**  full names.
**
**  NOTE: in some cases (e.g. for joins or subqueries), we may want to keep some
**  fields unresolved in an expression. We do that by passing the flag
**  resolveFullNames = False. However then, other queries (containing that table
**  'tableName', to which the field belongs), must resolve such long field names.
**
**  TODO: Also in general, making sure that all field names eventually get resolved /
**  validated, seems a right thing to do.
*)

q_DBQueryBuilderObject @ resolveExpression[
    expr_,
    resolveFullNames: True | False : True,
    resolveGeneratedFields: True | False: True,
    resolverSettings: _?AssociationQ : <||>,
    subqueryLevel: _Integer?NonNegative : 0
] := With[{rsettings = getResolverSettings[resolverSettings]},
    FixedPoint[
        Composition[
            ReplaceAll[
                (*
                ** We need RuleCondition / Trott-Strzebonski technique here, because
                ** some of the expressions involving fields, may be inside assocs.
                *)
                {
                    (* Subquery, resolved against parent query. Normally, this action
                    ** can only be non-trivial for correlated subqueries, which after
                    ** being resolved themselves, still can have unresolved fields
                    ** from parent (enclosing) queries.
                    *)
                    sub_DBQueryBuilderObject :> RuleCondition[
                        q @ resolveSubquery[sub, rsettings, subqueryLevel]
                    ]
                    ,
                    (* outerRef is the low-level mechanism to refer to the enclosing
                    ** query, for a subquery. Note that it is only used when query is
                    ** constructed bottom-up, in an interpreted fashion, using low-level
                    ** SQLBuilder primitives. E.g. DatabaseFunction uses a different
                    ** mechanism, simply keeping some slots unresolved until the parent
                    ** query gets compiled, which resolves thpse.
                    *)
                    outer: DBSQLOuterRef[e_] :> RuleCondition @ If[subqueryLevel > 0,
                        q @ resolveExpression[
                            e,
                            resolveFullNames,
                            resolveGeneratedFields,
                            rsettings,
                            subqueryLevel - 1
                        ],
                        (* else *)
                        outer
                    ]
                    ,
                    (* 1 - arg DBSQLField, non-fully-qualified fields *)
                    DBSQLField[field_] /; Or[
                        MatchQ[field, $DBTopLevelFieldNamePattern],
                        And[
                            TrueQ[rsettings["ResolveRawFieldsInExpression"]],
                            MatchQ[field, $rawFieldNamePattern]
                        ]
                    ] :>
                        With[
                            {resolved =
                                q @ resolveFieldEnsureUnique[field, All, rsettings]},
                            fieldToF[resolved] /; resolved =!= None
                        ]
                    ,
                    (*
                    ** Resolving native fields to themselves. TODO: check that this is
                    ** still needed with the new aliasing / field resolution mechanism
                    *)
                    f: DBSQLField[table_String, field_String] :> With[{tbl = q @ getInnerTable[table]},
                        f /; And[
                            tbl @ getType[] === "NativeTable",
                            MemberQ[tbl @ getSelectedFieldNames[], field]
                        ]
                    ]
                    ,
                    (* Fully qualified fields, usually only need resolving to check *)
                    DBSQLField[table_String, field: $DBTopLevelFieldNamePattern] /; resolveFullNames  :>
                        With[
                            {resolved =
                                q @ resolveFieldEnsureUnique[field, table, rsettings]},
                            fieldToF[resolved] /; resolved =!= None
                        ]
                    ,
                    (* Fields, resolved as annotated / created by the parent query itself.
                    ** This rule resolves them into fully-qualified fields with specific
                    ** table name / alias.
                    ** TODO: analyze, whether it ever makes sense to not resolve them in
                    ** one step to actual fully-qualified field in  resolveFieldEnsureUnique
                    *)
                    DBSQLField[None, field : $rawFieldNamePattern ] /; TrueQ[resolveGeneratedFields] :>
                        RuleCondition[
                            With[{fval = (q @ get["ProperFields"])[field]},
                                If[MissingQ[fval],
                                    DBRaise[
                                        resolveExpression,
                                        "annotated_field_expression_not_found",
                                        {q, field}
                                    ]
                                ];
                                fval["Expression"]
                            ]
                        ]
                }
            ],
            DBIterationLimiter[$exprssionResolverIterationLimit, resolveExpression]
        ],
        expr
    ]
]