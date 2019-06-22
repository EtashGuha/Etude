Package["Databases`SQL`"]

PackageImport["Databases`"]
PackageImport["Databases`Common`"]


PackageExport["DBQueryBuilderObject"]
PackageExport["DBQueryBuilderObjectQ"]
PackageExport["DBRawFieldName"]
PackageExport["DBRawFieldID"]
PackageExport["$DBFieldNamePattern"] (* TODO: to be removed  in favor of $DBTopLevelFieldNamePattern *)
PackageExport["$DBTopLevelFieldNamePattern"]
PackageExport["$DBPrefixedFieldNamePattern"]
PackageExport["$DBRelaxedTopLevelFieldNamePattern"]
PackageExport["DBCreateTable"]  (* Constructor for a DBQueryBuilderObject  *)
PackageExport["DBUncompiledExpression"]

PackageScope["$plainFieldPattern"]
PackageScope["$prefixedFieldLookupPattern"]
PackageScope["addCoreMethods"]
PackageScope["getDataAssoc"]
PackageScope["get"]
PackageScope["set"]
PackageScope["transform"]
PackageScope["append"]
PackageScope["inherit"]

PackageScope["$rawFieldNamePattern"]

$plainFieldPattern = _String

$prefixedFieldLookupPattern = {(_String | {_String, _Integer}).., _String}

$DBFieldNamePattern = $plainFieldPattern | $prefixedFieldLookupPattern


$rawFieldNamePattern = _DBRawFieldName

$DBPrefixedFieldNamePattern = _ -> inner_ /; MatchQ[inner, _String | $DBPrefixedFieldNamePattern ]

$DBTopLevelFieldNamePattern = _String | DBPrefixedField[_String | $DBPrefixedFieldNamePattern ]

$DBRelaxedTopLevelFieldNamePattern = $DBPrefixedFieldNamePattern | $DBTopLevelFieldNamePattern

(* TODO: To be rewritten and moved to Common module, as a part of immutable object machinery *)

$blank = Repeated[_, {0, 1}]


DBRawFieldName /: Databases`Schema`DBType[DBRawFieldName[_, _, type_]] := type


DBRawFieldID[DBRawFieldName[_, id_, _]] := id

DBRawFieldID[fields: {___DBRawFieldName}] := Map[DBRawFieldID, fields]

DefError @ DBRawFieldID


addCoreMethods[typeSymbol_Symbol] :=
    Module[{},
        typeSymbol[o_, meta: $blank] @ getDataAssoc[] := o;

        (q:typeSymbol[_, meta: $blank]) @ set[field_String -> sub_ -> val_] :=
            Module[{assoc = q @ getDataAssoc[]},
                If[!KeyExistsQ[assoc, field],
                    DBRaise[
                        set,
                        "can_not_add_new_fields_to_" <> ToString[typeSymbol],
                        {q, field}
                    ]
                ];
                assoc[field][sub] = val;
                typeSymbol[assoc, meta]
            ];

        (q:typeSymbol[_, meta: $blank]) @ set[kv: (_String -> _) | _?AssociationQ] :=
            Module[{assoc = q @ getDataAssoc[], badFields},
                badFields = Complement[Keys @ Association @ kv, Keys @ assoc ];
                If[badFields =!= {},
                    DBRaise[
                        set,
                        "can_not_add_new_fields_to_" <> ToString[typeSymbol],
                        {q, badFields}
                    ]
                ];
                typeSymbol[Append[assoc, kv], meta]
            ];

        q_typeSymbol @ get[prop_String -> sub_] :=
            With[{pval = q @ get[prop]},
                Which[
                    !AssociationQ[pval],
                        DBRaise[get, "property_is_not_an_assoc", {q, prop -> sub}],
                    !KeyExistsQ[pval, sub],
                        DBRaise[get, "invalid_query_subproperty", {q, prop -> sub}],
                    True,
                        pval[sub]
                ]
            ];

        q_typeSymbol @ get[prop_String] :=
            If[MissingQ[#],
                DBRaise[get, "invalid_query_property", {q, prop}],
                (* else *)
                #
            ] & @ Lookup[q @ getDataAssoc[], prop];

		q_typeSymbol @ get[prop_List] := AssociationMap[q @ get[#]&, prop];

        q_typeSymbol @ transform[field_String -> sub_,  f_] :=
            q @ set[field -> sub -> f @ q @ get[field -> sub]];

        q_typeSymbol @ transform[field_String, f_] :=
            q @ set[field -> f @ q @ get[field]];

        q_typeSymbol @ append[field_String, val_] :=
            q @ transform[field, Append[val]];

        q_typeSymbol @ append[
            field_String -> sub_,
            val_,
            type : List | Association | Automatic : Automatic
        ] :=
            Module[{contanerType, getType, test, expectedType},
                getType = Replace[{_Rule -> Association, _ -> List}];
                test = <|List -> ListQ, Association -> AssociationQ|>;
                contanerType = If[type =!= Automatic, type, getType[val]];
                q @ transform[
                    field,
                    Function[assoc,
                        Which[
                            !AssociationQ[assoc],
                                DBRaise[append, "field_value_must_be_an_assoc", {q, field}],
                            !KeyExistsQ[assoc, sub],
                                Append[assoc, sub -> contanerType[val]],
                            CompoundExpression[
                                expectedType = If[type === Automatic, Head @ assoc[sub], type];
                                !test[expectedType][assoc[sub]]
                            ],
                                DBRaise[
                                    append,
                                    StringJoin[
                                        "subfield_value_must_be_a_",
                                        ToString[expectedType]
                                    ],
                                    {q, field -> sub}
                                ],
                            True,
                                Append[assoc, sub -> Append[assoc[sub], val]]
                        ]
                    ]
                ]
            ];

		q_typeSymbol @ inherit[a_List] := q @ set[q @ get[a]];

		q_typeSymbol @ inherit[a_] := q @ inherit[{a}];
    ]


DBQueryBuilderObjectQ[DBQueryBuilderObject[assoc_Association?AssociationQ]] :=
    (* Make sure DBQueryBuilderObject has been already constructed and its alias generated *)
    MatchQ[Unevaluated @ assoc, <|___, "Alias" ->_String, ___|>]

DBQueryBuilderObjectQ[_] = False


(* These definitions should be added first *)

q_DBQueryBuilderObject[(fst: f_[___])[rest_]] := q[fst][rest]

$operationsNeedWrappingPattern = Alternatives[
    DBWhere,
    DBAnnotate,
    DBAggregate,
    DBGroupBy, (* Not sure *)
    DBJoin,
    DBOrderBy
]

q_DBQueryBuilderObject[oper: $operationsNeedWrappingPattern[args___]] /;
    TrueQ[q @ get @ "WrapOnNextOperation"] :=
    q @ wrapInQuery[All] @ oper


DBQueryBuilderObject @ (t: _table | _new)[arg_] := DBQueryBuilderObject[t][arg]


addCoreMethods[DBQueryBuilderObject] (* Generates definitions for get, set, append, transform *)
