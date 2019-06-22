Package["Databases`SQL`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"] (* DBRaise, DBHandleError *)
PackageImport["Databases`Schema`"] (* DBTyped *)


PackageExport["DBWhere"]
PackageExport["DBAnnotate"]
PackageExport["DBAggregate"]
PackageExport["DBOrderBy"]
PackageExport["DBGroupBy"]
PackageExport["DBValues"]
PackageExport["DBDistinct"]
PackageExport["DBJoin"]
PackageExport["DBLimit"]
PackageExport["DBOffset"]
PackageExport["DBPrimaryKey"]

PackageScope["addTable"]
PackageScope["wrapInQuery"]


(* ============================================================================ *)
(* ============================     Helpers         =========================== *)
(* ============================================================================ *)


(*
** Low-level method. Adds a table to a list of inner tables in the query.
** By default, does not add any "public" (exported) fields from this table
** to a list of selected fields for a given query. We can also pass a field
** spec as a second argument, which can be one of All (all public fields are
** added), None (the default, no fields are added), or an arbitrary pattern
** that a field name must satisfy to be added.
**
*)
q_DBQueryBuilderObject @ addTable[table_DBQueryBuilderObject, selectedFieldsPattern_: None] :=
    Module[{sfields, filter, result = q},
        result = result @ append["Tables", table];
        sfields = table @ getSelectedFieldNames[];
        filter = Replace[selectedFieldsPattern, { None -> Except[_], All -> _ }];
        result @ transform[
            "SelectedFields",
            Function[alreadySelected, 
                Join[alreadySelected, Cases[sfields, filter]]
            ]
        ]
    ]

(*
** Add / remove the DISTINCT keyword to / from the SELECT list of the query
*)
q_DBQueryBuilderObject @ DBDistinct[flag: True | False : True] := q @ set["Distinct" -> flag]

(*
** Creates a new Query object, adding the current query as its first inner "table"
*)
q_DBQueryBuilderObject @ wrapInQuery[All] := q @ wrapInQuery[Apply[
    Alternatives,
    q @ getSelectedFieldNames[]
]]

oldQ_DBQueryBuilderObject @ wrapInQuery[selectedFieldsToExport_: None] := With[
    {q = oldQ @ DBResolveActualFields[True, False]},
    DBQueryBuilderObject @ new["Query"] @ addTable[
        q , selectedFieldsToExport
    ] @ set[
        "FieldPrefixTrie" -> q @ get["FieldPrefixTrie"]
    ] @ DBDistinct[
        (* Note: inherit DISTINCT property from inner query, by default *)
        q @ get @ "Distinct"
    ] @ set[q @ get[{"PrimaryKey", "SingleRow"}]] @ set["Schema" -> Inherited]
]

(* =============                        DBWhere                     ============ *)

(*
** Add a condition to the WHERE clause in the query
*)
q_DBQueryBuilderObject @ DBWhere[expr_] /; q @ isAggregateQuery[] :=
    (*
    ** TODO: optimize for cases when WHERE condition does not involve aggregated
    ** fields - then there is no need for an extra SELECT level.
    *)
    q @ wrapInQuery[All] @ DBWhere[expr]

q_DBQueryBuilderObject @ DBWhere[expr_] :=
    q @ append[
        "Where",
        (* Setting the full name resolution to False to allow correlated subqueries *)
        q @ resolveExpression[expr, False]
    ]


(* =============                 DBAnnotate, DBAggregate               ============ *)

q_DBQueryBuilderObject @ priorAnnotationsCanBeInlined[_] /; q @ isAggregateQuery[] := False

q_DBQueryBuilderObject @ priorAnnotationsCanBeInlined[expr:_List|_Association?AssociationQ] := 
    AllTrue[expr, q @ priorAnnotationsCanBeInlined[#]&]
    
q_DBQueryBuilderObject @ priorAnnotationsCanBeInlined[_String -> expr_] := 
    q @ priorAnnotationsCanBeInlined[expr]

q_DBQueryBuilderObject @ priorAnnotationsCanBeInlined[expr_] := 
    With[{fields = Cases[expr, DBSQLField[_], {0, Infinity}, Heads -> True]},
        ! MemberQ[
            q @ resolveExpression[fields],
            Automatic | e_ /; !FreeQ[e, _DBQueryBuilderObject]
        ]
    ]
    
q_DBQueryBuilderObject @ DBAnnotate[fieldName_String -> expr_, addToParent_: False] :=
    If[
        ! q @ priorAnnotationsCanBeInlined[expr],
            (*
            ** TODO: we can allow inlining in the future, as an option -
            ** instead of query - wrapping
            *)
            q @ wrapInQuery[All] @ DBAnnotate[fieldName -> expr, addToParent],
            (* else *)
            q @ addField[fieldName -> expr, addToParent]
    ]

q_DBQueryBuilderObject @ DBAnnotate[annotations: {__Rule}, addToParent_: False] :=
    Fold[# @ DBAnnotate[#2, addToParent]&, q, annotations]

q_DBQueryBuilderObject @ DBAnnotate[
    annotations_Association?AssociationQ,
	addToParent_: False
] := q @ DBAnnotate[ Normal @ annotations, addToParent]


q_DBQueryBuilderObject @ DBAggregate[aggs_, rest___] :=
    q @ wrapInQuery[All] @ DBAggregate[aggs, rest] /; ! q @ priorAnnotationsCanBeInlined[aggs]
    
q_DBQueryBuilderObject @ DBAggregate[
    aggregations: _Rule | {__Rule},
    allowedFields: {$DBTopLevelFieldNamePattern...} : {}
] :=
    q @ DBAggregate[ Association @ aggregations, allowedFields]

q_DBQueryBuilderObject @ DBAggregate[
    fieldSpec_?AssociationQ,
    allowedFields: {$DBTopLevelFieldNamePattern...} : {}
] :=
    Module[{ obj = q},
        obj = obj @ addField[fieldSpec];
        If[!obj @ isAggregateQuery[],
            obj = obj @ keepOnlySelectedFields[
                Join[DeleteDuplicates @ allowedFields, Keys @ fieldSpec]
            ]
        ];
        obj @ setAggregateQuery[True] @ set["PrimaryKey" -> {}]
    ]


(* =============                      group by                     ============ *)


q_DBQueryBuilderObject @ DBGroupBy[fields: {$DBTopLevelFieldNamePattern..}, aggregations_: <||>] :=
    q @ wrapInQuery[All] @ DBGroupBy[fields, aggregations] /; Or[
        q @ isAggregateQuery[],
        MemberQ[
            Map[q @ DBResolveTopLevelField[#, True, False]&, fields],
            Apply[Alternatives] @ q @ getProperFieldNames[]
        ]
    ]

q_DBQueryBuilderObject @ DBGroupBy[field: $DBTopLevelFieldNamePattern, aggregations_] :=
    q @ DBGroupBy[{field}, aggregations]


q_DBQueryBuilderObject @ DBGroupBy[fields_, aggregations: _Rule | {___Rule}] :=
    q @ DBGroupBy[fields, Association @ aggregations]


q_DBQueryBuilderObject @ DBGroupBy[
    fields: {$DBTopLevelFieldNamePattern..},
    aggregations : _?AssociationQ : <||>
] :=
    q @ DBAggregate[
        aggregations, fields
    ] @ set[<|
        "GroupBy" ->  <|
            "Having" -> {},
            "Fields" -> Map[fieldToF @ q @ resolveFieldEnsureUnique[#]&, fields]
        |>,
        "PrimaryKey" -> fields
    |>]


(* =============                       DBValues                      ============ *)


q_DBQueryBuilderObject @ DBValues[fields_List] /; q @ getType[] === "NativeTable" :=
    DBRaise[DBValues, "operation_not_supported_for_native_tables", {q}]

q_DBQueryBuilderObject @ DBValues[field: $DBTopLevelFieldNamePattern] := q @ DBValues[{field}]

q_DBQueryBuilderObject @ DBValues[fields: {___?(MatchQ[#, $DBTopLevelFieldNamePattern] &)}] :=
    q @ keepOnlySelectedFields[fields]


(* =============                        DBOrderBy                    ============ *)


q_DBQueryBuilderObject @ ord_orderBy /; q @ getType[] === "NativeTable" :=
    DBRaise[DBOrderBy, "operation_not_supported_for_native_tables", {q}]

q_DBQueryBuilderObject @ DBOrderBy[
    field: $DBTopLevelFieldNamePattern | ($DBTopLevelFieldNamePattern -> _)
] :=
    q @ DBOrderBy[{field}]

q_DBQueryBuilderObject @ DBOrderBy[
    fields: {( $DBTopLevelFieldNamePattern  | ($DBTopLevelFieldNamePattern -> _))...}
] :=
    With[{resolvedFields = Map[resolveOrderedField[q],  fields]},
        If[MemberQ[resolvedFields, DBSQLField[None, _] | (DBSQLField[None, _] -> _)],
            (* Check for annotated fields *)
            q @ wrapInQuery[All] @ DBOrderBy[fields],
            (* else *)
            (*
            ** Since there can only be one ORDER BY statement, we replace the
            ** old one with the current one, if it did exist before
            *)
            q @ set["OrderBy" ->  resolvedFields]
        ]
    ]


resolveOrderedField[query_DBQueryBuilderObject][field: $DBTopLevelFieldNamePattern] :=
    fieldToF @ query @ resolveFieldEnsureUnique[field]

resolveOrderedField[query_DBQueryBuilderObject][field_ -> ascending: (True | False)] :=
    resolveOrderedField[query][field] -> ascending

(* =============                        DBLimit                         ============ *)

q_DBQueryBuilderObject @ DBLimit[lim_] := q @ transform[
    "Limit",
    Function[
        Min[Replace[#, None -> Infinity], q @ resolveExpression[lim, False]]
    ]
] @ set["WrapOnNextOperation" -> True]


(* =============                        DBOffset                        ============ *)

q_DBQueryBuilderObject @ DBOffset[off_] := q @ transform[
    "Offset",
    Function[
        Replace[#, None -> 0] + q @ resolveExpression[off, False]
    ]
] @ set["WrapOnNextOperation" -> True]

(* =============                        join                          ============ *)


q_DBQueryBuilderObject @ DBJoin[args___] := q @ wrapInQuery[All] @ DBJoin[args] /; Or[
    q @ getType[] === "NativeTable",
    q @ isAggregateQuery[]
]

q_DBQueryBuilderObject @ DBJoin[
    other_DBQueryBuilderObject,
    joinType: Inner | Left | Outer,
    onFields: {
        ($DBTopLevelFieldNamePattern | ($DBTopLevelFieldNamePattern -> $DBTopLevelFieldNamePattern))..
    },
    opts___?OptionQ
] :=
    Module[{qfields, otherFields},
        {qfields, otherFields} = Composition[
            Transpose,
            Replace[
                #,
                (field_ -> otherField_) :> {
                    fieldToF @ q @ resolveFieldEnsureUnique[field],
                    fieldToF @ other @ resolveFieldEnsureUnique[
                        otherField,
                        <| "SearchInnerFields" -> False, "UseMainTableNameForField" -> True |>
                    ]
                },
                {1}
            ]&,
            Replace[#, field: $DBTopLevelFieldNamePattern :> (field -> field), {1}]&
        ][onFields];

        q @ DBJoin[
            other,
            joinType,
            Apply[DBSQLSymbol["And"]] @ Thread[DBSQLSymbol["="][qfields, otherFields]],
            opts
        ]
    ]

q_DBQueryBuilderObject @ DBJoin[
    other_DBQueryBuilderObject,
    joinType: Inner | Left | Outer,
    on_,
    opts___?OptionQ
] :=
    Module[{onResolved, fTable, SQLAlias = other @ DBGetName[], firstTable, collisionsResolution,
        optval, qCollisionSpec, otherCollisionSpec, prefixedOtherFieldsRules},
        (* TODO: add check that the other query doesn't have the same query alias, as
        one of the already joined tables, or if the other query has alias None *)

        (* We assume long names for the joined table, and only resolve the fields of self *)
        (* TODO: in general, we can't resolve fully- qualified fields here, because this
        can be within a subquery. But we need the query - level checks that we don't refer
        to unknown fields, at some point. Perhaps, DBAsSQL can receive an option telling that
        this is a query or a subquery, and then in the case of a query perform overall check
        that no unknown field is present *)
        onResolved = other @ resolveExpression[
            q @ resolveExpression[on, False, <|"ErrorOnFieldNotFound" -> False|>],
            False,
            <| "SearchInnerFields" -> False, "UseMainTableNameForField" -> True |>
        ];

        (* Resolve the first table *)
        fTable = If[MatchQ[onResolved, _DBUncompiledExpression],
            (* 
            ** TODO: the _DBUncompiledExpression code path is temporary, until we 
            ** implement lightweight compilation in a better always 
            *)
            None, 
            (* else *)
            Composition[
                (* TODO: I can imagine the we may want to relax this condition, if other
                tables would be involved in the ON condition. This can be in a number of
                scenarios - either tables are already inner tables from prev. joins in
                this query, or we are writing a correlated subquery and the table is from
                the outside. The question of which table is being joined, which we answer
                here, should be answered by looking at which tables are already in inner
                tables list, and which is missing. This still does not resolve the ambiguity
                with the table coming from external query, however. Perhaps, a query should
                maintain a list of tables it works with, as a dynamic variable, available to
                subqueries - this would help to solve this issue *)
                If[Length[#] =!= 1,
                    DBRaise[DBJoin, "more_than_two_tables_in_on_condition", {onResolved}],
                    (* else *)
                    First[#]
                ]&,
                DeleteCases[SQLAlias],
                DeleteDuplicates
            ] @ Cases[onResolved, DBSQLField[table_, _] :> table, {0, Infinity}]
        ];
            
        q @ addTable[
            other, All
        ] @ transform[
            "FieldPrefixTrie",
            Function[trie,
                trie @ "extend"[] @ "addParent"[other @ getFieldPrefixTrie[]]
            ]
        ] @ append[
            "Joins",
             <|
                "JoinType" -> joinType,
                "JoinedTables" -> {fTable, SQLAlias},
                "On" -> onResolved
            |>
        ] @ set[
            "PrimaryKey" -> Replace[
                (* TODO: this can be rewritten without DBGetCurrentAlias[], by just
                ** using field resolution to raw fields and then back - on the 
                way back should be resolved against the joined trie / resolver *)
                With[{qalias = q @ DBGetCurrentAlias[], otheralias = other @ DBGetCurrentAlias[]},
                    Join[
                        If[qalias === None, Identity, Map[DBPrefixedField[qalias -> #]&]][
                            q @ get["PrimaryKey"]
                        ],
                        If[otheralias === None, Identity, Map[DBPrefixedField[otheralias -> #]&]][
                            other @ get["PrimaryKey"]
                        ]
                    ]
                ],
                {x__, x__} :> None
            ]
        ]
    ]

q_DBQueryBuilderObject @ DBPrimaryKey[] := q @ DBGetPrefixedFields[q @ get["PrimaryKey"]]
