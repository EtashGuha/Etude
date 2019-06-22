Package["Databases`Database`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`SQL`"]
PackageImport["Databases`Schema`"]


PackageScope["processOrderingField"]
PackageScope["processField"]
PackageScope["fullProcess"]
PackageScope["stripExpressionType"]


(* Dealing with ascending and descending qualifiers *)
processOrderingField[(head: DatabaseSQLAscending | DatabaseSQLDescending)[
    field: Except[DBDelayedPattern[$orderByHeadPattern]]
]] :=
    processField[field] -> (head === DatabaseSQLAscending)

processOrderingField[arg_] := processField[arg]


(* Making sure field expressions are valid  *)
processField[field: _String | $DBPrefixedFieldNamePattern] := DBPrefixedField[field]
processField[field: _DBSQLField | _DBPrefixedField] := field

DBDefError[processField, processOrderingField]


(* Full preprocessor *)
fullProcess[type_, compilationContext_, rest___][expr_] := 
    process[type, compilationContext, rest][preprocess[expr]]


(*
** Expression preprocessor. Trivial so far, but I decided to keep it still (previously
** was used to transform DatabaseSQLField to DBSQLField)
*)
preprocess[expr_] := expr
    

(*
** Preprocessor that takes into account different expression types, and also corretly
** processes annotations / aggregations
*)
process[mode_ , compilationContext_, query_?DBQueryBuilderObjectQ, rest___][expr_] :=
    process[mode, compilationContext, {query}, rest][expr]

process[
    mode:"Expression" | "Annotation" | "Relation",
    compilationContext_Association?AssociationQ,
    queries: {__?DBQueryBuilderObjectQ},
    isAggregate: True | False : False,
    opts___?OptionQ
][expr_] :=
    Switch[mode,
        "Expression",
            Replace[expr, {
                df_DatabaseFunction :> validateCompiledDBF @ DBCompileDatabaseFunction[
                    df,
                    Append[
                        compilationContext,
                        { 
                            "Aggregate" -> isAggregate || TrueQ[compilationContext["Aggregate"]],
                            "QueryBuilderObjects" -> queries
                        }
                    ]
                ]
                ,
                e_ :> DBValidateSQLExpression[e, {DBType, DBTyped}]
            }]
        ,
        "Annotation",
            With[{processor = Function[
                    pmode, 
                    process[pmode, compilationContext, queries, isAggregate]]
                },
                Replace[expr, {
                    (key_String -> assoc_?AssociationQ) /;
                        Sort[Keys[assoc]] === {"DestinationModel", "ModelMapping"} :>
						processor["Relation"][key -> assoc]
                    ,
                    (key_String -> value_) :>  key -> Composition[
                        checkAnnotationType,
                        processor["Expression"]
                    ][value]
                    ,
                    annots: {(_String -> _)...} :>
                        Map[processor["Annotation"], annots]
                    ,
                    assoc_?AssociationQ :>
                        Composition[
                            Association, processor["Annotation"], Normal
                        ][assoc]
                }]
            ],
        "Relation",
            With[{name = expr[[1]], assoc = expr[[2]]},
            With[{
                relType = singleRelationType[
                    compilationContext["Store"],
                    First[queries],
                    assoc
                ]},
            With[{
                fields = getFields[
                    assoc["ModelMapping"],
                    Or[
                        compilationContext["Store"][assoc["DestinationModel"], "ConcreteModelQ"],
                        relType =!= "ManyToOne"
                    ]
                ],
                x = DBUniqueTemporary["x"]},
                name -> DBTyped[
                    Automatic,
                    DBType[
                        If[relType === "ManyToOne", "DatabaseModelInstance", "Query"],
                        <|
                            "Deserializer" -> DBRelationFunction[assoc, relType][
                                (* 
                                ** This curried form of passing compilationContext
                                ** allows to replace it dynamically in DBRelationFunction
                                ** expression, which is necessary in some cases.
                                *)
                                Append[
                                    KeyTake[compilationContext, "Store"],
                                    "SlotQueryMap" -> <||>
                                ]
                            ],
                            "Serializer" -> Values @* Last,
                            "Dependencies" -> If[
                                MatchQ[fields, {__?StringQ}],
                                First[queries][
                                    DBGetPrefixedFields[fields]
                                ],
                                MapThread[
                                    With[{f = #2, n = name, cc = compilationContext},
                                        #1 -> {
                                            Function[
												query,
												DBCompileDatabaseFunction[
													DatabaseFunction[x, x[n][f]],
													Append[
														cc,
														{
															"QueryBuilderObjects" -> {query},
															"Aggregate" -> False
														}
													]
												]
											],
                                            #3
										}
                                    ]&,
                                    {
                                        DBGenerateUniquePropertyName /@ fields[[All, 1]],
                                        fields[[All, 2]],
                                        fields[[All, 1]]
                                    }
                                ]
                            ]
                        |>
                    ]
                ]
            ]]]
        ]

process[type_, _, rest___][arg_] :=
    DBRaise[process, "expression_processing_error", {{type}, arg}]



validateCompiledDBF[DBTyped[expr_, type_]] := DBTyped[
    DBValidateSQLExpression[expr, {   (*
        ** Exceptions for DBValidateSQLExpression, needed for inner correlated
        ** subqueries, where some DBSQLSlot-s may still be unresolved at this point.
        *)
        DBSQLSlot[_Symbol], 
        _DBPrefixedField,
        If[TrueQ[$DBQueryLightWeightCompilation], _DBUncompiledExpression, Nothing]
    }],
    type
]

(* TODO: remove this after we completely move to new compilation scheme *)
validateCompiledDBF[expr_] /; TrueQ[$DBUseOldDBFCompilationScheme] := expr

validateCompiledDBF[expr_] :=
    DBRaise[validateCompiledDBF, "untyped_expression_error", {expr}]

DBDefError @ validateCompiledDBF


stripExpressionType[DBTyped[expr_, type_]] := expr

(* TODO: remove this after we completely move to new compilation scheme *)
stripExpressionType[expr_] /; TrueQ[$DBUseOldDBFCompilationScheme] := expr

stripExpressionType[expr_] :=
    DBRaise[stripExpressionType, "untyped_expression", {expr}]

DBDefError @ stripExpressionType
