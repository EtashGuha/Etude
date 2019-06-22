(* Wolfram Language package *)
Package["Databases`SQL`"]

PackageImport["Databases`"]
PackageImport["Databases`Common`"] (* DBRaise *)


PackageExport["SAFullQuery"]
PackageExport["SAQueryString"]

PackageExport["DBSQLAlchemyQueryGenerate"]
PackageExport["$SQLAlchemyQueryAsString"]
PackageExport["DBSQLAlchemyQueryQ"]



DBSQLAlchemyQueryQ[_String | _SAFullQuery | _SAQueryString] := True
DBSQLAlchemyQueryQ[_] := False


(* ============================================================================ *)
(* ============================================================================ *)
(* ======== Translator from SymbolicSQL to SymbolicSQLAlchemy       =========== *)
(* ============================================================================ *)
(* ============================================================================ *)


(* ============================================================================ *)
(* ========================        Query splitter      ======================== *)
(* ============================================================================ *)

(* We are only interested in aliased tables, since only those require
special treatment for translation to SQLAlchemy *)

collectAliasedTables[None] := {}

collectAliasedTables[nt: (nativeTable_String -> alias_String)] := {nt}



collectAliasedTables[(table: _DBSQLSelect | _DBSQLJoin) -> alias_String] :=
    Append[collectAliasedTables[table], table -> alias]

collectAliasedTables[t: DBSQLSelect[fields_, table_, w: _DBSQLWhere | None : None, ___]] :=
    Join[
        collectAliasedTablesFromFields[fields],
        collectAliasedTables[table],
        collectAliasedTables[w]
    ]

collectAliasedTables[DBSQLJoin[table_, rest___]] :=
    Flatten @ Join[
        collectAliasedTables[table],
        Cases[
            {rest}, 
            {jtype_, t_, cond_} :> collectAliasedTables[t]
        ]
    ]

collectAliasedTables[DBSQLWhere[condition_]] := collectAliasedTables[condition]

collectAliasedTables[
    expr: (DBSQLSymbol[_][___])
] :=
    Flatten[Map[collectAliasedTables, List @@ expr]]

collectAliasedTables[arg_] := {}

DBDefError @ collectAliasedTables


collectAliasedTablesFromFields[fields_List] :=
    Flatten @ Map[collectAliasedTablesFromFields, fields]
    
collectAliasedTablesFromFields[field_ -> alias_] := 
    collectAliasedTablesFromFields[field]
    
collectAliasedTablesFromFields[expr_] := collectAliasedTables[expr]
    
DBDefError @  collectAliasedTablesFromFields   
    

transformRules[rules_List] := transformRules[{}, rules]

transformRules[transformed_List, {}] := transformed

transformRules[transformed_List, {lhs_ -> rhs_String, rest___}] :=
    With[{appended = Join[transformed,
            {
                (lhs -> rhs) -> SAAlias[rhs],
                DBSQLField[rhs, field_] :> DBSQLField[SAAlias[rhs], field]
            }
        ]},
        transformRules[appended, ReplaceAll[{rest}, appended]]
    ]

DBDefError @ transformRules


extractQueryAliasDependencies[query_] :=
	DeleteDuplicates @ Cases[query, DBSQLField[al_, _] :> al, {0, Infinity}]

DBDefError @ extractQueryAliasDependencies


(*
**  Takes a list of query -> alias rules, and reorders it so that subsequent queries
**  can only refer to / depend on aliases of queries, defined previously in the list.
*)

(*  Start with a list of rules on the form {(query -> alias_String_...}, and transform
** that to a list of the form
**
**  {{queryIndex_Integer, queryAlias_String, dependencies: {___String}}...}
*)
tableRulesOrdering[{}] = {}

tableRulesOrdering[tableAliasRules: {(_ -> _String)..}] :=
	tableRulesOrdering @ MapIndexed[
		{First @ #2, Last @ #, extractQueryAliasDependencies @ #}&,
		tableAliasRules
	]

(* The actual work starts here *)
tableRulesOrdering[deps: {{_Integer, _String, {___String}}..}] :=
    tableRulesOrdering[deps, {}]

(* All dependencies processed, return the list of ordered indexes *)
tableRulesOrdering[{}, accum_] := accum

(* Main branch with recursion *)
tableRulesOrdering[deps_List, accum_List] :=
	With[{indep = FirstCase[deps, {_, _, {}}]}, (* Locate first remaining entry with no dependencies*)
		If[MissingQ[indep],
            (*
            ** This would mean either that there are tables referenceed which are not
            ** present in the query, or that there are circular dependencies
            *)
			DBRaise[tableRulesOrdering, "can_not_order_dependencies", {deps, accum}]
		];
		With[{index = First @ indep, name = indep[[2]]},
            (*  Add this entry's index to accumulated list of indices, and remove
            ** entry's name from dependencies in the rest of the entries. Then call
            ** the function again on the resulting arguments
            *)
			tableRulesOrdering[
				Map[MapAt[DeleteCases[name], #, -1]&, DeleteCases[deps, indep]],
				Append[accum, index]
			]
		]
	]

DBDefError @ tableRulesOrdering


reorderTableRules[tableAliasRules: {(_ -> _String)...}] :=
	tableAliasRules[[tableRulesOrdering[tableAliasRules]]]

DBDefError @ reorderTableRules


queryRules = Composition[
    transformRules,
    reorderTableRules,
    (* 
    ** TODO: perhaps remove DeleteDuplicates after we simplify the query 
    ** generation, should be no duplicate aliases ideally 
    *)
    DeleteDuplicates, 
    collectAliasedTables
]


getSplitQuery[query_] :=
    With[{rules = queryRules @ query},
        {
            Composition[
                Map[Replace[((lhs_ -> _) -> a_) :> (lhs -> a)]],
                DeleteCases[_RuleDelayed]
            ] @ rules
            ,
            query //. rules
        }
    ]

DBDefError @ getSplitQuery

(* ============================================================================ *)
(* ===============  SymbolicSQLAlchemy query generator      =================== *)
(* ============================================================================ *)


convertToHLSA[a_SAAlias] := a

convertToHLSA[
    DBSQLSelect[
        selectList_,
        table:_:None,
        w: _DBSQLWhere | None: None,
        gb: _DBSQLGroupBy | None : None,
        ob:_DBSQLOrderBy | None : None,
		ol:_DBSQLOffsetLimit | None : None
    ]
] :=
    Module[{result},
        result = "SASelect"[convertSelectListToHLSA @ selectList];
        If[table =!= None,
            result = "SASelectFrom"[result, convertToHLSA @ table]
        ];
        If[w =!= None,
            result = "SAWhere"[result, convertToHLSA @ w]
        ];
        If[gb =!= None,
            result = "SAGroupBy"[
               result, convertToHLSA @ gb
            ]
        ];
        If[ob =!= None,
            result = "SAOrderBy"[result, convertToHLSA @ ob]

        ];
        If[Head[selectList] === DBSQLDistinct,
            result = "SADistinct"["query", result]
        ];
		If[ol =!= None,
			result = "SAOffsetLimit"[result, convertToHLSA @ ol]
		];
        result
    ]

convertToHLSA[DBSQLJoin[a_, joined__]] :=
    Fold[
        Function[{table, spec},
          Replace[First @ spec,
            {
                Inner -> "SAJoin",
                j: Left | Outer ->
                   Function["SAOuterJoin"[#1, #2, Replace[j, {Left -> "Left", Outer -> "Full"}], #3]]
            }
          ][table, spec[[2]], convertExprToHLSA[Last @ spec]]
        ],
        a,
       {joined}
    ]

convertToHLSA[DBSQLWhere[cond_]] := convertExprToHLSA[cond]

convertToHLSA[DBSQLGroupBy[fields_List,  having_: None]] :=
    If[having === None,
        convertFieldListToHLSA[fields],
        (* else *)
        Sequence @@ {
            convertFieldListToHLSA[fields],
            "SAHaving"[convertExprToHLSA[having]]
        }
    ]

convertToHLSA[DBSQLGroupBy[field_,  having_: None]] :=
    convertToHLSA[DBSQLGroupBy[{field}, having]]

convertToHLSA[DBSQLOrderBy[fields_]] :=
    convertOrderByFieldsListToHLSA[fields]

convertToHLSA[DBSQLOffsetLimit[off_, lim_]] :=
	Sequence @@ (convertExprToHLSA /@ {off, lim})

DBDefError[convertToHLSA]


(*  TODO: this problem with fully-qualified DBDataToAST reflects the problem in design,
**  namely that functions that need it (currently DatabaseCreate, DatabaseInsert),
**  are not fully integrated into symbolic compilation pipeline.
**
**  Change this when we add support for DDL, and also support for VALUES as an sql
**  primitive.
*)

convertFieldListToHLSA[fields_List] :=
    Replace[fields, {
        f: _String | _DBSQLField :> convertExprToHLSA[f],
        f_ :> DBRaise[convertFieldListToHLSA, "invalid_field_expression", {f}]
        },
        {1}
    ]

DBDefError @ convertFieldListToHLSA


convertOrderByFieldsListToHLSA[fields_List] :=
    Replace[fields, {
            DBSQLDesc[f: _String | _DBSQLField ] :> "SADescending" @ convertExprToHLSA[f],
            DBSQLAsc[f: _String | _DBSQLField ] :> "SAAscending" @ convertExprToHLSA[f],
            f: _String | _DBSQLField :> convertExprToHLSA[f],
            f_ :> DBRaise[convertOrderByFieldsListToHLSA, "invalid_field_expression", f]
        },
        {1}
    ]

DBDefError @ convertOrderByFieldsListToHLSA


convertSelectListToHLSA[Star] := {}

(* Distinct taken care of in the code for SELECT conversion *)
convertSelectListToHLSA[DBSQLDistinct[fields_]] := convertSelectListToHLSA[fields]

convertSelectListToHLSA[fields_List] := Map[convertSelectFieldToHLSA, fields]

convertSelectListToHLSA[field_] := convertSelectListToHLSA[{field}]

DBDefError @ convertSelectListToHLSA


convertSelectFieldToHLSA[field_ -> alias_String] :=
    "SALabel"[convertSelectFieldToHLSA[field], alias]

convertSelectFieldToHLSA[field_] := 
    convertExprToHLSA[field]

DBDefError @ convertSelectFieldToHLSA

With[
    {dispatch = Dispatch @ {
        DBSQLSymbol["DeleteDuplicates"][e_]        :> "SADistinct"["aggregation", e],
        DBSQLSymbol["-"][a_, b_]                   :> "SASubtract"[a, b],
        DBSQLSymbol["-"][a_]                       :> "SAMinus"[a],
        DBSQLSymbol["+"][args___]                  :> "SAAdd"[args],
        DBSQLSymbol["*"][args___]                  :> "SAMultiply"[args],
        DBSQLSymbol["/"][args___]                  :> "SADivide"[args],
        DBSQLSymbol["%"][args___]                  :> "SAMod"[args],
        DBSQLSymbol["^"][args___]                  :> "SAPower"[args],
        DBSQLSymbol[">"][args___]                  :> "SAGreater"[args],
        DBSQLSymbol[">="][args___]                 :> "SAGreaterEqual"[args],
        DBSQLSymbol["<"][args___]                  :> "SALess"[args],
        DBSQLSymbol["<="][args___]                 :> "SALessEqual"[args],
        DBSQLSymbol["="][args___]                  :> "SAEqual"[args],
        DBSQLSymbol["<>"][args___]                 :> "SAUnequal"[args],
        DBSQLSymbol["Regexp"][a_, b_]              :> "SARegexp"[a, b, False],
        DBSQLSymbol["IRegexp"][a_, b_]             :> "SARegexp"[a, b, True],
        DBSQLSymbol[name_String][arg___]           :> StringJoin["SA", name][arg],
        DBSQLDate[args___]                         :> "SADate"[args],
        DBSQLDateTime[args___]                     :> "SADateTime"[args],
        DBSQLTime[args___]                         :> "SATime"[args],
        DBSQLSecondsToTimeQuantity[arg_]           :> "SASecondsToTimeQuantity"[arg],
		DBSQLBoolean[args_]                        :> "SABoolean"[args],
        DBSQLField[args___]                            :> "SATableField"[args],
        s_DBSQLSelect                              :> convertToHLSA @ s
    }},
    convertExprToHLSA[expr_] :=
        Replace[expr, dispatch, {0, Infinity}]
]

DBDefError @ convertExprToHLSA

(* TODO: This function really seems to be longer needed, since now the main
** DBSQLAlchemyQueryGenerate seems to handle literal queries fine, and we can
** add the fullQuery also there (to be used in DBInsert, until it gets redone
** to integrate with the symbolic query generation machinery)
*)

convertRules[rules_List] := Map[convertRules, rules];
convertRules[table_String -> SAAlias[a_String]] :=
	"SACreateAlias"[table, a];
convertRules[q_ -> SAAlias[a_String]] :=
	"SACreateAlias"[convertToHLSA @ q, a];

symSQLToHLSA[query_] :=
    Module[{rules, stmt},
        {rules, stmt} = getSplitQuery[query];
        SAFullQuery[Sequence @@ convertRules[rules], convertToHLSA  @ stmt]
    ]

DBDefError @ symSQLToHLSA


$SQLAlchemyQueryAsString = False


DBSQLAlchemyQueryGenerate[query_?DBQueryBuilderObjectQ] :=
    DBSQLAlchemyQueryGenerate[query @ DBAsSQL[]]

DBSQLAlchemyQueryGenerate[query_?DBSymbolicSQLQueryQ] :=
    If[TrueQ[$SQLAlchemyQueryAsString],
        SAQueryString,
        Identity
    ] @ symSQLToHLSA[query]
