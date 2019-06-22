Package["Databases`Database`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`Schema`"] (* DBSchemaGet *)
PackageImport["Databases`SQL`"]


PackageExport["DBQueryToSymbolicSQL"]
PackageExport["DBQueryToSQLAlchemy"]
PackageExport["DBQueryToQueryBuilderObject"]
PackageExport["DBGenerateUniquePropertyName"]
PackageExport["$DBQueryLightWeightCompilation"]


$DBQueryLightWeightCompilation = False


$ctr = 0

DBGenerateUniquePropertyName[baseName_String:"synthetic_prop_"] :=
    StringJoin[baseName, ToString[$ctr++]]


(*
** Top - level / module-exported query compiler.
**
** It behaves in the following way:
**    - If the query in question is already a DBQueryBuilderObject, it returns it as is and
**      does not attempt to check it against the provided store.
**    - If the store is not provided or Automatic is passed for it, then it
**      attempts to use the current global one, as per DBSchemaGet[]
*)
DBQueryToQueryBuilderObject[store_DatabaseStore] :=
    Function[query, DBQueryToQueryBuilderObject[query, store]]

DBQueryToQueryBuilderObject[query_?DBUncompiledQueryQ] :=
    DBQueryToQueryBuilderObject[query, Automatic]
(*
** TODO: perhaps, here, if store is explicitly provided, check that it
** is compatible with the DBQueryBuilderObject
*)
DBQueryToQueryBuilderObject[query_?DBQueryBuilderObjectQ, _] := query

DBQueryToQueryBuilderObject[query_, schema : _RelationalDatabase]:=
    DBQueryToQueryBuilderObject[query, storeCheck @ DatabaseStore @ schema]

DBQueryToQueryBuilderObject[query_, Automatic] :=
    DBQueryToQueryBuilderObject[query, Quiet @ DBSchemaGet[]]

addValues[qbo_?DBQueryBuilderObjectQ] := ReplaceAll[
    qbo,
	(q_?DBQueryBuilderObjectQ)[field_] :> RuleCondition @ addValues[q] @ DBValues[field]
]

DBQueryToQueryBuilderObject[query: _?DBInertQueryQ | _?StringQ, store_DatabaseStore] :=
    checkUnboundSlots @ addValues @ compileQuery[
        query, 
        <|"Store" -> store, "SlotQueryMap" -> <||>|>
    ]

DBDefError @ DBQueryToQueryBuilderObject


(*
**  Top - level function to perform both compilation steps, from inert Database*
**  query via DBQueryBuilderObject to symbolic SQL
*)
DBQueryToSymbolicSQL[
    query_?(DBUnevaluatedPatternCheck[DBUncompiledQueryQ]),
    store: _DatabaseStore | Automatic : Automatic
] := DBQueryToQueryBuilderObject[query, store] @ DBAsSQL[]

DBDefError @ DBQueryToSymbolicSQL


DBQueryToSQLAlchemy[q_?DBQueryObjectQ, rest___] :=
    DBQueryToSQLAlchemy[q["QueryBuilderInstance"], rest]

DBQueryToSQLAlchemy[q_?DBSQLAlchemyQueryQ, _] := q

DBQueryToSQLAlchemy[query: _?DBSymbolicSQLQueryQ | _?DBQueryBuilderObjectQ, _] :=
    DBSQLAlchemyQueryGenerate[query]

DBQueryToSQLAlchemy[
    query_?(DBUnevaluatedPatternCheck[DBUncompiledQueryQ]),
    store: _DatabaseStore
] :=  DBQueryToSQLAlchemy[DBQueryToSymbolicSQL[query, store], store]

DBDefError @  DBQueryToSQLAlchemy
