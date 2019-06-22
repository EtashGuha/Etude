Package["Databases`Database`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`Schema`"]
PackageImport["Databases`SQL`"]


PackageScope["checkQuery"]
PackageScope["checkQueries"]
PackageScope["checkBoolean"]
PackageScope["checkSlotMap"]
PackageScope["checkUnboundSlots"]
PackageScope["checkExpressionType"]
PackageScope["checkAnnotationType"]


(*
**  A post - check to ensure that the (intermediate) compilation result is a
**  valid DBQueryBuilderObject.
*)
checkQuery[query_?DBQueryBuilderObjectQ] := query
checkQuery[arg_] := DBRaise[checkQuery, "not_a_valid_query", {arg}]
DBDefError @ checkQuery

checkBoolean[flag: True | False] := flag
checkBoolean[arg_] := DBRaise[checkBoolean, "boolean_value_expected", {arg}]
DBDefError @ checkBoolean

checkQueries[queries_List] := Map[checkQuery, queries]
checkQueries[arg_] := DBRaise[checkQueries, "not_a_valid_query_list", {arg}]
DBDefError @ checkQueries

checkSlotMap[
    map: <|(DBSQLSlot[_Symbol] -> _?(DBUnevaluatedPatternCheck[DBQueryBuilderObjectQ]))...|>
] := map
checkSlotMap[arg_] := 
    DBRaise[checkSlotMap, "an_assoc_with_sqlslot_to_qbo_expected", {arg}]
DBDefError @ checkSlotMap        
        
checkAnnotationType[typed: DBTyped[_, type_?(DBUnevaluatedPatternCheck[DBTypeQ])]] :=
    If[TrueQ[type["Depth"] == 0 || TrueQ[$DBQueryLightWeightCompilation]],
        typed,
        (* else *)
        DBRaise[checkAnnotationType, "cannot_annotate_with_a_nonscalar", {typed}]
    ]
checkAnnotationType[DBTyped[_, t_]] :=
    DBRaise[checkAnnotationType, "invalid_annotation_type", {t}]    
DBDefError @  checkAnnotationType
   
checkUnboundSlots[expr_] :=
    With[{unbound = DeleteDuplicates @ Cases[expr, _DBSQLSlot, Infinity, Heads -> True]},
        If[unbound =!= {},
            DBRaise[
                checkUnboundSlots,
                "unbound_slots",
                {},
                <| "Expression" -> expr, "UnboundSlots" -> unbound |>
            ]
        ];
        expr
    ]
DBDefError @ checkUnboundSlots

(* TODO: remove this after we completely move to new compilation scheme *)
checkExpressionType[_][expr_] /; TrueQ[
    $DBQueryLightWeightCompilation || $DBUseOldDBFCompilationScheme
] := expr

checkExpressionType[expectedType_][
    typed: DBTyped[_, type_]
] /; DBTypeContainsQ[expectedType, type] := typed

checkExpressionType[expectedType_][DBTyped[expr_, type_]] :=
    DBRaise[
        checkExpressionType,
        "type_mismatch",
        {},
        <| "Expression" -> expr, "ExpectedType" -> expectedType, "PassedType" -> type |>
    ]
    
checkExpressionType[pars___][args___] :=
    DBRaise[checkExpressionType, "bad_args", {{pars}, {args}}]
