Package["Databases`Common`"]

PackageImport["Databases`"]


PackageExport["DBObjectMethod"]
PackageExport["DBDefMethod"]
PackageExport["DBDefMethodChaining"]

(*
**  Longer form with object type symbol. Need to flatten it out so that we could
**  use UpValues
*)
DBObjectMethod[sym_Symbol, methodName_String][args___] :=  DBObjectMethod[sym, methodName, args]

(*
**  Operator form for method call. Allows one to call method as
**
**     obj // DBObjectMethod[name][ags]
*)
DBObjectMethod[methodName_String] :=
	Function[Function[obj, DBObjectMethod[Head @ obj, methodName, obj, ##]]]

(*
**  Catch-all definition
*)
DBObjectMethod[type_Symbol, method_String, ___] :=
    DBRaise[
        DBObjectMethod,
        "method_does_not_exist_for_type",
        {},
        <| "ObjectType" -> type, "MethodName" -> method |>
    ]

(*
**  Defines a custom operator DBDefMethod[typeSymbol], which can be used to generate
**  actual method definitions from much simpler / more concise syntax. Basically,
**  this is used to reduce / auto-generate boilerplate code.
*)
DBDefMethod /: SetDelayed[DBDefMethod[sym_Symbol] @ methodName_String[o_, args___], rhs_] :=
	CompoundExpression[
		sym /: DBObjectMethod[sym, methodName, o, args] := rhs
		,
		sym /: DBObjectMethod[sym, methodName] :=
			Function[DBObjectMethod[sym, methodName, ##]]
        ,
        sym /: DBObjectMethod[sym, methodName, params___] :=
            DBRaise[
                DBObjectMethod,
                "invalid_method_arguments",
                {},
                <|
                    "ObjectType" -> sym,
                    "MethodName" -> methodName,
                    "MethodArguments" -> {params}
                |>
            ]
	]


(*
**  Short form for DBDefMethod. NOTE: Requires the first argument (object pattern) to be
**  of the form var_objectHead
*)
DBDefMethod /: SetDelayed[
	DBDefMethod @ methodName_String[patt: Verbatim[Pattern][_, Verbatim[Blank][sym_Symbol]], args___],
	rhs_
] := DBDefMethod[sym] @ methodName[patt, args] := rhs


(*
** Allows using the prefix form for method calls, like
**
**      obj @ "method1"[x, y] @ "method2"[z]
**
*)
DBDefMethodChaining[sym_Symbol] :=
	CompoundExpression[
		o_sym[method_String[args___][inner_]] := o[method[args]][inner],
		o_sym[method_String[args___]] := DBObjectMethod[sym, method][o, args]
	]
