(* Wolfram Language package *)

Package["Databases`Schema`"]

PackageImport["Databases`"] (* DatabaseReference, RelationalDatabase *)

PackageImport["Databases`Common`"] (* Global Patterns *)

(* 
    Types are curated on the python side $Types is reading a dump that can be generated using 

    python run.py create_type_dump

    or 

    RunProcess[
        {"python", FileNameJoin[{PacletResource["Databases", "Python"], "run.py"}],
        "create_data_dump"}
    ]

*)

PackageExport["DBCreateValidDatabase"]

DBCreateValidDatabase[assoc_, ref_DatabaseReference] := With[{
    res = KeyDrop[assoc, "ID"]},
    RelationalDatabase[
        Append[res, "ID" -> IntegerString[Hash[{res, ref}], 36]],
        ref
    ]
]

$missing = Missing["NotAvailable"]


$FieldDefaults = <|
    "BackendType" -> None,
    "Default"     -> None,
    "Indexed"     -> False,
    "Nullable"    -> False,
    "NativeTypeString" -> Missing["NotAvailable"]
|>


$messageTemplates = <|
    "paclet_download_error"     :> "MessageTemplate" :> RelationalDatabase::resnodwnld,
    "internal_error"            :> "MessageTemplate" :> RelationalDatabase::interr,
    "internet_disallowed"       :> "MessageTemplate" :> RelationalDatabase::intdisal,
    "paclet_install_failure"    :> "MessageTemplate" :> RelationalDatabase::pcltinsterr,
    _ :> "MessageTemplate" :> "Unknown internal error"
|>

makeFailure[code_String, failureCode_, details_:<||>] := 
    makeFailure[{code}, failureCode, details]
    
makeFailure[{code_String, params___}, failureCode_, details_:<||>] :=
    Failure[
        "DatabaseFailure", <|
            Replace[code, $messageTemplates],
            "MessageParameters" -> {params},
            "FailureCode" -> failureCode,
            "FailureDetails" -> details
        |>
    ]

$rdbErrorHandler = Function[{sym, failure},
    Replace[failure, {
        Failure[code: "paclet_download_error" | "internet_disallowed" , ___] :> makeFailure[
            code,
            "missing_resources",
             <| "ErrorType" -> "paclet_download_error" |>
        ], 
        Failure[code: "paclet_install_failure" , ___] :> makeFailure[
            code,
            "missing_resources",
             <| "ErrorType" -> "paclet_install_error" |>
        ],
        any_ :> any
    }]
]
    
Options[RelationalDatabase] := {
    IncludeRelatedTables -> False
}    
    
(* automatic inspection.  *)

RelationalDatabase[d : $DBReferencePattern, opts:OptionsPattern[]] :=
    DBHandleError[$rdbErrorHandler] @ DBInspect[d, All, opts]

RelationalDatabase[tables: $DBTablePattern, d : $DBReferencePattern, opts:OptionsPattern[]] :=
    DBHandleError[$rdbErrorHandler] @ DBInspect[d, tables, opts]

RelationalDatabase[
    expr_,
    conn: _Rule | _RuleDelayed | {RepeatedNull[_Rule | _RuleDelayed]},
    rest___
] :=
    RelationalDatabase[expr, DatabaseReference[conn], rest]

RelationalDatabase[pre___, conn: _URL | _File, post___] :=
    RelationalDatabase[pre, DatabaseReference[conn], post]

RelationalDatabase[assoc_, ref_DatabaseReference] /;
    IntegerString[Hash[{KeyDrop[assoc, "ID"], ref}], 36] =!= assoc["ID"] :=
        Failure[
            "DatabaseFailure",
            <|
                "MessageTemplate"   :> RelationalDatabase::edited,
                "MessageParameters" -> {}
            |>
        ]

RelationalDatabase[args___] := (
    ArgumentCountQ[RelationalDatabase, Length[{args}], 1, 2];
    Null /; False
)
(* Global Properties *)

tablePart[obj_, table_, rest___] := obj[[1, "Tables", table, rest]]


(obj : _RelationalDatabase)["DatabaseReference"|"Connection"] := Replace[
    DBHandleError[] @ DBAuthenticationGet[obj],
    f_?FailureQ :> Missing["NoConnection"]
]

(obj : _RelationalDatabase)["Exists"] := True

(obj : _RelationalDatabase)["Properties"] := {"Tables", "Connection"}


(obj : _RelationalDatabase)["Tables"] :=
	Keys[Lookup[obj[[1]], "Tables", <||>]]

(* Table Properties *)

(obj : _RelationalDatabase)[table_, "Exists"] := MemberQ[obj["Tables"], table]
(obj : _RelationalDatabase)[table_, "Properties"] := 
    If[
        obj[table, "Exists"],
        {"Columns", "ForeignKeys", "Name", "PrimaryKey", "UniquenessConstraints", "Indexes"},
        {}
    ]
    
(obj : _RelationalDatabase)[table_, "Name"] := 
    If[
        obj[table, "Exists"], 
        table, 
        $missing
    ]

(obj : _RelationalDatabase)[table_, "Columns"] :=
    If[
        obj[table, "Exists"],
		Keys[tablePart[obj, table, "Columns"]],
        $missing
    ]

(obj : _RelationalDatabase)[table_, "PrimaryKey", prop:All|{___String}|_String:All] :=
    If[
        obj[table, "Exists"],
		Replace[tablePart[obj, table, "PrimaryKey"], _Missing :> <|"Columns" -> {}, "ConstraintName" -> None|>][[prop]],
        $missing
    ]

(obj : _RelationalDatabase)[from_, p:"UniquenessConstraints"|"ForeignKeys"|"Indexes", prop:All|{___String}|_String:All] :=
    If[
        obj[from, "Exists"],
        Replace[tablePart[obj, from, p, All, prop], _Missing :> {}],
        $missing
    ]

(* Column Properties *)

(obj : _RelationalDatabase)[table_, column_, "Properties"] := 
    If[
        obj[table, column, "Exists"], 
        {"NativeTypeString", "Default", "Indexed", "Name", "Nullable"},
        {}
    ]


(obj : _RelationalDatabase)[table_, column_, "Exists"] := 
    And[
        obj[table, "Exists"],
        MemberQ[obj[table, "Columns"], column]
    ]

(obj : _RelationalDatabase)[table_, column_, "Type"] := 
    If[
        obj[table, column, "Exists"],
        Replace[
            getTypeInfo[obj[table, column, "BackendType"], obj],
            a_Association :> a["Type"]
        ],
        $missing
    ]

(obj : _RelationalDatabase)[table_, column_, "TypeInfo"] := 
    If[
        obj[table, column, "Exists"],
        validateType[obj[table, column, "BackendType"], obj],
        $missing
    ]

(obj : _RelationalDatabase)[table_, column_, "Name"] := 
    If[
        obj[table, column, "Exists"],
        column,
        $missing
    ]

(obj : _RelationalDatabase)[
    table_,
    column_,
    prop: Alternatives @@ Keys[$FieldDefaults]
] :=
    If[
        obj[table, column, "Exists"],
        Lookup[
            tablePart[obj, table, "Columns", column],
            prop,
            $FieldDefaults[prop]
        ],
        $missing
    ]

(obj : _RelationalDatabase)[path: Repeated[_, {0, 2}], All] :=
    If[
        obj[path, "Exists"],
        obj[path, obj[path, "Properties"]],
        $missing
    ]

(obj : _RelationalDatabase)[path: Repeated[_, {0, 2}], props_List] :=
    If[
        obj[path, "Exists"],
        AssociationMap[
            obj[path, #] &,
            props
        ],
        $missing
    ]
