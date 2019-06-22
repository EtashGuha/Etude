(* Wolfram Language package *)

Package["Databases`Schema`"]


PackageImport["DatabaseLink`"]
PackageImport["Databases`"] (* RelationalDatabase, DatabaseReference *)
PackageImport["Databases`Common`"] (* DBRaise, Global Patterns *)
PackageImport["Databases`Python`"] (* DBRunPython *)

PackageExport["DBAuthenticationGet"]
PackageExport["DBSchemaGet"]
PackageExport["DBReferences"]
PackageExport["DBConnectionValidQ"]

SetAttributes[{specificationLookup, DBAuthenticationGet}, HoldRest]

connFailure[args___] :=
    Failure[
        "DatabaseFailure", <|
            "MessageTemplate"   :> DatabaseReference::nvldconn,
            "MessageParameters" -> {args}
        |>
    ]

specificationLookup[name_, settings_] := Replace[
    {name, settings}, {
        {Automatic | Inherited, KeyValuePattern["Default" -> conf_]} :> conf,
        {Automatic | Inherited, conf: _List | _Association} :> First[conf],
        {Automatic | Inherited, Automatic | Inherited | None} :> <|"Backend" -> "SQLite"|>,
        {Automatic | Inherited, conf_} :> conf,
        {part_Integer, conf: _List | _Association} :> conf[[part]],
        {key_Key, conf_?AssociationQ} :> key[conf],
        {key_, conf_} :> (Message[settings::nvldconn, key, conf]; $Failed)
    }
]

(* Retrieving python opened connections *)

DBReferences[] :=
    DBRunPython["DatabaseReferences"[]]


(* DBSchemaGet will try to take Default or first defined schema in the database *)

DBSchemaGet[Inherited | Automatic, rest__] := DBSchemaGet[rest]
DBSchemaGet[expr_, rest__] := DBSchemaGet[expr]
DBSchemaGet[name: Automatic | Inherited | _Integer | _Key: Automatic] :=
    DBSchemaGet[
        Replace[
            specificationLookup[name, Databases`$Databases],
            path_String :> File[path]
        ]
    ]

DBSchemaGet[RelationalDatabase[f_File, rest___]] := Replace[
    Get[f],
    schema: Except[_?FailureQ] :> DBSchemaGet[RelationalDatabase[schema, rest]]
]
DBSchemaGet[db: RelationalDatabase[$DBSchemaPattern, ___]] := db
DBSchemaGet[schema: $DBSchemaPattern] := DBSchemaGet[RelationalDatabase[schema]]

DBSchemaGet[schema_] := 
    DBRaise @ connFailure[schema]

DBAuthenticationGet[Inherited | Automatic, rest__] := DBAuthenticationGet[rest]
DBAuthenticationGet[expr_, rest__] := DBAuthenticationGet[expr]

DBAuthenticationGet[name: Automatic | Inherited | _Key | _Integer: Automatic] :=
    DBAuthenticationGet[
        Replace[
            specificationLookup[name, Databases`$DatabaseAuthentications],
            s_String :> URL[s]
        ]
    ]

DBAuthenticationGet[config_] :=
    Replace[
        DatabaseReference[config],
        Except[_?DBConnectionValidQ] :> 
            DBRaise @ connFailure[config]
    ]

(* TODO: perhaps, make a stricter check in the future *)
DBConnectionValidQ[DatabaseReference[_Association?AssociationQ]] := True
DBConnectionValidQ[_] := False


DatabaseReference[spec: Inherited | Automatic] :=
    DBHandleError[] @ DBAuthenticationGet[spec]
    
DatabaseReference[None] :=
    DatabaseReference[<|"Backend" -> "SQLite"|>]

DatabaseReference[
    rules: {RepeatedNull[_Rule | _RuleDelayed]} | _RuleDelayed | _Rule
] :=
    DatabaseReference[<|rules|>]

DatabaseReference[sql : _RelationalDatabase] :=
    DatabaseReference[First[Cases[sql, _DatabaseReference], Inherited]]

DatabaseReference[File[s_String, rest___]] :=
    DatabaseReference[
        <|"Backend" -> "SQLite", "Name" -> s|>
    ] 


DatabaseReference[s_String] :=
    DatabaseReference @ If[
        StringMatchQ[s, WordCharacter.. ~~ "://" ~~ ___],
        URL[s],
        File[s]
    ]
DatabaseReference[URL[url_String, ___]] :=
    First @ StringReplace[
        url, {
            StartOfString ~~ "sqlite"|"sqllite" ~~ "://" ~~ path___ :> DatabaseReference @ <|
                "Backend" -> "sqlite",
                "Name" -> Replace[path, "" -> None]
            |>,
            StartOfString ~~ WordCharacter.. ~~ "://" ~~ ___ :>
                Replace[
                    Quiet @ URLParse[
                        url,
                        {"Scheme", "PathString", "Port", "Domain", "Username", "Password"}
                    ], {
                    {engine_, path_, port_, host_, username_, password_} :> DatabaseReference @ <|
                          "Backend" -> engine,
                            "Name" -> Replace[path, Except[""] :> StringTake[path, {2, -1}]],
                            "Port" -> port,
                            "Host" -> host,
                        "Username" -> username,
                        "Password" -> password
                    |>, 
                    _ :> invalidConn[DatabaseReference::nvldconn, url]
            }],
            ___ :> invalidConn[DatabaseReference::nvldconn, url]
        },
        IgnoreCase -> True
    ]

DatabaseReference[db_DatabaseReference] :=
    db

DatabaseReference[args___] := (
    ArgumentCountQ[DatabaseReference, Length[{args}], 1, 1];
    Null /; False
)

(* Starting High level DatabaseReference construct *)

$DatabaseReferenceProps = "Backend" | "Host" | "ID" | "Name" | "Password" | "Port" | "Username"

_DatabaseReference ? DBConnectionValidQ["Properties"] :=
    List @@ $DatabaseReferenceProps


DatabaseReference[spec_Association?AssociationQ]["ID"] :=
    Lookup[
        spec,
        "ID",
        IntegerString[
            Hash[
                Lookup[
                    spec,
                    {"Backend", "Name", "Port", "Host", "Username", "Password"},
                    None
                ]
            ],
            36
        ]
    ]


(c: _DatabaseReference ? DBConnectionValidQ)[
    props: {RepeatedNull[$DatabaseReferenceProps]}
] :=
    AssociationMap[c, props]

SetAttributes[invalidConn, HoldFirst]
invalidConn[msg_, args___] := 
    Failure[
        "DatabaseFailure",
        <|
            "MessageTemplate"   :> msg,
            "MessageParameters" :> {args}
        |>
    ]



(dbc: DatabaseReference[spec_Association?AssociationQ])["Name"] := 
    Replace[
        {dbc["Backend"], Lookup[spec, "Name", None]}, {
            {"SQLite", ""} :> invalidConn[None, "Name"],
            {"SQLite", path:_String|_File} :> ExpandFileName[path],
            {"SQLite", None|_Missing|Null} :> Missing["InMemory"],
            {"SQLite", f_} :> invalidConn[DatabaseReference::nvldvalue, f, "Name"],
            {_, "" | _Missing} :> invalidConn[DatabaseReference::emptyname],
            {_, f: Except[_String]} :> invalidConn[DatabaseReference::nvldvalue, f, "Name", "string"],
            {_, s_String} :> s
        }
    ]

DatabaseReference[spec_Association?AssociationQ][prop: $DatabaseReferenceProps] :=
    Replace[
        {prop, Lookup[spec, prop, None]}, {
            {"Backend", s_String} :>
                Replace[
                    ToLowerCase[s], {
                        "oracle"|"oraclesql" -> "Oracle",
                        "mssql"|"msql"|"microsoftsql"|"microsoft"|"ms" -> "MicrosoftSQL",
                        "sqlite"|"sqllite" -> "SQLite",
                        "mysql" -> "MySQL",
                        "postgresql"|"postgres"|"postgressql" -> "PostgreSQL",
                        value_ :> invalidConn[
                            DatabaseReference::nvldback, 
                            value,
                            "MicrosoftSQL, MySQL, Oracle, PostgreSQL, SQLite"
                        ]
                    }
                ],
            {"Host", IPAddress[s_String, ___]} :> s,
            {"Host"|"Username"|"Password"|"Port", None|_Missing|Automatic} :> None,
            {"Host"|"Username"|"Password", s_String} :> s,
            {"Port", value_Integer} /; IntegerQ[value] && TrueQ[NonNegative[value]] :> value,
            {"Port", value_} :>
                invalidConn[
                    DatabaseReference::nvldvalue,
                    value,
                    "Port",
                    "non-negative integer or Automatic"
                ],
            {_, value_} :> 
                invalidConn[
                    DatabaseReference::nvldvalue,
                    value,
                    prop,
                    "a string or None"
                ]
        }
    ]

(c: _DatabaseReference ? DBConnectionValidQ)[prop:"Connected"|"Disconnect"|"Connect"] := 
    DBHandleError[] @ DBRunPython[
        ("Database" <> prop)[],
        Authentication -> DBAuthenticationGet[c]
    ]

(c: _DatabaseReference ? DBConnectionValidQ)[All] :=
    c[List @@ $DatabaseReferenceProps]

(c: _DatabaseReference ? DBConnectionValidQ)[props_List] := 
    AssociationMap[c, props]

_DatabaseReference ? DBConnectionValidQ[prop_] := 
    Failure[
        "DatabaseFailure", <|
            "MessageTemplate"   :> DatabaseReference::nvldprop,
            "MessageParameters" -> {prop, StringJoin[Riffle[List @@ $DatabaseReferenceProps, ", "]]}
        |>
    ]
     

DatabaseConnect[conn:$DBReferencePattern|$DBSchemaPattern|_DatabaseReference?DBConnectionValidQ] :=
    DBHandleError[] @ DBAuthenticationGet[conn]["Connect"]

DatabaseDisconnect[conn:$DBReferencePattern|$DBSchemaPattern|_DatabaseReference?DBConnectionValidQ] :=
    DBHandleError[] @ DBAuthenticationGet[conn]["Disconnect"]