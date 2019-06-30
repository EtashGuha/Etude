(* Wolfram Language package *)

Package["Databases`Database`"]

PackageImport["Databases`"]
PackageImport["Databases`Python`"]
PackageImport["Databases`Common`"] (* DBRaise *)
PackageImport["Databases`Schema`"] (* $DBReferencePattern, $DBTablePattern, $DBSchemaPattern *)
PackageImport["Databases`SQL`"] (* DBQueryObjectQ *)


PackageExport["DBResultSet"]
PackageExport["DBRunQuery"]
PackageExport["$DBQueryLogger"]

$DBQueryLogger = Function[{expr}, Null, HoldAllComplete]


Options[DBResultSet] = Options[DBRunQuery] = {
    Authentication -> Inherited,
    "MissingElementFunction" -> Automatic,
    "InvalidColumnFunction" -> Automatic,
    "Lazy" -> False
}

pythonCommand[command_, query_, schema_, auth_:Inherited] :=
    With[
        {using = DBSchemaGet[schema]},
        DBRunPython[
            command[DBQueryToSQLAlchemy[query, using]],
            Authentication -> DBAuthenticationGet[auth, using],
            "Schema"       -> using
        ]
    ] 

DBResultSet[query_?DBUncompiledQueryQ, schema:$DBSchemaPattern:Inherited, opts:OptionsPattern[]] :=
    DBResultSet[DBQueryObject[DBQueryToQueryBuilderObject[query, DBSchemaGet[schema]]], opts]

DBResultSet[query_?DBQueryObjectQ, opts:OptionsPattern[]] :=
    DBResultSet[<|
        "Query" -> query,
        "Schema" -> query["Schema"],
        "QueryData" -> None,
		"FieldPostProcess" -> query["FieldPostProcess"],
        "RelationFunction" -> query["RelationFunction"],
        Options[DBResultSet], 
        opts
    |>]["Init"]

DBResultSet[sql:_String|Except[_?DBUncompiledQueryQ, _?DBQueryQ], schema:$DBSchemaPattern:Inherited, opts:OptionsPattern[]] :=
    DBResultSet[<|
        "Query" -> sql,
        "Schema" -> schema,
        "QueryData" -> None,
        Options[DBResultSet], 
        opts
    |>]["Init"]

DBResultSet[options_?AssociationQ]["Option", key_, default_:None] := 
    Lookup[options, key, default]

set_DBResultSet[prop_, Automatic] :=
    (* this is done in order to write an easier pattern in DBRunQuery *)
    set[prop]

set_DBResultSet["Properties"] := {
    "Rows", "RowAssociation", "RowCount",
    "Columns", "ColumnAssociation", "ColumnNames", "PrefixedColumnNames",
    "Dataset", "RelationFunction",
    "MissingElementFunction", "InvalidColumnFunction", "FieldPostProcess",
    "SQLString"
}

set_DBResultSet[All, rest___] :=
    set[set["Properties"], rest]

set_DBResultSet[s_List, rest___] :=
    AssociationMap[set[#, rest] &, s]

set_DBResultSet[Authentication] :=
    set["Option", Authentication, Inherited]

set_DBResultSet["Lazy"] := 
    set["Option", "Lazy", False]

set_DBResultSet[prop:"Query"|"Schema"|"QueryData"|"FieldPostProcess"|"RelationFunction"] :=
    set["Option", prop, None]

set_DBResultSet["MissingElementFunction"] :=
    Replace[
        set["Option", "MissingElementFunction"], 
        Automatic|_Missing|None :> Function[
            Missing["ColumnAbsent", #]
        ]
    ]

set_DBResultSet["InvalidColumnFunction"] :=
    Replace[
        set["Option", "InvalidColumnFunction"],
        Automatic|_Missing|None :> Function[
            ConstantArray[
                #["MissingElementFunction"][#2],
                #["RowCount"]
            ]
        ]
    ]



set_DBResultSet["Init"] := 
    If[
        And[
            Not @ set["Lazy"],
            Not @ MatchQ[set["QueryData"], _Association]
        ], 
        set["Evaluate"],
        set
    ]

myTranspose[<|(_ -> {})..|>, _] := {}
myTranspose[{}, keys_] := AssociationMap[{}&, keys]
myTranspose[a_, _] := Transpose[a, AllowedHeads -> All]

applyPostProcess[func_][qca_] := myTranspose[
	func /@ myTranspose[qca, None],
    Keys[qca]
]

set_DBResultSet["Evaluate"] := If[
    Not @ MatchQ[set["QueryData"], _Association],
    ReplacePart[
        set, 
        {1, "QueryData"} -> 
            Replace[
                pythonCommand[
                    "DatabaseQuery",
                    set["Query"],
                    set["Schema"],
                    set[Authentication]
                ],
                a_Association :> (
                    $DBQueryLogger[set["SQLString"]];
                    Append[
                        KeyDrop[a, {"QueryKeys", "QueryRows"}],
                        "QueryColumnAssociation" ->  AssociationThread[
                            a["QueryKeys"], 
                            If[
                                Length[a["QueryRows"]] === 0,
                                ConstantArray[{}, Length[a["QueryKeys"]]],
                                Replace[
                                    Transpose[a["QueryRows"]],
                                    None :> Missing["NotAvailable"],
                                    {2}
                                ]
                            ]
                        ]
                    ]
                )
            ]  
    ],
    set
]

set_DBResultSet["QueryColumnAssociation"] :=
    set["Evaluate"]["QueryData"]["QueryColumnAssociation"]

set_DBResultSet["SQLString"] := 
    pythonCommand[
        "DatabaseQueryToString",
        set["Query"],
        set["Schema"],
        set[Authentication]
    ]

set_DBResultSet["PrefixedColumnNames"] :=
    Replace[
        set["Query"], {
            q_?DBQueryObjectQ :> q["PrefixedColumnNames"],
            _ :> Map[DBPrefixedField, set["ColumnNames"]]
        }
    ]

set_DBResultSet["ColumnNames"] :=
    Replace[
        set["Query"], {
            q_?DBQueryObjectQ :> q["ColumnNames"],
            _ :> Keys @ set["QueryColumnAssociation"]
        }
    ] 

set_DBResultSet["KeyConversionFunction"] :=
    Replace[
        set["Query"], {
            q_?DBQueryObjectQ :> q["KeyConversionFunction"],
            _ :> Identity
        }
    ]    


set_DBResultSet["RowCount"] :=
    Length @ First[set["QueryColumnAssociation"], {}]

set_DBResultSet["ColumnAssociation"] := set["ColumnAssociation", All]

set_DBResultSet["ColumnAssociation", All] :=
    Replace[
        set["Query"], {
            _String :> 
                set["QueryColumnAssociation"],
            query_ :> 
                With[{
                    reversekcf = Composition[
                        query["RawToPrefixedFieldsMap"],
                        query["ReverseFieldMap"]
                    ],
                    invalid = set["InvalidColumnFunction"],
                    data = set["QueryColumnAssociation"],
                    fpp = set["FieldPostProcess"]
                    },
                    If[
                        fpp === Identity,
                        Identity,
                        applyPostProcess[
                            If[
                                set["RelationFunction"] =!= Automatic,
                                ReplaceAll[
                                    DBRelationFunction -> set["RelationFunction"]
                                ],
                                Identity
                            ][fpp]
                        ]
                    ] @ KeyMap[reversekcf, data]
                ]

        }
    ]




set_DBResultSet["ColumnAssociation", cols_List] := Part[set["ColumnAssociation"], cols]

set_DBResultSet["ColumnAssociation", All, rest___] :=
    set["ColumnAssociation", All, rest]

set_DBResultSet["ColumnAssociation", cols_, rest__] :=
	Part[set["ColumnAssociation", cols], All, rest]

set_DBResultSet["ColumnAssociation", col_, rest___] :=
    set["ColumnAssociation", {col}, rest]


set_DBResultSet["Columns", keys_List, rest___] :=
    (* this is done because we want to handle properly duplicated keys *)
    Lookup[set["ColumnAssociation", keys, rest], keys]

set_DBResultSet["Columns", rest___] :=
    Values @ set["ColumnAssociation", rest]

set_DBResultSet["Rows", rest___] :=
    Replace[
        set["Columns", rest],
        c:Except[{}] :> Transpose @ c
    ]

set_DBResultSet["RowAssociation", rest___] :=
	Replace[
		set["ColumnAssociation", rest],
        {<||> :> {}, a_ :> myTranspose[a, None]}
	]

set_DBResultSet["Dataset", rest___] :=
    Dataset @ set["RowAssociation", rest]

set_DBResultSet["Preview", rest___] :=
    Dataset @ Replace[
        KeyMap[
            StringJoin @ Riffle[Cases[#, _String, Infinity], "."] &,
            set["ColumnAssociation", rest]
        ],
        {<||> :> {}, a_ :> myTranspose[a, None]}
    ]

(* Automatic means data extraction is lazy, None means that is done immediately *)
dataMethod["Rows"|"RowAssociation"|"RowCount"|"Columns"|"ColumnAssociation"|"Dataset"|"Preview"] :=
    False
dataMethod[s_List] := AnyTrue[s, dataMethod]
dataMethod[_] := True
    
DBRunQuery[
    query_?DBQueryObjectQ,
    method : _String|{__String} | All : "Preview",
    keys : Except[_?OptionQ] : Automatic,
    opts : OptionsPattern[]
    ] :=
    DBResultSet[query, "Lazy" -> dataMethod[method], opts][method, keys]

DBRunQuery[
    query:_?DBQueryQ|_String,
    schema : $DBSchemaPattern:Inherited,
    method : _String | {__String} | All : "Preview",
    keys : Except[_?OptionQ] : Automatic,
    opts : OptionsPattern[]
    ] :=
    DBResultSet[query, schema, opts, "Lazy" -> dataMethod[method]][method, keys]
