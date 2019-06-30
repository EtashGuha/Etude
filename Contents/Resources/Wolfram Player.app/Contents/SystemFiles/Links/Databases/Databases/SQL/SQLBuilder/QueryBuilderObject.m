Package["Databases`SQL`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"] (* DBRaise, DBHandleError *)
PackageImport["Databases`Schema`"] (* Types *)


PackageExport["$DBUseDecriptiveTableNames"]

PackageExport["DBAlias"] (* WL aliasing *)
PackageExport["DBGetCurrentAlias"]
PackageExport["DBEnsureAlias"]
PackageExport["DBIsSingleRow"]
PackageExport["DBSetSingleRow"]
PackageExport["DBGetName"]

PackageScope["new"]
PackageScope["generateAlias"]
PackageScope["isRenameable"]
PackageScope["setRenameable"]
PackageScope["getType"]
PackageScope["getTables"]
PackageScope["getInnerTable"]
PackageScope["getFieldsInfo"]
PackageScope["isAggregateQuery"]
PackageScope["setAggregateQuery"]
PackageScope["getPK"]
PackageScope["getFieldPrefixTrie"]
PackageScope["appendFieldPrefixTrie"]
PackageScope["createFieldName"]
PackageScope["getSchema"]



(* ============================================================================ *)
(* ==========        DBQueryBuilderObject constructor / getters / setters      ========= *)
(* ============================================================================ *)


$DBUseDecriptiveTableNames = True

Module[{ctr = 0},
    generateAlias[base_:"T"] := StringJoin[base, ToString[++ctr]]
]


DBQueryBuilderObject @ new["Base"] :=
    DBQueryBuilderObject @ <|
        "TableType" -> None,
        "Renameable" -> True,
        (*
        ** A name of the native database table. This field is non-trivial only for native tables
        *)
        "TableName" -> None,
		"PrimaryKey" -> None,
        "Alias" -> generateAlias[],
        "Distinct" -> False,
        "WrapOnNextOperation" -> False,
        "AggregateQuery" -> False, (* This field is non-trivial only for queries *)
        "FieldsInfo" -> <||>, (* This field is non-trivial only for native tables *)
        "SelectedFields" -> {}, (* This field is non-trivial only for queries *)
        "ProperFields" -> <||>, (* This field is non-trivial only for queries *)
        "Tables" -> {}, (* This field is non-trivial only for queries *)
        "Joins" -> {}, (* This field is non-trivial only for queries *)
        "Where" -> {}, (* This field is non-trivial only for queries *)
        "GroupBy" -> None, (* This field is non-trivial only for queries *)
        "OrderBy" -> None, (* This field is non-trivial only for queries *)
		"FieldPrefixTrie" -> <||>,
		"Limit" -> None,
		"Offset" -> None,
        "Schema" -> None,
        "SingleRow" -> False
    |>


createFieldName[name_String -> type_] :=
    createFieldName[name, type]

createFieldName[name_String, type_?DBTypeQ] :=
    DBRawFieldName[name, CreateUUID[], type]

createFieldName[name_String, type_] :=
    With[{normalizedType = DBType[type]},
        createFieldName[name, normalizedType] /; DBTypeQ[normalizedType]
    ]

DBDefError @ createFieldName

(*
**  Constructing DBQueryBuilderObject
*)
newTableQuery[
    modelName_String,
    tableName_String,
    aliasAssoc_Association?AssociationQ,
    schema : _RelationalDatabase
] :=
	Module[
        {
            actualColumns, generatedFields,fieldsForPrefixTrie,
            nativeQuery, fieldsToAdd, actualColumnsInfo
        }
        ,
		actualColumns =  Values @ aliasAssoc;

        actualColumnsInfo = AssociationMap[
            schema[tableName, #, "TypeInfo"] &,
            actualColumns
        ];

        (* Construct inner native table *)
		nativeQuery = With[{base = DBQueryBuilderObject @ new["Base"]}, base @ set[
			<|
				"TableType" -> "NativeTable",
				"TableName" -> tableName, (*Resolve to actual db name here*)
				"FieldsInfo"-> actualColumnsInfo,
                "Schema" -> schema,
				"PrimaryKey" -> schema[tableName, "PrimaryKey", "Columns"],
                "SelectedFields" -> Keys @ actualColumnsInfo,
                If[TrueQ[$DBUseDecriptiveTableNames],
                    "Alias" -> StringJoin[tableName, "_", base @ get["Alias"]],
                    <||>
                ]
			|>
		]];

        (* Generate universally unique "exported" field names *)
		generatedFields = AssociationThread[
            Keys @ aliasAssoc,
            Map[createFieldName[# -> actualColumnsInfo[#]]&, actualColumns]
        ];

        (* Resolve back actual db table column names to aliases *)
		fieldsForPrefixTrie = AssociationThread[
            Keys @ aliasAssoc, Values @ generatedFields
        ];

        (* Construct actual field expressions for unique generated fields *)
		With[{nativeTableName = nativeQuery @ DBGetName[]},
			fieldsToAdd = Association @ Replace[
				Normal @ generatedFields,
				(field_ -> genField_) :> (genField -> <|
					"Expression" -> DBSQLField[nativeTableName, field],
					"Alias" -> genField
				|>),
				{1}
			]
		];

        (* Construct and return the final DBQueryBuilderObject *)
		DBQueryBuilderObject @ new["Query"] @ append[
			"Tables", nativeQuery
		] @ set[
			"ProperFields" ->  fieldsToAdd
		] @ set[
			"SelectedFields" ->  Keys @ fieldsToAdd
		] @ set[
			"FieldPrefixTrie" -> DBQueryPrefixTrie @ new[
                $rawFieldNamePattern, fieldsForPrefixTrie
            ] @ "extend"[modelName]
		] @ set[
			<|
            	"Schema" -> Inherited,
				"PrimaryKey" -> schema[tableName, "PrimaryKey", "Columns"]
			|>
        ] @ wrapInQuery[
            (*
            ** This moves query fields from annotations to inner fields. This is
            ** necessary, so that we can refer to them in queries. This wasn't needed
            ** before, because we could resolve to native table fields. Now we treat
            ** the new table with renamed fields as a black box / starting point, so
            ** we need this extra level of SELECT.
            *)
            All
        ]
	]


(*
**  DBQueryBuilderObject constructor
*)
DBQueryBuilderObject @ new["NativeTable", args__] := newTableQuery[args]

DBQueryBuilderObject @ new["Query", name_: Automatic] :=
    DBQueryBuilderObject @ new["Base"] @ set[
        <|
            "TableType" -> "Query",
            If[name =!= Automatic, "Alias" -> name, Sequence @@ {}]
        |>
    ] @ set[
        "FieldPrefixTrie" -> DBQueryPrefixTrie @ new[
            $rawFieldNamePattern,
            <||>,
            If[name === Automatic, None, name]
        ]
    ]


(*
**  Short - cut constructor for tables
*)
DBQueryBuilderObject @ DBCreateTable[
    modelName_String,
    tableName_String,
    aliasAssoc_Association?AssociationQ,
    schema : _RelationalDatabase
] := 
    DBQueryBuilderObject @ new[
        "NativeTable", modelName, tableName, aliasAssoc, schema
    ]

DBQueryBuilderObject @ DBCreateTable[args___] :=
    DBRaise[DBCreateTable, "invalid_table_constructor_arguments", {args}]



(* ============        Getters and setters, and basic accessors      ========== *)

(*
**  Type
*)
q_DBQueryBuilderObject @ getType[] := q @ get["TableType"]

q_DBQueryBuilderObject @ DBIsSingleRow[] := q @ get["SingleRow"]

q_DBQueryBuilderObject @ DBSetSingleRow[flag: True | False] := q @ set["SingleRow" -> flag]


(*
**  (SQL) name / alias, and renaming
*)
q_DBQueryBuilderObject @ setRenameable[canBeRenamed_: True] :=
    q @ set["Renameable" -> canBeRenamed]

q_DBQueryBuilderObject @ isRenameable[] := TrueQ[q @ get["Renameable"]]

q_DBQueryBuilderObject @ DBGetName[] :=
    Which[
        q @ get @ "Alias" =!= None,
            q @ get @ "Alias",
        q @ getType[] === "NativeTable",
            q @ get @ "TableName",
        True,
            DBRaise[DBGetName, "can_not_determine_query_name_or_alias"]
    ]


q_DBQueryBuilderObject @ getSchema[]:=
    With[{s = q @ get["Schema"]},
        If[s === Inherited,
            First[q @ getTables[]] @ getSchema[], 
            (* else *)
            s
        ]
    ]
    

(*
**  Inner  tables
*)
q_DBQueryBuilderObject @ getTables[] := q @ get @ "Tables"

q_DBQueryBuilderObject @ getInnerTable[name_] :=
    SelectFirst[q @ getTables[], # @ DBGetName[] === name&]


(*
**  Native table field information
*)
q_DBQueryBuilderObject @ getFieldsInfo[] := q @ get @ "FieldsInfo"


(*
**  Field prefix trie and query aliasing
*)
query_DBQueryBuilderObject @ getFieldPrefixTrie[] := query @ get @ "FieldPrefixTrie"

query_DBQueryBuilderObject @ appendFieldPrefixTrie[trie_?DBQueryPrefixTrieQ] :=
    query @ transform[
        "FieldPrefixTrie",
        Function[t, t @ "addParent"[trie] ]
    ]

q_DBQueryBuilderObject @ DBAlias[al_, sticky_:False] :=
    q @ transform[
        "FieldPrefixTrie",
        Function[
            trie, trie @ "setAliasOrExtend"[al, sticky]
        ]
    ]

q_DBQueryBuilderObject @ DBGetCurrentAlias[] :=
    With[{trie = q @ getFieldPrefixTrie[]},
        If[Length[trie @ get["ProperFields"]] === 0,
            trie @ "getAlias"[],
            (* else *)
            None
        ]
    ]

right_DBQueryBuilderObject @ DBEnsureAlias[left_?DBQueryBuilderObjectQ, hardError_:False] :=
    With[
        {   (*  
            **  At this point left and right don't "know" yet, that they will be 
            **  joined, so we temporarily set their current (latest) aliases 
            **  sticky to ensure that fields will get prefixed with them 
            *)
            leftSticky = left @ transform[
                "FieldPrefixTrie", Function[trie, trie @ "setSticky"[]]],
            rightSticky = right @ transform[
                "FieldPrefixTrie", Function[trie, trie @ "setSticky"[]]]    
        },
        If[
    		Intersection[
    			leftSticky @ DBGetPrefixedFields[],
    			rightSticky @ DBGetPrefixedFields[]
    		] =!= {},
    		If[
    			TrueQ[hardError],
                DBRaise[DBEnsureAlias, "ambiguous_aliases_in_queries_for_join", {left, right}],
                right @ DBAlias[createDummyQueryAlias[]]
            ],
            right
        ]    
    ]
    

$aliasCtr = 0

createDummyQueryAlias[] := StringJoin["QueryAlias_", ToString[++$aliasCtr]]


(*
**  Aggregate query flag
*)
q_DBQueryBuilderObject @ isAggregateQuery[] := TrueQ @ q @ get @ "AggregateQuery"

q_DBQueryBuilderObject @ setAggregateQuery[flag: True | False : True] :=
    q @ set["AggregateQuery" -> flag]
