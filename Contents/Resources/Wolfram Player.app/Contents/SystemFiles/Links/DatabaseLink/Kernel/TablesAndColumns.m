(*===================================================================*)
(*======== Table, Column, and DataType Lookup Functionality =========*)
(*===================================================================*)

Begin["`SQL`Private`"] 


SQLTableInformation::tabletype = "Illegal value for TableType option: `1`"
SQLTableInformation::catalog = "Illegal value for Catalog option: `1`"
SQLTableInformation::schema = "Illegal value for Schema option: `1`"

SQLTableVersionColumns::catalog = "Illegal value for Catalog option: `1`"
SQLTableVersionColumns::schema = "Illegal value for Schema option: `1`"

SQLUserDefinedTypeInformation::types = "Illegal value for Types option: `1`"
SQLUserDefinedTypeInformation::catalog = "Illegal value for Catalog option: `1`"
SQLUserDefinedTypeInformation::schema = "Illegal value for Schema option: `1`"

userDefinedTypeCheck::udtype = "Illegal value for Types option: `1`"

SQLColumnInformation::catalog = "Illegal value for Catalog option: `1`"
SQLColumnInformation::schema = "Illegal value for Schema option: `1`"


Options[ SQLTable ] = { 
    "TableType" -> $DefaultTableType
}

Options[ SQLTables ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "TableType" -> $DefaultTableType 
}

Options[ SQLTableNames ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "TableType" -> $DefaultTableType 
}

Options[ SQLTableInformation ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "ShowColumnHeadings"->False,
    "TableType" -> $DefaultTableType
}

Options[ SQLTablePrivileges ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "ShowColumnHeadings"->False
}

Options[ SQLTableExportedKeys ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "ShowColumnHeadings" -> False
}

Options[ SQLTableImportedKeys ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "ShowColumnHeadings" -> False
}

Options[ SQLTableIndexInformation ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "ShowColumnHeadings" -> False
}

Options[ SQLTablePrimaryKeys ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "ShowColumnHeadings"->False
}
    
Options[ SQLTableVersionColumns] = { 
    "Catalog" -> None, 
    "Schema" -> None, 
    "ShowColumnHeadings" -> False
}
    
Options[ SQLUserDefinedTypeInformation ] = { 
    "Catalog" -> None,
    "Schema" -> None,
    "Types" -> None,
    "ShowColumnHeadings" -> False
}

Options[ SQLColumn ] = { 
    "DataTypeName" -> None, 
    "DataLength" -> None,
    "Default" -> None,
    "Nullable" -> False,
    "PrimaryKey" -> False
}

Options[ SQLColumns ] = { 
    "Catalog" -> None, 
    "Schema" -> None
}

Options[ SQLColumnNames ] = { 
    "Catalog" -> None, 
    "Schema" -> None
}

Options[ SQLColumnInformation] = { 
    "Catalog" -> None, 
    "Schema" -> None, 
    "ShowColumnHeadings" -> False
}

Options[ SQLColumnPrivileges] = { 
    "Catalog" -> None, 
    "Schema" -> None, 
    "ShowColumnHeadings" -> False
}

Options[ SQLDataTypeInformation ] = { 
    "ShowColumnHeadings" -> False
}

Options[ SQLSchemaInformation ] = { 
    "ShowColumnHeadings" -> False
}


If[!StringQ[$DefaultTableType],
    $DefaultTableType = "TABLE"
];


SQLTableInformation[
    SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
    table_String | table:Null,
    opts:OptionsPattern[]
] := JavaBlock[Module[
    {useOpts, tt, sch, meta, rs, schema, catalog}, 
    Block[{$JavaExceptionHandler = ThrowException}, 
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLTableInformation]];
            catalog = Lookup[useOpts, "Catalog"];
            schema = Lookup[useOpts, "Schema"];
            tt = Lookup[useOpts, "TableType"];
            sch = Lookup[useOpts, "ShowColumnHeadings"];

            Which[
                tt === None, tt = Null,
                StringQ[tt], tt = {tt}, 
                !MatchQ[tt, {___String}], Message[SQLTableInformation::tabletype, tt]; Return[$Failed]
            ];
            Which[
                catalog === None, catalog = Null, 
                !StringQ[catalog], Message[SQLTableInformation::catalog, catalog]; Return[$Failed]
            ];
            Which[
                schema === None, schema = Null, 
                !StringQ[schema], Message[SQLTableInformation::schema, schema]; Return[$Failed]
            ];
            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
      
            meta = connection@getMetaData[]; 
            rs = meta@getTables[catalog, schema, table, tt];

            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            SQLStatementProcessor`getAllResultData[ rs, False, TrueQ[sch]]
        ] (* Catch *)
    ] (* Block *)
]]

SQLTableInformation[conn_SQLConnection, opts:OptionsPattern[]] :=
    SQLTableInformation[conn, Null, opts]
  
SQLTableNames[conn_SQLConnection, table_String | table:Null, opts:OptionsPattern[]] := Module[
    {tables},
    tables = SQLTables[conn, table, opts];
    If[tables === $Failed, 
        $Failed,
        First /@ tables
    ]
]

SQLTableNames[conn_SQLConnection, opts:OptionsPattern[]] :=
    SQLTableNames[conn, Null, opts]

SQLTables[conn_SQLConnection, table_String | table:Null, opts:OptionsPattern[]] := Module[
    {data, nameIndex, typeIndex, useOpts, tt, catalog, schema}, 
    useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLTables]];
    tt = Lookup[useOpts, "TableType"];
    catalog = Lookup[useOpts, "Catalog"];
    schema = Lookup[useOpts, "Schema"];
    
    data = SQLTableInformation[conn, table, "Catalog" -> catalog, 
                                            "Schema" -> schema, 
                                            "TableType" -> tt, 
                                            "ShowColumnHeadings" -> True];
    If[data === $Failed, Return[$Failed]];
    
    {nameIndex, typeIndex} =
        Flatten[Position[ToUpperCase /@ data[[1]], #] & /@ {"TABLE_NAME", "TABLE_TYPE"}];
       
    data = SQLTable[#[[nameIndex]], "TableType" -> #[[typeIndex]]] & /@ Drop[data, 1]
]

SQLTables[conn_SQLConnection, opts:OptionsPattern[]] :=
    SQLTables[conn, Null, opts]

SQLTablePrivileges[
    conn:SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
    table_String | table:Null,
    opts:OptionsPattern[]
] := sqlTableInfoHelper[conn, SQLTablePrivileges, table, opts];

SQLTablePrivileges[conn_SQLConnection, opts:OptionsPattern[]] :=
    SQLTablePrivileges[conn, Null, opts]

SQLTableExportedKeys[
    conn:SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
    table_String | table:Null,
    opts:OptionsPattern[]
] := sqlTableInfoHelper[conn, SQLTableExportedKeys, table, opts];

SQLTableExportedKeys[conn_SQLConnection, opts:OptionsPattern[]] :=
    SQLTableExportedKeys[conn, Null, opts]

SQLTableImportedKeys[
    conn:SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
    table:(_String|_SQLTable),
    opts:OptionsPattern[]
] := sqlTableInfoHelper[conn, SQLTableImportedKeys, table, opts];

SQLTablePrimaryKeys[ 
    conn:SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ], 
    table:(_String|_SQLTable),
    opts:OptionsPattern[]
] := sqlTableInfoHelper[conn, SQLTablePrimaryKeys, table, opts];

SQLTableIndexInformation[
    conn:SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
    table:(_String|_SQLTable),
    opts:OptionsPattern[]
] := sqlTableInfoHelper[conn, SQLTableIndexInformation, table, opts];

sqlTableInfoHelper[conn_SQLConnection, func_Symbol, SQLTable[name_String, to:OptionsPattern[]], 
    opts:OptionsPattern[]] :=
    sqlTableInfoHelper[conn, func, name, opts];
sqlTableInfoHelper[  SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
                     func_Symbol,
                     table_String | table:Null, 
                     opts:OptionsPattern[]] := JavaBlock[Module[
    {useOpts, sch, meta, rs, schema, catalog}, 
    Block[{$JavaExceptionHandler = ThrowException}, 
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLTablePrivileges]];
            catalog = Lookup[useOpts, "Catalog"];
            schema = Lookup[useOpts, "Schema"];
            sch = Lookup[useOpts, "ShowColumnHeadings"];

            Which[
                catalog === None, catalog = Null, 
                !StringQ[catalog], Message[SQLTableInformation::catalog, catalog]; Return[$Failed]
            ];
            Which[
                schema === None, schema = Null, 
                !StringQ[schema], Message[SQLTableInformation::schema, schema]; Return[$Failed]
            ];
            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
      
            meta = connection@getMetaData[];
            Switch[func, 
                SQLTablePrivileges,
                rs = meta@getTablePrivileges[catalog, schema, table],
                
                SQLTableExportedKeys, 
                rs = meta@getExportedKeys[catalog, schema, table],
                 
                SQLTableImportedKeys,
                rs = meta@getImportedKeys[catalog, schema, table],
                
                SQLTablePrimaryKeys,
                rs = meta@getPrimaryKeys[catalog, schema, table],
                
                SQLTableIndexInformation,
                rs = meta@getIndexInfo[catalog, schema, table, False, True]
            ];

            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            SQLStatementProcessor`getAllResultData[ rs, False, TrueQ[sch]]
        ] (* Catch *)
    ] (* Block *)
]]

SQLTableVersionColumns[
    SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ],
    table_String | table:Null,
    opts:OptionsPattern[]
] := JavaBlock[Module[
    {useOpts, sch, meta, rs, schema, catalog},
    Block[
        {$JavaExceptionHandler = ThrowException},
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLColumnInformation]];
            catalog = Lookup[useOpts, "Catalog"];
            schema = Lookup[useOpts, "Schema"];
            sch = Lookup[useOpts, "ShowColumnHeadings"];
            Which[
              catalog === None, catalog = Null, 
              !StringQ[catalog], Message[SQLTableVersionColumns::catalog, catalog];
                                 Return[$Failed]
            ];
            Which[
              schema === None, schema = Null, 
              !StringQ[schema], Message[SQLTableVersionColumns::schema, schema];
                                Return[$Failed]
            ];
            If[!JavaObjectQ[connection],
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
            meta = connection@getMetaData[];
            rs = meta@getVersionColumns[catalog, schema, table];
            
            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            SQLStatementProcessor`getAllResultData[rs, False, TrueQ[sch]]
        ] (* Catch *)
    ] (* Block *)
]]

userDefinedTypeCheck[udtype_] := Module[ 
    {},
    Block[
        {$JavaExceptionHandler = ThrowException},
        Catch[
            Which[
                StringQ[udtype], Switch[udtype,
                    "DISTINCT", {2001}, 
                    "STRUCT", {2002}, 
                    "JAVA_OBJECT", {2000}, 
                    _, Message[userDefinedTypeCheck::udtype, udtype];
                       Return[$Failed]
                ],
                True, Message[userDefinedTypeCheck::udtype, udtype];
                      Return[$Failed]
            ]
        ]
    ]
]

SQLUserDefinedTypeInformation[
    SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
    typeName_String | typeName:Null,
    opts:OptionsPattern[]
] := JavaBlock[Module[
    {useOpts, types, sch, meta, rs, schema, catalog},
    Block[
        {$JavaExceptionHandler = ThrowException},
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLUserDefinedTypeInformation]];
            catalog = Lookup[useOpts, "Catalog"];
            schema = Lookup[useOpts, "Schema"];
            types = Lookup[useOpts, "Types"];
            sch = Lookup[useOpts, "ShowColumnHeadings"];
            Which[
              types === None, typesValues = Null,
              StringQ[types], Check[typesValues = userDefinedTypeCheck[types], Return[$Failed]],
              ListQ[types], Check[typesValues = Flatten[userDefinedTypeCheck/@types], Return[$Failed]],
              True, Message[SQLUserDefinedTypeInformation::types, types];
                    Return[$Failed]
            ];
            Which[
              catalog === None, catalog = Null, 
              !StringQ[catalog], Message[SQLUserDefinedTypeInformation::catalog, catalog];
                                 Return[$Failed]
            ];
            Which[
              schema === None, schema = Null, 
              !StringQ[schema], Message[SQLUserDefinedTypeInformation::schema, schema];
                                Return[$Failed]
            ];
            If[ !JavaObjectQ[connection],
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
            meta = connection@getMetaData[];
            rs = meta@getUDTs[catalog, schema, typeName, MakeJavaObject[typesValues]];
            
            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            SQLStatementProcessor`getAllResultData[rs, False, TrueQ[sch]]
        ] (* Catch *)
    ] (* Block *)
]]

SQLUserDefinedTypeInformation[ conn_SQLConnection, opts:OptionsPattern[]] :=
  SQLUserDefinedTypeInformation[ conn, Null, opts]
  
SQLColumnInformation[
    SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ],
    {table_String | table:Null, column_String | column:Null},
    opts:OptionsPattern[]
] := JavaBlock[Module[
    {useOpts, sch, meta, rs, schema, catalog}, 
    Block[
        {$JavaExceptionHandler = ThrowException},
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLColumnInformation]];
            catalog = Lookup[useOpts, "Catalog"];
            schema = Lookup[useOpts, "Schema"];
            sch = Lookup[useOpts, "ShowColumnHeadings"];

            Which[
                catalog === None, catalog = Null, 
                !StringQ[catalog], Message[SQLColumnInformation::catalog, catalog]; Return[$Failed]
            ];
            Which[
                schema === None, schema = Null, 
                !StringQ[schema], Message[SQLColumnInformation::schema, schema]; Return[$Failed]
            ];
            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
      
            meta = connection@getMetaData[]; 
            rs = meta@getColumns[catalog,schema,table,column];

            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            SQLStatementProcessor`getAllResultData[rs, False, TrueQ[sch]]
        ] (* Catch *)
    ] (* Block *)
]]

SQLColumnInformation[conn_SQLConnection, opts:OptionsPattern[]] := 
  SQLColumnInformation[conn, {Null, Null}, opts]
  
SQLColumnInformation[conn_SQLConnection, table_String, opts:OptionsPattern[]] := 
  SQLColumnInformation[conn, {table, Null}, opts]

SQLColumnInformation[conn_SQLConnection, SQLTable[table_String, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnInformation[conn, {table, Null}, opts]

SQLColumnInformation[conn_SQLConnection, SQLColumn[col_String, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnInformation[conn, {Null, col}, opts]

SQLColumnInformation[conn_SQLConnection, SQLColumn[{table_String, col_String}, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnInformation[conn, {table, col}, opts]

SQLColumns[conn_SQLConnection, {table_String | table:Null, column_String | column:Null}, opts:OptionsPattern[]] := Module[
    {data, tableIndex, columnIndex, typeIndex, nullableIndex, lengthIndex, useOpts, catalog, schema,
        defIndex}, 

    useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLTableInformation]];
    catalog = Lookup[useOpts, "Catalog"];
    schema = Lookup[useOpts, "Schema"];

    data = SQLColumnInformation[ conn, {table, column}, "Catalog" -> catalog,
                                                        "Schema" -> schema, 
                                                        "ShowColumnHeadings" -> True];
    If[data === $Failed, Return[$Failed]];
    
    {tableIndex, columnIndex, typeIndex, nullableIndex, lengthIndex, defIndex} =
        Flatten[Position[ToUpperCase /@ data[[1]], #] & /@ 
            {"TABLE_NAME", "COLUMN_NAME", "TYPE_NAME", "NULLABLE", "COLUMN_SIZE", "COLUMN_DEF"}];
       
    data = SQLColumn[{#[[tableIndex]], #[[columnIndex]]}, 
             "DataTypeName" -> #[[typeIndex]],
             "DataLength" -> #[[lengthIndex]],
             "Default" -> #[[defIndex]],
             "Nullable" -> #[[nullableIndex]]
             
    ] & /@ Drop[data, 1]
]

SQLColumns[conn_SQLConnection, opts:OptionsPattern[]] := 
  SQLColumns[conn, {Null, Null}, opts]
  
SQLColumns[conn_SQLConnection, table_String, opts:OptionsPattern[]] := 
  SQLColumns[conn, {table, Null}, opts]

SQLColumns[conn_SQLConnection, SQLTable[table_String, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumns[conn, {table, Null}, opts]

SQLColumns[conn_SQLConnection, SQLColumn[col_String, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumns[conn, {Null, col}, opts]

SQLColumns[conn_SQLConnection, SQLColumn[{table_String, col_String}, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumns[conn, {table, col}, opts]

SQLColumnNames[conn_SQLConnection, {table_String | table:Null, column_String | column:Null}, opts:OptionsPattern[]] := Module[
    {columns},
    columns = SQLColumns[conn, {table, column}, opts];
    If[columns === $Failed, 
        $Failed,
        First /@ columns
    ]
]
  
SQLColumnNames[conn_SQLConnection, opts:OptionsPattern[]] := 
  SQLColumnNames[conn, {Null, Null}, opts]
  
SQLColumnNames[conn_SQLConnection, table_String, opts:OptionsPattern[]] := 
  SQLColumnNames[conn, {table, Null}, opts]

SQLColumnNames[conn_SQLConnection, SQLTable[table_String, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnNames[conn, {table, Null}, opts]

SQLColumnNames[conn_SQLConnection, SQLColumn[col_String, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnNames[conn, {Null, col}, opts]

SQLColumnNames[conn_SQLConnection, SQLColumn[{table_String, col_String}, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnNames[conn, {table, col}, opts]

SQLColumnPrivileges[
    SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ],
    {table_String | table:Null, column_String | column:Null},
    opts:OptionsPattern[]
] := JavaBlock[Module[
    {useOpts, sch, meta, rs, schema, catalog}, 
    Block[{$JavaExceptionHandler = ThrowException},
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLColumnPrivileges]];
            catalog = Lookup[useOpts, "Catalog"];
            schema = Lookup[useOpts, "Schema"];
            sch = Lookup[useOpts, "ShowColumnHeadings"];

            Which[
                catalog === None, catalog = Null, 
                !StringQ[catalog], Message[SQLColumnInformation::catalog, catalog]; Return[$Failed]
            ];
            Which[
                schema === None, schema = Null, 
                !StringQ[schema], Message[SQLColumnInformation::schema, schema]; Return[$Failed]
            ];
            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
      
            meta = connection@getMetaData[]; 
            rs = meta@getColumnPrivileges[catalog, schema, table, column];
        
            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            SQLStatementProcessor`getAllResultData[rs, False, TrueQ[sch]]
        ] (* Catch *)
    ] (* Block *)
]]

SQLColumnPrivileges[conn_SQLConnection, opts:OptionsPattern[]] := 
  SQLColumnPrivileges[conn, {Null, Null}, opts]
  
SQLColumnPrivileges[conn_SQLConnection, table_String, opts:OptionsPattern[]] := 
  SQLColumnPrivileges[conn, {table, Null}, opts]

SQLColumnPrivileges[conn_SQLConnection, SQLTable[table_String, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnPrivileges[conn, {table, Null}, opts]

SQLColumnPrivileges[conn_SQLConnection, SQLColumn[col_String, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnPrivileges[conn, {Null, col}, opts]

SQLColumnPrivileges[conn_SQLConnection, SQLColumn[{table_String, col_String}, ___?OptionQ], opts:OptionsPattern[]] := 
  SQLColumnPrivileges[conn, {table, col}, opts]


SQLDataTypeInformation[SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ],
                       opts:OptionsPattern[] ] := JavaBlock[Module[ 
    {useOpts, sch, meta, rs}, 
    Block[{$JavaExceptionHandler = ThrowException}, 
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLTableInformation]];
            sch = Lookup[useOpts, "ShowColumnHeadings"];

            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
      
            meta = connection@getMetaData[]; 
            rs = meta@getTypeInfo[];

            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            SQLStatementProcessor`getAllResultData[rs, False, TrueQ[sch]]      
        ]
    ]
]]

SQLDataTypeNames[conn_SQLConnection] := Module[
    { data, nameIndex }, 
    data = SQLDataTypeInformation[ conn, ShowColumnHeadings->True];
    If[data === $Failed, Return[$Failed]];
    
    {nameIndex} = Flatten[Position[ToUpperCase /@ data[[1]], "TYPE_NAME"]];   
    data = Flatten[#[[nameIndex]] & /@ Drop[data, 1]]
]

SQLTableTypeNames[SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ]] := JavaBlock[Module[
    {meta, rs, data}, 
    Block[{$JavaExceptionHandler = ThrowException}, 
        Catch[
            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
      
            meta = connection@getMetaData[]; 
            rs = meta@getTableTypes[];

            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            data = SQLStatementProcessor`getAllResultData[ rs, False, False];
            If[MatrixQ[data], data = Flatten[data]];
            data
        ]
    ]
]]
  
SQLSchemaInformation[SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
                     opts:OptionsPattern[]] := JavaBlock[Module[
    {meta, rs, data, useOpts, sch},
    Block[{$JavaExceptionHandler = ThrowException}, 
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLTableInformation]];
            sch = Lookup[useOpts, "ShowColumnHeadings"];

            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
      
            meta = connection@getMetaData[]; 
            rs = meta@getSchemas[];

            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            data = SQLStatementProcessor`getAllResultData[rs, False, TrueQ[sch]]
        ]
    ]
]]

SQLSchemaNames[ conn_SQLConnection] := Module[
    { data, nameIndex }, 
    data = SQLSchemaInformation[conn, ShowColumnHeadings->True];
    If[data === $Failed, Return[$Failed]];
    
    {nameIndex} = Flatten[Position[ToUpperCase /@ data[[1]], "TABLE_SCHEM"]];   
    data = Flatten[#[[nameIndex]] & /@ Drop[data, 1]]
]  
  
SQLCatalogNames[SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ]] := JavaBlock[Module[
    {meta, rs, data}, 
    Block[{$JavaExceptionHandler = ThrowException},
        Catch[
            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
        
            meta = connection@getMetaData[]; 
            rs = meta@getCatalogs[];

            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
            data = SQLStatementProcessor`getAllResultData[ rs, False, False];
            If[MatrixQ[data], data = Flatten[data]];
            data
        ]
    ]
]]


End[] (* `SQL`Private` *)
