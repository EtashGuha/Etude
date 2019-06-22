(*===================================================================*)
(*================= SQLConnectionPool Functionality =================*)
(*===================================================================*)

Begin["`SQL`Private`"] 


$SQLUseConnectionPool = False;


SQLConnectionPool::til = "Illegal value for TransactionIsolationLevel option: `1`"
SQLConnectionPool::readonly = "Illegal value for ReadOnly option: `1`"
SQLConnectionPool::catalog = "Illegal value for Catalog option: `1`"
SQLConnectionPool::maxactive = "Illegal value for MaximumActiveConnections option: `1`"
SQLConnectionPool::maxidle = "Illegal value for MaximumIdleConnections option: `1`"
SQLConnectionPool::minidle = "Illegal value for MinimumIdleConnections option: `1`"


Options[ SetSQLConnectionPoolOptions ] = JoinOptions[
    Options[ SetSQLConnectionOptions ], 
    {
      "MaximumActiveConnections" -> Automatic,
      "MaximumIdleConnections" -> Automatic,
      "MinimumIdleConnections" -> Automatic
    }
  ]
  
Options[ SQLConnectionPool ] =  Options[SetSQLConnectionPoolOptions]


$connectionPools = {};

If[!ListQ[$connectionPools], 
    $connectionPools = {};
];

(*
 * Maintain mapping of which connections are associated with which pools, so that
 * we can close connections on pool closure.
 *)
If[!MatchQ[$poolToConnections, _Association], 
    $poolToConnections = Association[];
];

SQLConnectionPools[] := $connectionPools;

SQLConnectionPools[SQLConnection[
                         jdbc_JDBC,
                         connection_,
                         id_Integer,
                         options:OptionsPattern[]]] := 
  Module[{pool, desc, location, name, pw, relativePath, un, v, ucp, properties, cat, transactionIsolationLevel, ro},
  
    {cat, desc, location, name, pw, properties, ro, relativePath, transactionIsolationLevel, ucp, un, v} = Lookup[
        Join[canonicalOptions[Flatten[{options}]], Options[OpenSQLConnection]],
        {"Catalog", "Description", "Location", "Name", "Password", 
            "Properties", "ReadOnly", "RelativePath", "TransactionIsolationLevel", 
            "UseConnectionPool", "Username", "Version"}
    ];
    
    pool = FirstCase[$connectionPools, SQLConnectionPool[
        _?JavaObjectQ, 
        jdbc, 
        _Integer, 
        "Catalog" -> cat, "Description" -> desc, "Location" -> location, "Name" -> name, "Password" -> pw,
        "Properties" -> properties, "ReadOnly" -> ro, "RelativePath" -> relativePath,
        "TransactionIsolationLevel" -> transactionIsolationLevel, "UseConnectionPool" -> ucp, 
        "Username" -> un, "Version" -> v
    ]]
]

SQLConnectionPools[SQLConnection[jdbc_JDBC, opts:OptionsPattern[]]] := 
    SQLConnectionPools[SQLConnection[jdbc, Null, -1, opts]]

SQLConnectionPools[name_String] := With[{dataSource = DataSources[name]},
    If[dataSource =!= Null,
      SQLConnectionPools[dataSource]
    ]
]


SQLConnectionPool /:
    SetOptions[ SQLConnectionPool[
                        javaObject_,
                        jdbc_JDBC,
                        id_Integer,
                        opts:OptionsPattern[]], opts2___] := 
        SetSQLConnectionPoolOptions[ SQLConnectionPool[javaObject, jdbc, id, opts], opts2]



SetSQLConnectionPoolOptions[SQLConnectionPool[
                                     javaObject_,
                                     jdbc_JDBC,
                                     id_Integer,
                                     opts:OptionsPattern[]], 
                                   opts2:OptionsPattern[]] := Module[
    {cat, desc, location, name, pw, properties, ro, relativePath, til, 
        un, ucp, v, pool, maxActive, maxIdle, minIdle, optTest},
    
    Block[{$JavaExceptionHandler = ThrowException},
        Catch[
            {desc, location, name, pw, properties, relativePath, ucp, un, v} = Lookup[
                Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]],
                {"Description", "Location", "Name", "Password", "Properties", 
                   "RelativePath", "UseConnectionPool", "Username", "Version"}
            ];
    
            optTest = FilterRules[{opts2}, Except[Options[SetSQLConnectionPoolOptions]]];
            If[optTest =!= {}, optionsErrorMessage[optTest, SQLConnectionPool, SQLConnectionPool]; Return[$Failed]];

            {cat, ro, til} = Lookup[
                Join[canonicalOptions[Flatten[{opts2}]], canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]],
                {"Catalog", "ReadOnly", "TransactionIsolationLevel"}
            ];
    
            {maxActive, maxIdle, minIdle} = Lookup[
                Join[canonicalOptions[Flatten[{opts2}]], Options[SetSQLConnectionPoolOptions]],
                {"MaximumActiveConnections", "MaximumIdleConnections", "MinimumIdleConnections"} 
            ];
        
            If[!JavaObjectQ[javaObject], 
              Message[SQLConnection::conn];
              Return[$Failed]
            ]; 
         
            (* Catalog *)
            Switch[cat, 
              _?StringQ, 
                javaObject@setDefaultCatalog[cat],
              Automatic, 
                Null,
              _, 
                Message[SQLConnectionPool::catalog, cat];
                cat = Lookup[Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]], "Catalog"];
            ];
    
            (* Transaction Isolation Level *)
            Switch[til, 
              "ReadUncommitted",
                javaObject@setDefaultTransactionIsolation[1],
              "ReadCommitted",
                javaObject@setDefaultTransactionIsolation[2],
              "RepeatableRead",
                javaObject@setDefaultTransactionIsolation[4],
              "Serializable",
                javaObject@setDefaultTransactionIsolation[8],
              Automatic, 
                Null,
              _, 
                Message[SQLConnectionPool::til, til];
                til = Lookup[Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]], "TransactionIsolationLevel"];
            ];
    
            (* Read Only *)
            Switch[ro, 
              (True | False), 
                javaObject@setDefaultReadOnly[ro],
              Automatic, 
                Null,
              _, 
                Message[SQLConnectionPool::readonly, ro];
                ro = Lookup[Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]], "ReadOnly"];
            ];
    
            Switch[maxActive, 
              _Integer, 
                javaObject@setMaxActive[maxActive],
              Automatic, 
                Null,
              _, 
                Message[SQLConnectionPool::maxactive, maxActive]
            ];

            Switch[maxIdle, 
              _Integer, 
                javaObject@setMaxIdle[maxIdle],
              Automatic, 
                Null,
              _, 
                Message[SQLConnectionPool::maxidle, maxIdle]
            ];

            Switch[minIdle, 
              _Integer, 
                javaObject@setMinIdle[minIdle],
              Automatic, 
                Null,
              _, 
                Message[SQLConnectionPool::minidle, minIdle]
            ];
    
            pool = SQLConnectionPool[javaObject, jdbc, id, "Catalog" -> cat, "Description" -> desc,
                "Location" -> location, "Name" -> name, "Password" -> pw,
                "Properties" -> properties, "ReadOnly" -> ro, "RelativePath" -> relativePath,
                "TransactionIsolationLevel" -> til, "UseConnectionPool" -> ucp, "Username" -> un, "Version" -> v
            ];
            $connectionPools = Replace[$connectionPools, SQLConnectionPool[_, _, id, ___] -> pool, 1];
            pool
        ]
    ]
];

SQLConnectionPoolClose[ SQLConnectionPool[
                      javaObject_?JavaObjectQ,
                      jdbc_JDBC,
                      id_Integer,
                      options:OptionsPattern[]]] := Block[
    {$JavaExceptionHandler = ThrowException},
    Catch[
        (* This will close all connections checked out from the pool. The documented Java behavior
         * is to leave connections "checked out to clients" unaffected by the pool closure; however,
         * if you try to close one of these connections after the pool is gone you get a
         * "pool not open" exception. So force closure of all pool connections on pool close.
         *)
        If[KeyExistsQ[$poolToConnections, id],
            (* Lots of Association stuff currently broken ... *)
            CloseSQLConnection[#] & /@ $poolToConnections[id];
            (*$poolToConnections = KeyDrop[$poolToConnections, id];*)
            $poolToConnections = Association[DeleteCases[Normal@$poolToConnections, id -> _]];
        ];
        javaObject@close[];
        ReleaseJavaObject[javaObject];
        $inTransaction = False;
        $connectionPools = DeleteCases[
            $connectionPools, 
            SQLConnectionPool[_, _, id, ___?OptionQ]
        ];
    ]
];

SQLConnectionPoolClose[SQLConnection[jdbc_JDBC, opts:OptionsPattern[]]] := 
    SQLConnectionPoolClose[SQLConnection[jdbc, Null, -1, opts]]

SQLConnectionPoolClose[conn:SQLConnection[
                       jdbc_JDBC,
                       connection_,
                       id_Integer,
                       options:OptionsPattern[]]] := Module[
    {pool},
    pool = SQLConnectionPools[conn];
    If[!MatchQ[pool, _Missing|Null], 
        SQLConnectionPoolClose[pool]
    ]
]

SQLConnectionPoolClose[name_String] := Module[
    {dataSource = DataSources[name]},
    If[dataSource =!= Null,
        SQLConnectionPoolClose[dataSource]
    ]
]


End[] (* `SQL`Private` *)
