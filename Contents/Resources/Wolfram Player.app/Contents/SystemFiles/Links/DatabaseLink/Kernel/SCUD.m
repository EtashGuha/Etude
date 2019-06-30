(*===================================================================*)
(*================ Select, Create, Update, Delete ===================*)
(*===================================================================*)

Begin["`SQL`Private`"] 


SQLColumn::datatypename = "Illegal value for DataTypeName option: `1`";
SQLColumn::datalength = "Illegal value for DataLength option: `1`";
SQLColumn::nullable = "Illegal value for Nullable option: `1`";
SQLColumn::derbytype = "Type `1` is not supported by Apache Derby. Try using `2` instead.";
SQLTable::tablename = "\"`1`\" is not a valid table name. A valid table name is a string with no spaces which may be wrapped in SQLTable."
SQLInsert::unsupported = "TimeObject is not supported for Microsoft Access databases. Try using SQLDateTime instead."


Options[SQLCreateTable] = {
    "Timeout" :> $SQLTimeout,
    "Index" -> None
}

Options[SQLDropTable] = {
    "Timeout" :> $SQLTimeout
}

Options[SQLInsert] = Options[SQLExecute]

Options[ SQLSelect ] = { 
    "SortingColumns" -> None, 
    "ColumnSymbols" -> None, 
    "Distinct" -> False,
    "EscapeProcessing" -> True,
    "FetchDirection" -> "Forward",
    "FetchSize" -> Automatic, 
    "GetAsStrings" -> False, 
    "MaxFieldSize" -> Automatic,
    "MaxRows" -> Automatic, 
    "ShowColumnHeadings" -> False,
    "Timeout" :> $SQLTimeout,
    "BatchSize" -> 0,
    "JavaBatching" -> True,
    "ScanQueryFiles" -> False
}

Options[SQLDelete] = {
    "Timeout" :> $SQLTimeout
}

Options[SQLUpdate] = {
    "Timeout" :> $SQLTimeout
}
         
SQLCreateTable[conn_SQLConnection,
               table:(_SQLTable | _String), 
               col:(_SQLColumn | {__SQLColumn}),
               opts:OptionsPattern[]] := sqlCreateTable[conn,table,col,opts] 
    
sqlCreateTable[conn_SQLConnection,
               table:(_SQLTable | _String), 
               col:(_SQLColumn | {__SQLColumn}),
               opts:OptionsPattern[]] := 
  Module[ {tbl, useOpts, timeout, index, cols, res, sql}, 
    Catch[
      useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLCreateTable]];
      timeout = Lookup[useOpts, "Timeout"];
      index   = Flatten@List[Lookup[useOpts, "Index"]];

      (* Providing a string for the table name is okay, but it needs
         to be wrapped in SQLTable so the formatSQL function knows 
         what to do with it. *)   
      If[invalidTableNameQ[table],Message[SQLTable::tablename,If[Head[table]===SQLTable,First@table,ToString@table]],tbl = If[StringQ[table], SQLTable[table], table];
      
      cols = sqlCreateColumn /@ Flatten[{col}];
    
      SQLBeginTransaction[conn];
      sql = StringTemplate["CREATE TABLE `1` (`2`)"]["`1`", StringRiffle[ToString /@ cols, ", "]];
      res =Check[SQLExecute[conn, sql, {tbl}, "Timeout" -> timeout],
      	         checkDerbyColumnType[conn,col],
      	         JDBC::error
      	         ];
      	      If[index =!= {None},
                 sql = StringTemplate["CREATE INDEX `1``2`_idx ON `3` (`4`)"]
              	[Replace[tbl, SQLTable[t_] :> t], 
                  First@index,
                  "`1`",
                  StringRiffle[index, ", "]
                ];
         SQLExecute[conn, sql, {tbl}, "Timeout" -> timeout];
              ];
       
       SQLCommitTransaction[conn];
      res
      
      ]
    ]
  ]


 
checkDerbyColumnType[conn_SQLConnection,col:(_SQLColumn | {__SQLColumn})] := Module[{types,str,txtstr,strDatetime,strVarbinary},
	 If[
      	 DatabaseLink`SQL`Private`getRDBMS[conn] == "Apache Derby",
      	 types = Lookup[Options[#]&/@col,"DataTypeName"];
         str = StringContainsQ[types,#,IgnoreCase->True]&/@{"DATETIME","VARBINARY"};
         txtstr = TextString[#,ListFormat->{"{","||","}"}]&/@str;
         {strDatetime,strVarbinary} = First@ToExpression[#]&/@txtstr;
         Which[
         	   strDatetime&&strVarbinary == True,
               Message[SQLColumn::derbytype,"DATETIME","TIMESTAMP"];Message[SQLColumn::derbytype,"VARBINARY","VARCHAR FOR BIT DATA or BLOB"];$Failed,
               strDatetime == True,
               Message[SQLColumn::derbytype,"DATETIME","TIMESTAMP"];$Failed,
               strVarbinary == True,
               Message[SQLColumn::derbytype,"VARBINARY","VARCHAR FOR BIT DATA or BLOB"];$Failed 
      	       ],
      	       Message[JDBC::error] 
      	 ]
]
          
sqlCreateColumn[SQLColumn[(name_String | {_String, name_String}), opts:OptionsPattern[]]] := Module[
    {dt, dtn, dw, nl, pk, def, useOpts, stmt}, 
    useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLColumn]];
    dtn = Lookup[useOpts, "DataTypeName"]; 
    dw  = Lookup[useOpts, "DataLength"]; 
    nl  = Lookup[useOpts, "Nullable"]; 
    pk  = Lookup[useOpts, "PrimaryKey"]; 
    def = Lookup[useOpts, "Default"]; 

    dt = Which[ 
        StringQ[dtn], StringTemplate[" `1`"][dtn], 
        True, Message[SQLColumn::datatypename, dtn]; Throw[$Failed];
    ]; 
    dw = Which[ 
        dw === None , "", 
        IntegerQ[dw], StringTemplate["(`1`)"][dw], 
        True, Message[SQLColumn::datalength, dw]; Throw[$Failed]; 
    ]; 
    nl = Which[ 
        nl === None, "",
        TrueQ[nl], " NULL",
        nl === False, " NOT NULL", 
        True, Message[SQLColumn::nullable, nl]; Throw[$Failed]; 
    ]; 
    pk = Which[ 
        TrueQ[pk], " PRIMARY KEY", 
        True, ""
    ];
    def = Which[ 
        def === None, "",
        True, StringTemplate[" DEFAULT '`1`'"][def]
    ];
    stmt = StringJoin@{ name, dt, dw, def, nl, pk }
]

SQLDropTable[ conn_SQLConnection, table:(_SQLTable | _String), opts:OptionsPattern[]] := Module[
    {tbl, useOpts, timeout}, 

    useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLDropTable]];
    timeout = Lookup[useOpts, "Timeout"];

    tbl = If[StringQ[table], SQLTable[table], table];
    
    SQLExecute[conn, "DROP TABLE `1`", {tbl}, "Timeout" -> timeout]
]

SQLInsert[conn_SQLConnection,
          table:(_SQLTable | _String),
          names:{___String}, 
          values:{__},
          opts:OptionsPattern[]] :=
  SQLInsert[conn, table, SQLColumn /@ names, values, opts]

SQLInsert[conn_SQLConnection,
          table:(_SQLTable | _String),
          names:{{__String,_String}...}, 
          values:{__},
          opts:OptionsPattern[]] :=
  SQLInsert[conn, table, SQLColumn[Last[#]] & /@ names, values, opts]

SQLInsert[conn_SQLConnection,
          (SQLTable[table_String, ___?OptionQ] | table_String),
          names:{ SQLColumn[(_String|{_String, _String}), ___?OptionQ] ... }, 
          values:{__},
          opts:OptionsPattern[]] := Module[ 
    { useOpts, timeout, tbl, cols, vals, rgk, maxrows, gas, sch, mfs, fs, fd, ep, cs, bs, jb, sql },

    useOpts  = Join[canonicalOptions[Flatten[{opts}]], Options[SQLInsert]];
    rgk      = Lookup[useOpts, "GetGeneratedKeys"];
    timeout  = Lookup[useOpts, "Timeout"];
    gas      = Lookup[useOpts, "GetAsStrings"];
    sch      = Lookup[useOpts, "ShowColumnHeadings"];
    ep       = Lookup[useOpts, "EscapeProcessing"];
    cs       = Lookup[useOpts, "ColumnSymbols"];
    maxrows  = Lookup[useOpts, "MaxRows"];
    mfs      = Lookup[useOpts, "MaxFieldSize"];
    fs       = Lookup[useOpts, "FetchSize"];
    fd       = Lookup[useOpts, "FetchDirection"];
    bs       = Replace[Lookup[useOpts, "BatchSize"], {Infinity :> Length[values], Except[_Integer] :> Length[values]}]; 
    jb       = Lookup[useOpts, "JavaBatching"];

    tbl = If[StringQ[table], table, First[table]];

    cols = If[names === {},
        "",
        StringTemplate["(`1`)"][StringRiffle[getSQLColumnName /@ names, ","]]
    ];
   If[checkMSAccessColumnValues[conn,values],Message[SQLInsert::unsupported];$Failed, 
    vals = If[MatchQ[values, {__List}],
        StringTemplate["(`1`)"][StringRiffle[Table["?",Length@First@values], ","]],
        StringTemplate["(`1`)"][StringRiffle[Table["?", Length@values], ","]]
    ];
    
    sql = StringTemplate["INSERT INTO `table` `cols` VALUES `vals`"][<|
        "table" -> tbl,
        "cols" -> cols,
        "vals" -> vals
    |>];
    
    SQLExecute[conn, sql, values, 
        "Timeout" -> timeout, "GetGeneratedKeys" -> rgk, "GetAsStrings" -> gas, "ShowColumnHeadings" -> sch,
        "EscapeProcessing" -> ep, "ColumnSymbols" -> cs, "MaxRows" -> maxrows, "MaxFieldSize" -> mfs, 
        "FetchSize" -> fs, "FetchDirection" -> fd, "BatchSize" -> bs, "JavaBatching" -> jb
    ]
   ]
  ]

checkMSAccessColumnValues[conn_SQLConnection,values:{__}] := Module[{},
	If[DatabaseLink`SQL`Private`getRDBMS[conn] == "Ucanaccess for access db(Jet) using hasqldb"&&!FreeQ[values,_TimeObject],
    	True,False]
]

getSQLColumnName[SQLColumn[(col_String|{table_String,col_String}), opts:OptionsPattern[]]] := With[
    {nm = If[Head[Unevaluated[table]] =!= String, col, table <> "." <> col]},
    StringTrim@If[StringContainsQ[nm, WhitespaceCharacter], "\"" <> nm <> "\"", nm]
]

invalidTableNameQ[SQLTable[name_String, opts:OptionsPattern[]]] := invalidTableNameQ[name]

invalidTableNameQ[name_String] := Module[{},StringContainsQ[name," "]]

invalidTableNameQ[name_List] := Module[{}, AnyTrue[name,invalidTableNameQ]]

invalidTableNameQ[name_] := True

getTableName[SQLTable[name_, opts:OptionsPattern[]]] := name

getTableName[name_] := name

filterSQLTableOptions[SQLTable[table_String, opts:OptionsPattern[]]] := SQLTable[table]

filterSQLTableOptions[table_String] := table

SQLSelect[conn_SQLConnection | conn_String,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          opts:OptionsPattern[]
         ] := 
  SQLSelect[conn, table, SQLColumn["*"], None, opts];

SQLSelect[conn_SQLConnection | conn_String,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          columns:(_SQLColumn | {__SQLColumn}),
          opts:OptionsPattern[]] :=
  SQLSelect[conn, table, Flatten@{columns}, None, opts];

SQLSelect[conn_SQLConnection | conn_String,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          columns:(_String | {__String}),
          opts:OptionsPattern[]] :=
  SQLSelect[conn, table, SQLColumn /@ Flatten[{columns}], None, opts];

SQLSelect[conn_SQLConnection | conn_String,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          columns:{{_String, _String}..},
          opts:OptionsPattern[]] :=
  SQLSelect[conn, table, SQLColumn /@ columns, None, opts];

SQLSelect[conn_SQLConnection | conn_String,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          condition_,
          opts:OptionsPattern[]] := 
  SQLSelect[conn, table, SQLColumn["*"], condition, opts];

SQLSelect[conn_SQLConnection,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          columns:(_SQLColumn | {__SQLColumn}),
          condition_,
          opts:OptionsPattern[]] := Module[
    { tbls, cols, useOpts, distinct, orderby, order,
      maxrows, timeout, gas, sch, where, stmt,
      mfs, fd, fs, ep, rrs, cs, bs, jb, scanQueryFiles, invalidTables},
    
 	(*If there are multiple tables check if any of those have invalid table names, if yes then it issues a message for each such table*)
 	If[invalidTableNameQ[table],
 		invalidTables=Select[Flatten[{table}],TrueQ[invalidTableNameQ[#]]&];
 		invalidTables=getTableName/@invalidTables;
 		Message[SQLTable::tablename,If[Length[invalidTables] === 1, First[invalidTables],invalidTables]];
		$Failed,
		tbls=If[MatchQ[Flatten[{table}], {__SQLTable}], 
		SQLArgument @@ Flatten[{filterSQLTableOptions[table]}],
        SQLArgument @@ (SQLTable /@ Flatten[{table}])
        ];
        
    cols = SQLArgument @@ Flatten[{columns}];
            
    useOpts  = Join[canonicalOptions[Flatten[{opts}]], Options[SQLSelect]];
    distinct = Lookup[useOpts, "Distinct"]; 
    order    = Lookup[useOpts, "SortingColumns"];
    maxrows  = Lookup[useOpts, "MaxRows"];
    timeout  = Lookup[useOpts, "Timeout"];
    gas      = Lookup[useOpts, "GetAsStrings"];
    sch      = Lookup[useOpts, "ShowColumnHeadings"];
    mfs      = Lookup[useOpts, "MaxFieldSize"];
    fs       = Lookup[useOpts, "FetchSize"];
    fd       = Lookup[useOpts, "FetchDirection"];
    ep       = Lookup[useOpts, "EscapeProcessing"];
    cs       = Lookup[useOpts, "ColumnSymbols"];
    
    rrs      = Lookup[Join[useOpts, {"ResultSet" -> False}], "ResultSet"];   
    bs       = Replace[Lookup[useOpts, "BatchSize"], {Infinity :> 1, Except[_Integer] -> 0}];
    jb       = Lookup[useOpts, "JavaBatching"];
    scanQueryFiles = Lookup[useOpts, "ScanQueryFiles"];

    distinct = If[TrueQ[distinct], 
        "DISTINCT ", 
        ""
    ]; 
    orderby = If[order === None, 
        order = {};
        "",
        " ORDER BY `4`"
    ];
    where = If[condition === None,
        "",
        " WHERE `3`"
    ];

    stmt = StringTemplate["SELECT `distinct` `cols` FROM `tbls` `where` `orderby`"][<|
        "distinct" -> distinct,
        "cols" -> "`1`",
        "tbls" -> "`2`",
        "where" -> where,
        "orderby" -> orderby
    |>];
    
    SQLExecute[conn, stmt, {cols, tbls, condition, SQLArgument @@ Flatten[{order}]}, 
      "MaxRows" -> maxrows, "Timeout" -> timeout, "GetAsStrings" -> gas, 
      "ShowColumnHeadings" -> sch, "MaxFieldSize" -> mfs, "FetchDirection" -> fd, 
      "FetchSize" -> fs, "EscapeProcessing" -> ep, "ColumnSymbols" -> cs, "ResultSet" -> rrs,
      "BatchSize" -> bs, "JavaBatching" -> jb, "ScanQueryFiles" -> scanQueryFiles]
      ]
 	
 ]
 	
SQLSelect[conn_SQLConnection | conn_String,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          columns:(_String | {__String}),
          condition_,
          opts:OptionsPattern[]] :=
  SQLSelect[conn, table, SQLColumn /@ Flatten[{columns}], condition, opts];

SQLSelect[conn_SQLConnection | conn_String,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          columns:{{_String, _String}..},
          condition_,
          opts:OptionsPattern[]] :=
  SQLSelect[conn, table, SQLColumn /@ columns, condition, opts];

SQLDelete[ conn_SQLConnection,
           table:(_SQLTable | {__SQLTable} | _String | {__String}), 
           opts:OptionsPattern[]] := 
  SQLDelete[conn, table, None, opts] 

SQLDelete[ conn_SQLConnection,
           table:(_SQLTable | _String), 
           condition_, 
           opts:OptionsPattern[]] := Module[
    { useOpts, timeout, tbl, sql,invalidTables }, 

    useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLDelete]];
    timeout = Lookup[useOpts, "Timeout"];
    
  If[invalidTableNameQ[table],
 		invalidTables=Select[Flatten[{table}],TrueQ[invalidTableNameQ[#]]&];
 		invalidTables=getTableName/@invalidTables;
 		Message[SQLTable::tablename,If[Length[invalidTables] === 1, First[invalidTables],invalidTables]];
		$Failed,
		tbl = If[StringQ[table], SQLTable[table], SQLArgument @@ Flatten[{filterSQLTableOptions[table]}]];
		sql = StringTemplate["DELETE FROM `1` `2`"]["`1`", If[condition === None, "", " WHERE `2`"]];
		SQLExecute[conn, sql, {tbl, condition}, "Timeout" -> timeout]
	]
]

SQLUpdate[conn_SQLConnection,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          names:{__String}, 
          values:{__},
          opts:OptionsPattern[]] :=
  SQLUpdate[conn, table, SQLColumn /@ names, values, None, opts]

SQLUpdate[conn_SQLConnection,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          names:{{__String, _String}..}, 
          values:{__},
          opts:OptionsPattern[]] :=
  SQLUpdate[conn, table, SQLColumn[Last[#]] & /@ names, values, None, opts]

SQLUpdate[conn_SQLConnection,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          names:{__SQLColumn}, 
          values:{ __ },
          opts:OptionsPattern[]] := 
  SQLUpdate[conn, table, names, values, None, opts]


SQLUpdate[conn_SQLConnection,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          names:{__String}, 
          values:{__},
          condition_,
          opts:OptionsPattern[]] :=
  SQLUpdate[conn, table, SQLColumn /@ names, values, condition, opts]

SQLUpdate[conn_SQLConnection,
          table:(_SQLTable | {__SQLTable} | _String | {__String}),
          names:{{__String,_String}..}, 
          values:{__},
          condition_,
          opts:OptionsPattern[]] :=
  SQLUpdate[conn, table, SQLColumn[Last[#]] & /@ names, values, condition, opts]

SQLUpdate[ conn_SQLConnection,
           table:(_SQLTable | {__SQLTable} | _String | {__String}), 
           names:{__SQLColumn}, 
           values:{__},
           condition_,
           opts:OptionsPattern[]
         ] := Module[ 
    { useOpts, timeout, tbls, set, sql,invalidTables }, 
  
    useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLUpdate]];
    timeout = Lookup[useOpts, "Timeout"];
   If[invalidTableNameQ[table],
 		invalidTables=Select[Flatten[{table}],TrueQ[invalidTableNameQ[#]]&];
 		invalidTables=getTableName/@invalidTables;
 		Message[SQLTable::tablename,If[Length[invalidTables] === 1, First[invalidTables],invalidTables]];
		$Failed,
		tbls = If[MatchQ[Flatten[{table}], {__SQLTable}], 
        SQLArgument @@ Flatten[{filterSQLTableOptions[table]}],
        SQLArgument @@ (SQLTable /@ Flatten[{table}])
        ];
        
    sql = StringTemplate["UPDATE `1` SET `2` `3`"]["`1`", "`2`", If[condition === None, "", " WHERE `3`"]];
    set = MapThread[Rule[#1, #2] &, {names, values}];
    
    SQLExecute[conn, sql, {tbls, SQLArgument @@ set, condition}, "Timeout" -> timeout] 
   ]
   
]

End[] (* `SQL`Private` *)
