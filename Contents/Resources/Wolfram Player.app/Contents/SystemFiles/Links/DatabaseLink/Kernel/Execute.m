(*===================================================================*)
(*========================== SQLExecute =============================*)
(*===================================================================*)

Begin["`SQL`Private`"] 

SQLExecute::columnsymbols = "Illegal value for ColumnSymbols option: `1`"

SQLExecute::maxrows = "Illegal value for MaxRows option: `1`"

SQLExecute::timeout = "Illegal value for Timeout option: `1`"

SQLExecute::fetchsize = "Illegal value for FetchSize option: `1`"

SQLExecute::fetchdirection = "Illegal value for FetchDirection option: `1`"

SQLExecute::maxfieldsize = "Illegal value for MaxFieldSize option: `1`"

SQLExecute::multirowodbc = "Possible inline multirow insert; not supported by ODBC Access and Excel drivers. \
Try a parameterized query if problems persist."

SQLValue::illegalvalue = "The value `1` cannot be converted to a value in an SQL statement."

DatabaseLink`Utilities`$ClearParameters = Automatic;

Options[ SQLExecute ] = {
    "ColumnSymbols" -> None, 
    "EscapeProcessing" -> True,
    "FetchDirection" -> "Forward",
    "FetchSize" -> Automatic, 
    "GetAsStrings" -> False, 
    "GetGeneratedKeys" -> False,
    "MaxFieldSize" -> Automatic,
    "MaxRows" -> Automatic, 
    "ShowColumnHeadings" -> False,
    "Timeout" :> $SQLTimeout,
    "BatchSize" -> 1000,
    "JavaBatching" -> True,
    "ScanQueryFiles" -> False
}

SQLExecute[conn_SQLConnection, sql_String, opts:OptionsPattern[]] := Module[{query, pos},
	If[ TrueQ[OptionValue["ScanQueryFiles"]],
        If[ MemberQ[SQLQueryNames[], sql],
            pos = First[Flatten[Position[SQLQueryNames[], sql]]];
            query = SQLQueries[][[pos]];
            SQLExecute[conn, query],
            SQLExecute[ conn, sql, None, opts]
        ],
        SQLExecute[ conn, sql, None, opts]
    ]
];

SQLExecute[
    SQLConnection[JDBC[driver_, ___], connection_, _Integer, ___?OptionQ],
    ps_String,
    argsArg:{__}|None,
    opts:OptionsPattern[]
] := JavaBlock[Module[
    {sql = ps, params, useOpts, maxrows, 
        timeout, gas, sch, rrs, rrsBool, rsc = 1007, rst = 1003,
        mfs, fs, fd, ep, rgk, result, cols, cs, results, bs, jb,
        args = Replace[argsArg, None -> {}], rdbms = getRDBMS[connection], clearParams = DatabaseLink`Utilities`$ClearParameters},

    Block[{$JavaExceptionHandler = ThrowException},
    	
        Catch[

            If[!JavaObjectQ[connection], 
                Message[SQLConnection::conn];
                Return[$Failed]
            ];

            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLExecute]];
            maxrows = Lookup[useOpts, "MaxRows"];
            timeout = Lookup[useOpts, "Timeout"];
            gas     = Lookup[useOpts, "GetAsStrings"];
            sch     = Lookup[useOpts, "ShowColumnHeadings"];
            mfs     = Lookup[useOpts, "MaxFieldSize"];
            fs      = Lookup[useOpts, "FetchSize"];
            fd      = Lookup[useOpts, "FetchDirection"];
            ep      = Boole@TrueQ[Lookup[useOpts, "EscapeProcessing"]];
            rgk     = Lookup[useOpts, "GetGeneratedKeys"];
            cs      = Lookup[useOpts, "ColumnSymbols"];
            rrs     = Lookup[useOpts, "ResultSet", False];     (* set by SQLResultSet* *)
            bs      = Replace[Lookup[useOpts, "BatchSize"], {Infinity :> Length[args], Except[_Integer] :> Length[args]}];
            jb      = Lookup[useOpts, "JavaBatching"];
            
            clearParams = getClearStatementParameter[rdbms,DatabaseLink`Utilities`$ClearParameters];
            
            If[cs =!= Automatic && cs =!= None && !MatchQ[cs, _Function] && !MatchQ[cs, {___Symbol}],
                Message[SQLExecute::columnsymbols, cs];
                cs = None
            ];

            If[maxrows =!= Automatic && maxrows =!= All, 
                If[maxrows === None, maxrows = 0];
                If[!IntegerQ[maxrows] || maxrows < 0, 
                    Message[SQLExecute::maxrows, maxrows]
                ],
                maxrows = 0;
            ];

            If[timeout =!= None && timeout =!= Automatic,
                If[!IntegerQ[timeout] || timeout < 0,
                    Message[SQLExecute::timeout, timeout]
                ],
                (* This will have the effect of not calling setQueryTimeout at all in the statement processor,
                 * and presumably use the driver default *)
                timeout = 0;
            ];

            If[mfs =!= Automatic, 
                If[mfs === None, mfs = 0];
                If[!IntegerQ[mfs] || mfs < 0, 
                    Message[SQLExecute::maxfieldsize, mfs]
                ],
                mfs = 0;
            ];

            Switch[fd,
                "Forward", fd = 1000,
                "Reverse", fd = 1001, 
                "Unknown", fd = 1002,
                _, Message[SQLExecute::fetchdirection, fd]; fd = 1000;
            ];

            If[fs =!= Automatic && fs =!= All, 
                If[fs === None, fs = 0];
                If[!IntegerQ[fs] || fs < 0, 
                    Message[SQLExecute::fetchsize, fs]
                ],
                fs = 0;
            ];

            If[MatchQ[rrs, {_String}], 
                rst = Switch[First[rrs],
                    "ForwardOnly", 1003,
                    "ScrollInsensitive", 1004,
                    "ScrollSensitive", 1005,
                    "MySQLStreaming", 1003,
                    _, Message[SQLResultSetOpen::mode, First[rrs]]; Return[$Failed]
                ];
                rrsBool = True,
                rrsBool = False
            ];
          
            (* Format the SQL into something the driver will understand *)
            
            {sql, params} = formatSQL[ps, args];
            Spew[sql];
          
            (* Convert SQLExpr objects to strings ahead of time. *) 
            params = params /. {SQLExpr[x_] :> SQLExpr["SQLExpr[" <> ToString[InputForm[x]] <> "]"]};
          
            If[TrueQ[$printPreparedStatement], 
                Print[{sql, params}];
            ];

            (* Thin databases like SQLite don't support all options.
             * Clobber the unimplemented ones before handing off to Java.
             *)
            If[StringMatchQ[rdbms, "SQLite*"],
                ep = -1;
                fd = -1;
            ];
          
            (* The ODBC Microsoft Access driver and ODBC Excel driver don't support inline multirow inserts;
             * issue a message if detected.
             *)
            If[params === {{}} && StringMatchQ[rdbms, Alternatives @@ {"Microsoft Access*", "Microsoft Excel*"}]
                && StringMatchQ[sql, RegularExpression["^(?i)INSERT\\s.*VALUES\\s+(\\(.+\\),\\s*)+\\(.+\\)$"]],
                Message[SQLExecute::multirowodbc];
            ];

            (* Execute the JDBC prepared statement with parameters that can be handled by Java *)
            LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
          
            (* You can enable streaming result sets on many drivers by setting the appropriate options
             * (e.g. "ForwardOnly"/TYPE_FORWARD_ONLY for Oracle).
             * However MySQL requires some weird settings and a special unprepared statement code path.
             *)
            If[rrs === {"MySQLStreaming"} && params === {{}} && supportsMySQLStreamingQ[connection],
                (* rst set above *)
                rsc = 1007; (* present default, hard-coded in SQLExecute *)
                fs = -2^31; (* Integer.MIN_VALUE, required by driver, note Java has to handle values < 0 *)
                result = SQLStatementProcessor`processUnpreparedSQLStatement[
                    connection, sql, TrueQ[gas], TrueQ[sch], TrueQ[rrsBool], TrueQ[rgk], maxrows, timeout, rst, rsc, ep, fd, fs, mfs];,

                (* else *)
                (* Batch the inserts on the Mathematica side to reduce JLink memory usage.
                 * This could potentially cause problems if someone asks for a result set on an insert operation,
                 * but the Java side can't deal with that anyway.
                 *
                 * Avoid Partition[] on params here.
                 *)
                If[!TrueQ[jb] && bs > 0 && Length@params > 1,
                    takeIndices = With[{chunks = Ceiling[Length[params]/bs]},
                        Table[{i*bs + 1, Min[(i + 1)*bs, Length[params]]}, {i, 0, chunks - 1}]
                    ];
                    result = Flatten[Map[
                        With[{piece = Take[params, #]}, SQLStatementProcessor`processSQLStatement[
                            connection, sql, piece, TrueQ[gas], TrueQ[sch], TrueQ[rrsBool], TrueQ[rgk], clearParams,
                            maxrows, timeout, rst, rsc, ep, fd, fs, mfs, Length@piece]] &,
                        takeIndices
                    ], 1],
    
                    (* else *)
                    (* Escape processing can be set to false only for unprepared statements *)
                    If[ep != 1 && params == {{}},
                        result = SQLStatementProcessor`processUnpreparedSQLStatement[
                            connection, sql, TrueQ[gas], TrueQ[sch], TrueQ[rrsBool], TrueQ[rgk],
                            maxrows, timeout, rst, rsc, ep, fd, fs, mfs],
                        (* else *)
                        result = SQLStatementProcessor`processSQLStatement[
                            connection, sql, params, TrueQ[gas], TrueQ[sch], TrueQ[rrsBool], TrueQ[rgk], clearParams,
                            maxrows, timeout, rst, rsc, ep, fd, fs, mfs, bs]
                    ]
                ];
            ];

            Which[
                MatchQ[result, {_Integer}],
                First[result],
                
                MatchQ[result, {_?JavaObjectQ}],
                KeepJavaObject[First[result]];
                First[result],

                cs === None,
                result,
                
                cs === Automatic || MatchQ[cs, _Function],
                If[TrueQ[sch],
                    cols = First[result];
                    results = Drop[result, 1],
                    cols = Null;
                    results = result
                ];
                If[cs === Automatic,
                    setColumnSymbols["Global`", cols, results],
                    cs[cols, results]
                ];
                result,
                
                MatchQ[cs, {___Symbol}],
                If[TrueQ[sch], 
                    Evaluate[cs] = Transpose[Drop[result, 1]],
                    Evaluate[cs] = Transpose[result]
                ];
                result
            ]
        ] (* Catch *)
    ] (* Block *)
]]


SQLExecute[queryName_String] := Module[
    {query, pos},
    If[MemberQ[SQLQueryNames[], queryName], 
      pos = First[Flatten[Position[SQLQueryNames[], queryName]]];
      query = SQLQueries[][[pos]];
    ];
    SQLExecute[query]
]


(* 
 * This pattern (I think undocumented) manages its own connection and is used 
 * by e.g. DatabaseExplorer to programmatically construct queries that
 * will work without an active connection. --dillont
 *)
SQLExecute[SQLSelect[connName:(_String|_JDBC),
                     table:(_SQLTable | {__SQLTable} | _String | {__String}),
                     columns:(_SQLColumn | {__SQLColumn}),
                     condition_,
                     opts:OptionsPattern[]]] := Module[
    {conn = OpenSQLConnection[connName], data = {}},
    data = SQLSelect[conn, table, columns, condition, opts];
    CloseSQLConnection[conn];
    data
]

SQLExecute[conn_SQLConnection, 
           SQLSelect[connName:(_String|_JDBC),
                     table:(_SQLTable | {__SQLTable} | _String | {__String}),
                     columns:(_SQLColumn | {__SQLColumn}),
                     condition_,
                     opts:OptionsPattern[]]] :=
    SQLSelect[conn, table, columns, condition, opts]

getClearStatementParameter[rdbms_, clearParams_] := Switch[clearParams,
  			Automatic, If[rdbms == "SQLite", False, True],
  			True, True,
  			_, False
]																												
 
formatSQL[ps_String, args_List] := Module[
    {params, posList, otherPosList, sql, sortedPosList, indexes, somePosList, stringPosList},
  
    If[MatchQ[args, {__List}],
      params = First[args], 
      params = args
    ];
    (* Check to see if anything needs to be replaced *) 
    posList = StringPosition[ps, "`" <> ToString[#] <> "`"] & /@ Range[Length[params]];
    otherPosList = StringPosition[ps, "``"];
    AppendTo[posList, otherPosList];
    If[Flatten[posList] === {}, 
      sql = ps;
      If[!MatchQ[args, {__List}],
        params = {args},
        params = args
      ],        
      (* If things need to be replaced, get a list of indexes for each replacement. 
         Then replace the values and get a set of parameters that can be handled by Java *)
      sortedPosList = Sort[Flatten[posList, 1]];
      indexes = (First[Flatten[Position[posList,#]]] & /@ sortedPosList); 
      somePosList = Flatten[Position[indexes, Length[params] + 1]];
      indexes = If[# === Length[params] + 1, idx++; idx - 1, idx = # + 1; #] & /@ indexes;
      stringPosList = "`" <> ToString[indexes[[#]]] <> "`" & /@ somePosList;    
      If[MatchQ[args, {__List}],           
        {sql, params} = Transpose[formatSQL[StringReplacePart[ps, stringPosList, otherPosList], #, indexes] & /@ args];
        sql = First[sql], 
        {sql, params} = formatSQL[StringReplacePart[ps, stringPosList, otherPosList], args, indexes];
        params = {params};
      ];
    ];
    {sql, params}
];

formatSQL[stmt_String, params_List, indexes_List] := Module[
    {newStmt = stmt, newParams = {}, j, i, val, localStmt, localParams, 
        timesStmt, timesParams, plusStmt, plus, minus, operator,
        inequalityStmt = "", inequalityParams = {}},
    For[j = 1, j <= Length[indexes], j++,
      i = indexes[[j]];
      val = params[[i]];
      Switch[val, 
        _Integer | _Real | _String | True | False | 
        Null | _SQLBinary | _DateObject | _TimeObject | _SQLDateTime | _SQLExpr, 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "?"}];
          newParams = Append[newParams, val], 
        SQLTable[_String, ___?OptionQ], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> First[val]}], 
        SQLColumn[_String, ___?OptionQ], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> First[val]}],
        SQLColumn[{_String, _String}, ___?OptionQ], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> val[[1, 1]] <> "." <> val[[1,2]]}], 
        SQLArgument[_List], 
          {localStmt, localParams} = formatSQL[ "(`1`)", {SQLArgument@@val[[1]]}, {1}];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],
        _SQLArgument,
          {localStmt, localParams} = formatSQL[ generateBinaryStatementNoParens[",", Length[val]], List@@val, Range[Length[val]]];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],
        SQLStringMatchQ[SQLColumn[_String, ___?OptionQ], _String], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[1,1]] <> " LIKE ?)"}];
          newParams = Append[newParams, val[[2]]],
        SQLStringMatchQ[SQLColumn[{_String, _String}, ___?OptionQ], _String], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[1, 1, 1]] <> "." <> val[[1,1,2]] <> " LIKE ?)"}];
          newParams = Append[newParams, val[[2]]],
        SQLMemberQ[_List, _SQLColumn], 
          {localStmt, localParams} = formatSQL[ "(`1` IN (`2`))", {val[[2]], SQLArgument@@val[[1]]}, {1, 2}];          
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],
        Or[_, __],         
          {localStmt, localParams} = formatSQL[generateBinaryStatement["OR", Length[val]], List@@val, Range[Length[val]]];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],
        And[_, __], 
          {localStmt, localParams} = formatSQL[generateBinaryStatement["AND", Length[val]], List@@val, Range[Length[val]]];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],
        Not[SQLStringMatchQ[SQLColumn[_String, ___?OptionQ], _String]], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[1,1,1]] <> " NOT LIKE ?)"}];
          newParams = Append[newParams, val[[1, 2]]],
        Not[SQLStringMatchQ[SQLColumn[{_String, _String}, ___?OptionQ], _String]], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[1, 1, 1, 1]] <> "." <> val[[1,1,1,2]] <> " NOT LIKE ?)"}];
          newParams = Append[newParams, val[[1, 2]]],
        Not[SQLMemberQ[_List, _SQLColumn]], 
          {localStmt, localParams} = formatSQL[ "(`1` NOT IN (`2`))", {val[[1, 2]], SQLArgument@@val[[1, 1]]}, {1, 2}];          
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],
        Not[_], 
          {localStmt, localParams} = formatSQL[ "(NOT `1`)", {First[val]}, {1}];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams], 
        Equal[SQLColumn[_String, opts:OptionsPattern[]], Null], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[1,1]] <> " IS NULL)"}],
        Equal[SQLColumn[{_String, _String}, opts:OptionsPattern[]], Null], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[1, 1, 1]] <> "." <> val[[1,1,2]] <> " IS NULL)"}],
        Equal[Null, SQLColumn[_String, opts:OptionsPattern[]]], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[2,1]] <> " IS NULL)"}],
        Equal[Null, SQLColumn[{_String, _String}, opts:OptionsPattern[]]], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[2, 1, 1]] <> "." <> val[[2,1,2]] <> " IS NULL)"}],
        Unequal[SQLColumn[_String, opts:OptionsPattern[]], Null], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[1,1]] <> " IS NOT NULL)"}],
        Unequal[SQLColumn[{_String, _String}, opts:OptionsPattern[]], Null], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[1, 1, 1]] <> "." <> val[[1,1,2]] <> " IS NOT NULL)"}],
        Unequal[Null, SQLColumn[_String, opts:OptionsPattern[]]], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[2,1]] <> " IS NOT NULL)"}],
        Unequal[Null, SQLColumn[{_String, _String}, opts:OptionsPattern[]]], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(" <> val[[2, 1, 1]] <> "." <> val[[2,1,2]] <> " IS NOT NULL)"}],
        Equal[_, __] | Unequal[_, __] | LessEqual[_, __] | GreaterEqual[_, __] | Less[_, __] | Greater[_, __], 
          operator = Switch[Head[val], 
            Equal, "=", 
            Unequal, "!=", 
            LessEqual, "<=",
            GreaterEqual, ">=", 
            Less, "<", 
            Greater, ">"
          ];
          {localStmt, localParams} = formatSQL[generateEqualStatement[operator, Length[val]], List@@val, Range[Length[val]]];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams], 
        _Inequality,
          If[Length[val] > 4 && OddQ[Length[val]],  
            Do[If[OddQ[i], 
                 inequalityStmt = inequalityStmt <> 
                   Switch[i,
                     1, "(`" <> ToString[(i + 1)/2] <> "`", 
                     Length[val], "`" <> ToString[(i + 1)/2] <> "`)",
                     _, "`" <> ToString[(i + 1)/2] <> "`) AND (`" <> ToString[(i + 1)/2] <> "`"
                   ]; 
                 AppendTo[inequalityParams, val[[i]]]
                 ,
                 inequalityStmt = inequalityStmt <> 
                   Switch[val[[i]], 
                     Equal, " = ",
                     Unequal, " != ", 
                     LessEqual, " <= ", 
                     GreaterEqual, " >= ", 
                     Less, " < ", 
                     Greater, " > "]
               ], {i, Length[val]}];          
            {localStmt, localParams} = formatSQL[inequalityStmt, inequalityParams, Range[Length[inequalityParams]]];
            newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
            newParams = Join[newParams, localParams]
            ,
            Message[SQLValue::illegalvalue, val];
            Throw[$Failed] 
          ],
        Times[-1, SQLColumn[_String, opts:OptionsPattern[]]], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "-" <> val[[2,1]]}],          
        Times[-1, SQLColumn[{_String, _String}, ___?OptionQ]], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "-" <> val[[2, 1, 1]] <> "." <> val[[2,1,2]]}],          
        Times[_, __], 
          Switch[Numerator[val], 
            _Times,  
              timesStmt = "(" <> generateBinaryStatementNoParens["*", Length[Numerator[val]]] <> ")";
              timesParams = List@@ Numerator[val],       
            _, 
              timesStmt = "`1`";
              timesParams = {Numerator[val]};
          ];             
          Switch[Denominator[val], 
            1, 
              Null,
            _Times,  
              timesStmt = timesStmt <> " / (" <> generateBinaryStatementNoParens["*", Length[timesParams] + 1, Length[Denominator[val]] + Length[timesParams]] <> ")";
              timesParams = Join[timesParams, List@@ Denominator[val]],
            _, 
              timesStmt = timesStmt <> " / `" <> ToString[Length[timesParams] + 1] <> "`";
              timesParams = Join[timesParams, {Denominator[val]}];
          ];
          {localStmt, localParams} = formatSQL[timesStmt, timesParams, Range[Length[timesParams]]];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],              
        Power[SQLColumn[_String, opts:OptionsPattern[]], -1], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(1/" <> val[[1,1]] <> ")"}],
        Power[SQLColumn[{_String, _String}, ___?OptionQ], -1], 
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> "(1/" <> val[[1, 1, 1]] <> "." <> val[[1,1,2]] <> ")"}],
        Power[SQLColumn[_String, opts:OptionsPattern[]], x_Integer?Positive], 
          {localStmt, localParams} = formatSQL[ "(`1` * `2`)", {First[val], Power[val[[1]], val[[2]] - 1]}, {1, 2}];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],      
        Rational[ _Integer, _Integer ], 
          {localStmt, localParams} = formatSQL[ "(`1` / `2`)", {Numerator[val], Denominator[val]}, {1, 2}];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],                    
        Plus[_, __], 
          {plus, minus} = splitPlus[val];
          Switch[Length[plus], 
            0, 
              plusStmt = "", 
            1, 
              plusStmt = "`1`",
            x_/;x > 1,  
              plusStmt = generateBinaryStatementNoParens["+", Length[plus]];
          ];             
          Switch[Length[minus], 
            0, 
              Null,
            1,  
              plusStmt = plusStmt <> " - `" <> ToString[Length[plus] + 1] <> "`",
            _, 
              plusStmt = plusStmt <> " - " <> generateBinaryStatementNoParens["-", Length[plus] + 1, Length[plus] + Length[minus]];
          ];
          {localStmt, localParams} = formatSQL[plusStmt, Join[plus, minus], Range[Length[Join[plus, minus]]]];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],                        
        SQLColumn[_String, opts:OptionsPattern[]] -> "Ascending",
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> val[[1, 1]] <> " ASC"}],          
        SQLColumn[{_String, _String}, opts:OptionsPattern[]] -> "Ascending",
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> val[[1, 1, 1]] <> "." <> val[[1,1,2]] <> " ASC"}],          
        SQLColumn[_String, opts:OptionsPattern[]] -> "Descending",
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> val[[1, 1]] <> " DESC"}],          
        SQLColumn[{_String, _String}, opts:OptionsPattern[]] -> "Descending",
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> val[[1, 1, 1]] <> "." <> val[[1,1,2]] <> " DESC"}], 
        Rule[_SQLColumn, _], 
          {localStmt, localParams} = formatSQL[ "`1` = `2`", {First[val], Last[val]}, {1, 2}];
          newStmt = StringReplace[newStmt, {"`" <> ToString[i] <> "`" -> localStmt}];
          newParams = Join[newParams, localParams],
        _,
          Message[SQLValue::illegalvalue, val];
          Throw[$Failed]
      ];
    ];
    {newStmt, newParams}
]


generateBinaryStatement[ operator_String, length_Integer] := Module[{table},
    table = Reverse[Table["`" <> ToString[i] <> "`", {i, length}]]; 
    Fold["(" <> ToString[#2] <> " " <> operator <> " " <> ToString[ #1] <> ")" &, First[table], Rest[table]]    
]

generateBinaryStatementNoParens[ operator_String, length_Integer] := Module[{table},
    table = Table["`" <> ToString[i] <> "`", {i, length}]; 
    Fold[ToString[#1] <> " " <> operator <> " " <> ToString[ #2] &, First[table], Rest[table]]  
]
  
generateBinaryStatementNoParens[ operator_String, min_Integer, max_Integer] := Module[{table},
    table = Table["`" <> ToString[i] <> "`", {i, min, max}]; 
    Fold[ToString[#1] <> " " <> operator <> " " <> ToString[ #2] &, First[table], Rest[table]]  
]

generateEqualStatement[operator_String, length_Integer] := Module[{table},
    table = Reverse[Table["`" <> ToString[i] <> "`", {i, length}]];
    Fold[(
      Switch[#2, 
        #1, #2 <> ")", 
        Last[table],  "(" <> #2 <> " " <> operator <> " " <> #1 ,
        _, #2 <> ") AND (" <> #2 <> " " <> operator <> " " <> #1
      ]) &, First[table], table]
]
  
splitPlus[x_Plus] := Module[{list}, 
    list = List @@ x;
    list = Map[If[MatchQ[#, Times[x1_ /; x1 < 0, ___] | x1_ /; x1 < 0], {{}, {-1 #}}, {{#}, {}}] &, list];
    list = Transpose[list];
    Map[Flatten, list]
]

(* (unused) *)
mingleObject[lst_List, object_] := If[lst === {}, 
    {}, 
    Drop[Flatten[Thread[{lst, object}, List, 1], 1, List], -1]
]
    
setColumnSymbols[context_String, cols_List | cols:Null, results_List] := Module[
    {columnSymbols = {}},
    If[cols === Null, 
      columnSymbols = Symbol[context <> "col" <> ToString[#]] & /@ Range[Length[First[results]]];,
      (* else *)
      columnSymbols = Symbol[context <> normalizeSymbolName[#]] & /@ cols;
    ];
    Evaluate[columnSymbols] = Transpose[results];
];

normalizeSymbolName[name_String] := StringReplace[name, {
	"."->"", "_"->"", "~"->"", "!"->"", 
    "@"->"", "#"->"", "$"->"", "%"->"", "^"->"", 
    "&"->"", "*"->"", "("->"", ")"->"", "-"->"", 
    "+"->"", "="->"", "{"->"", "["->"", "}"->"", 
    "]"->"", "|"->"", "\\"->"", ":"->"", ";"->"",
    "\""->"", "\'"->"", "<"->"", ","->"", ">"->"",
    "?"->"", "/"->"", " "->""
}]

End[] (* `SQL`Private` *)
