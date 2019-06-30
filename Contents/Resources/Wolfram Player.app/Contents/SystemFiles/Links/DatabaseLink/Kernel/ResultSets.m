(*===================================================================*)
(*========================== ResultSets  ============================*)
(*===================================================================*)

Begin["`SQL`Private`"] 

SQLResultSet::forwardonly = "This SQLResultSet is required to be ForwardOnly."

SQLResultSetOpen::mode = "Invalid SQLResultSet mode: `1`"

SQLResultSetTake::invalidrange = "Invalid range: `1`"

Options[SQLResultSet] = {
    "FetchDirection" -> Automatic,
    "FetchSize" -> Automatic
}
    
Options[SQLResultSetOpen] = JoinOptions[
	{
		"Mode"->"ScrollInsensitive"
	},
    Options[SQLResultSet]
]

Options[SQLResultSetRead] = { 
    "GetAsStrings" -> False
}

Options[SQLResultSetCurrent] = {
    "GetAsStrings" -> False
}

Options[SQLResultSetTake] = { 
    "GetAsStrings" -> False
}


$resultSetIndex = 0;

If[!ListQ[$resultSets],
    $resultSets = {};
];

SQLResultSets[] := $resultSets;

SetAttributes[SQLResultSetOpen, HoldFirst]

SQLResultSetOpen[ (s_SQLSelect | s_SQLExecute), opts:OptionsPattern[]] := Module[
    {resultSet, useOpts, mode, fetchDir, fetchSize}, 
 
    useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLResultSetOpen]];
    mode = Lookup[useOpts, "Mode"];
    fetchDir = Lookup[useOpts, "FetchDirection"];
    fetchSize = Lookup[useOpts, "FetchSize"];
    
    resultSet = ReleaseHold[Insert[Hold[s], "ResultSet" -> {mode}, {1, -1}]];
    If[!JavaObjectQ[resultSet] || resultSet === $Failed, Return[$Failed]];
    resultSet = SQLResultSet[$resultSetIndex++, resultSet];
    AppendTo[$resultSets, resultSet];
    SetSQLResultSetOptions[resultSet, "FetchDirection" -> fetchDir, "FetchSize" -> fetchSize];
    resultSet
]


SQLResultSet /: SetOptions[SQLResultSet[id_Integer, rs_?JavaObjectQ], opts___] := 
    SetSQLResultSetOptions[SQLResultSet[id, rs], opts]


SetSQLResultSetOptions[SQLResultSet[id_Integer, rs_?JavaObjectQ], opts:OptionsPattern[]] := Module[
    {fs, fd, optTest, useOpts},
  
    optTest = FilterRules[{opts}, Except[Options[SQLResultSet]]];
    If[optTest =!= {}, optionsErrorMessage[optTest, SQLResultSetOpen, SQLResultSet]; Return[$Failed]];

    useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLResultSet]];
    {fs, fd} = Lookup[useOpts, {"FetchSize", "FetchDirection"}]; 
         
    Switch[fd,
      Automatic, Null,
      "Forward", rs@setFetchDirection[1000],
      "Reverse", rs@setFetchDirection[1001], 
      "Unknown", rs@setFetchDirection[1002],
      _, 
        Message[SQLExecute::fetchdirection, fd];
    ];

    If[fs =!= Automatic && fs =!= All, 
      If[fs === None, fs = 0];
      If[!IntegerQ[fs] || fs < 0, 
        Message[SQLExecute::fetchsize, fs]
      ],fs = 0;
    ];
    rs@setFetchSize[fs];
    SQLResultSet[ id, rs]
]

SQLResultSetClose[ SQLResultSet[ id_Integer, rs_?JavaObjectQ ] ] := Module[
    {}, 
    If[ ( MemberQ[ $resultSets, SQLResultSet[ id, rs ] ] ), 
      $resultSets = Drop[$resultSets, First@Position[$resultSets, SQLResultSet[id, rs]]]; 
      rs@close[];
      ReleaseJavaObject[rs];
    ] 
]
  
SQLResultSetCurrent[ SQLResultSet[ _Integer, rs_?JavaObjectQ], opts:OptionsPattern[] ] := Module[
    {useOpts, gas = False, results},
    Block[{$JavaExceptionHandler = ThrowException},     
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLResultSetCurrent]];
            gas = Lookup[useOpts, "GetAsStrings"]; 
      
            results = SQLStatementProcessor`getLimitedResultData[0, rs, gas];
            If[ListQ[results] && Length[results] > 0, First[results], results]
        ]
    ]
]

SQLResultSetRead[ rs_SQLResultSet, opts:OptionsPattern[] ] := Module[
    {results}, 
    results = SQLResultSetRead[rs, 1, opts];
    If[ListQ[results] && Length[results] > 0, First[results], results]
]

SQLResultSetRead[ rs_SQLResultSet, 0, opts:OptionsPattern[] ] := {}

SQLResultSetRead[ SQLResultSet[ _Integer, rs_?JavaObjectQ], limit_Integer, opts:OptionsPattern[]] := Module[
    {useOpts, gas = False},
    Block[{$JavaExceptionHandler = ThrowException},
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLResultSetRead]];
            gas = Lookup[useOpts, "GetAsStrings"]; 
      
            If[limit < 0 && rs@getType[] == 1003,
                Message[SQLResultSet::forwardonly];
                Return[$Failed];
            ];            
            SQLStatementProcessor`getLimitedResultData[limit, rs, gas]
        ]
    ]
]

SQLResultSetTake[rs_SQLResultSet, {start_}, opts:OptionsPattern[]] :=
    SQLResultSetTake[rs, {start, start}, opts]  

SQLResultSetTake[SQLResultSet[ _Integer, rs_?JavaObjectQ], {start_Integer, end_Integer}, opts:OptionsPattern[]] := Module[
    {useOpts, gas = False, eRow = 0, sRow = 0, diff, current},
    Block[{$JavaExceptionHandler = ThrowException},
        Catch[
            current = rs@getRow[];
        
            If[rs@getType[] == 1003,
              Message[SQLResultSet::forwardonly];
              Return[$Failed];
            ];            

            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLResultSetTake]];
            gas = Lookup[useOpts, "GetAsStrings"]; 
            
            If[start == 0, 
                Message[SQLResultSetTake::invalidrange, {start, end}]; Return[$Failed]
            ];

            If[end =!= 0, 
                If[!rs@absolute[end], 
                    resetCursor[rs, current];
                    Message[SQLResultSetTake::invalidrange, {start, end}];
                    Return[$Failed]
                ];
                eRow = rs@getRow[], 
                Message[SQLResultSetTake::invalidrange, {start, end}];
                Return[$Failed]
            ];

            If[!rs@absolute[start],
                resetCursor[rs, current];
                Message[SQLResultSetTake::invalidrange, {start, end}];
                Return[$Failed];
            ];
            sRow = rs@getRow[];
        
            diff = eRow - sRow;
            Which[
                diff < 0, 
                diff = diff - 1;
                If[!rs@relative[1], rs@afterLast[]],
                
                diff > 0, 
                diff = diff + 1;
                If[!rs@relative[-1], rs@beforeFirst[]]
            ];

            SQLStatementProcessor`getLimitedResultData[diff, rs, gas]
        ]
    ]
]

resetCursor[rs_, position_Integer] := Module[
    {}, 
    If[!rs@absolute[position], 
        If[position == 0, rs@beforeFirst[], rs@afterLast[]]
    ]
]

SQLResultSetShift[ SQLResultSet[ _Integer, rs_?JavaObjectQ], rows_Integer] := Module[
	{valid = False},
    Block[
    	{$JavaExceptionHandler = ThrowException},     
        Catch[
            If[rs@getType[] == 1003, (* This is TYPE_FORWARD_ONLY *)
                If[rows < 0, 
                    Message[SQLResultSet::forwardonly];
                    Return[$Failed];
                ];
                For[i = 0, i < rows, i++, valid = rs@next[]]; 
                valid,
                
                rs@relative[rows]        
            ]
        ]
    ]
]
  
SQLResultSetGoto[ SQLResultSet[ _Integer, rs_?JavaObjectQ], row_Integer | row:Infinity] := Block[
	{$JavaExceptionHandler = ThrowException},     
    Catch[
        If[rs@getType[] == 1003,
            Message[SQLResultSet::forwardonly];
            Return[$Failed];
        ];   
        Switch[row, 
            0, rs@beforeFirst[]; False,
            
            Infinity, rs@afterLast[]; False,
             
            _, rs@absolute[row]
        ] 
    ]
]

SQLResultSetPosition[ SQLResultSet[ _Integer, rs_?JavaObjectQ]] := Block[
	{$JavaExceptionHandler = ThrowException},
    Catch[rs@getRow[]]
]
    
SQLResultSetColumnNames[ SQLResultSet[ _Integer, rs_?JavaObjectQ]] := Block[
	{$JavaExceptionHandler = ThrowException},     
    Catch[
      SQLStatementProcessor`getHeadings[ rs, True]
    ]
]
  

SQLResultSet /: MakeBoxes[
    SQLResultSet[
        id_Integer,
        rs_Symbol,
        opts___?BoxForm`HeldOptionQ
    ],
    StandardForm] := Module[
    
    {icon = summaryBoxIcon, mode = "", fd = "", fs = "", sometimesOpts, 
        o = canonicalOptions[Join[Flatten[{opts}], Options[SQLResultSet]]], oPrime},

    If[JavaObjectQ[rs] && rs =!= Null,
        (* fs = ... *)
        mode = Quiet@Switch[rs@getType[],
            1003, "ForwardOnly",
            1004, "ScrollInsensitive",
            1005, "ScrollSensitive",
            _, Style["Unknown Mode", {Italic, GrayLevel[0.55]}]
        ];
        fd = Quiet@Switch[rs@getFetchDirection[],
            1000, "Forward",
            1001, "Reverse",
            1002, "Unknown",
            _, ""
        ];
        fs = With[{i = Quiet[rs@getFetchSize[]]}, If[IntegerQ[i], i, ""]];,
        (* else *)
        mode = Style["Closed", {Italic, GrayLevel[0.55]}]
    ];

    sometimesOpts = Sort[
        DeleteCases[Options[SQLResultSet][[All, 1]], Alternatives @@ {
            "FetchDirection"
        }]
    ];

    oPrime = FilterRules[Join[{"FetchSize" -> fs}, o], Options[SQLResultSet]];
    
    BoxForm`ArrangeSummaryBox[SQLResultSet, 
        SQLResultSet[id, rs, opts],
        icon,
        (* Always *)
        {
            {BoxForm`SummaryItem[{"Mode: ", mode}], BoxForm`SummaryItem[{"ID: ", id}]},
            {BoxForm`SummaryItem[{"FetchDirection: ", fd}], ""}
        },
        (* Sometimes *)
        BoxForm`SummaryItem[{# <> ": ", Replace[Lookup[oPrime, #], {Null -> "", None -> ""}]}] & /@ sometimesOpts,

        StandardForm
    ]
]


End[] (* `SQL`Private` *)
