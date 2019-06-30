(*===================================================================*) 
(*======================== Transactions =============================*) 
(*===================================================================*) 

Begin["`SQL`Private`"] 

SQLBeginTransaction::nested = "Nested transactions are not allowed.  Continuing with the first transaction."

SQLSetSavepoint::version = "This feature requires Java 1.4."

SQLReleaseSavepoint::sqlsavepoint = "Invalid sqlsavepoint: `1`"

SQLReleaseSavepoint::javasavepoint = "Invalid Java savepoint: `1` (in `2`)"


Options[SQLSavepoint] = {
    "Name" -> ""
}


$inTransaction = False;

SQLBeginTransaction[SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ]] := (
    If[!JavaObjectQ[connection], 
        Message[SQLConnection::conn];
        Return[$Failed]
    ];    
    If[$inTransaction,
        Message[SQLBeginTransaction::nested],
        $inTransaction = True;
        connection@setAutoCommit[False]
    ]
)

SQLCommitTransaction[SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ]] := (
    If[!JavaObjectQ[connection], 
        Message[SQLConnection::conn];
        Return[$Failed]
    ];    
    connection@commit[];
    $inTransaction = False;
    connection@setAutoCommit[True];
)

SQLRollbackTransaction[SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ]] := (
    If[!JavaObjectQ[connection], 
        Message[SQLConnection::conn];
        Return[$Failed]
    ];    
    If[!connection@getAutoCommit[], 
        connection@rollback[];
        $inTransaction = False;
        connection@setAutoCommit[True];
    ]
)

SQLRollbackTransaction[SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ], 
                       SQLSavepoint[savepoint_, opts:OptionsPattern[]]] := (
    If[!JavaObjectQ[connection], 
        Message[SQLConnection::conn];
        Return[$Failed]
    ];    
    If[!JavaObjectQ[savepoint], 
        Message[SQLRollbackTransaction::savepoint];
        Return[$Failed]
    ];    
    If[!connection@getAutoCommit[], 
        connection@rollback[savepoint];
    ]
)

SetAttributes[SQLReleaseSavepoint, HoldRest]

SQLReleaseSavepoint[
    SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ], 
    sqlSavepoint_Symbol
] := Module[
    {javaSavepoint, spSymbolName},
    If[!JavaObjectQ[connection],
        Message[SQLConnection::conn];
        Return[$Failed]
    ];
        
    spSymbolName=SymbolName[Unevaluated[sqlSavepoint]];
    If[!MatchQ[sqlSavepoint, SQLSavepoint[_, _]],
        Message[SQLReleaseSavepoint::sqlsavepoint,spSymbolName];
        Return[$Failed]
    ];
        
    javaSavepoint = ReleaseHold[sqlSavepoint][[1]];
    If[!JavaObjectQ[javaSavepoint],
        Message[SQLReleaseSavepoint::javasavepoint,javaSavepoint,spSymbolName];
        Return[$Failed]
    ];
        
    connection@releaseSavepoint[javaSavepoint];
    Clear[sqlSavepoint]
]

SQLSetSavepoint[SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ]] := (
    If[!JavaObjectQ[connection], 
        Message[SQLConnection::conn];
        Return[$Failed]
    ];    
    Check[SQLSavepoint[connection@setSavepoint[]],
        Message[SQLSetSavepoint::version];
        $Failed,
        JLink`Java::nometh
    ]
)

SQLSetSavepoint[SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ], name_String] := (
    If[!JavaObjectQ[connection], 
        Message[SQLConnection::conn];
        Return[$Failed]
    ];    
    Check[SQLSavepoint[connection@setSavepoint[name], "Name" -> name],
        Message[SQLSetSavepoint::version];
        $Failed,
        JLink`Java::nometh
    ]
)


SQLSavepoint /: MakeBoxes[
    SQLSavepoint[
        sp_Symbol,
        opts___?BoxForm`HeldOptionQ
    ],
    StandardForm] := Module[
    
    {name, id = "", icon = summaryBoxIcon, sometimesOpts, 
        o = canonicalOptions[Join[Flatten[{opts}], Options[SQLSavepoint]]], oPrime},
    
    name = Lookup[o, "Name"]; 
    
    If[JavaObjectQ[sp] && sp =!= Null,
        id = Quiet[Replace[sp@getSavepointId[], $Failed -> ""]],
        (* else *)
        Null
    ];

    sometimesOpts = Sort[
        DeleteCases[Options[SQLSavepoint][[All, 1]], Alternatives @@ {
            "Name"
        }]
    ];

    oPrime = FilterRules[Join[{}, o], Options[SQLSavepoint]];
    
    BoxForm`ArrangeSummaryBox[SQLSavepoint,
        SQLSavepoint[sp, opts],
        icon,
        (* Always *)
        {
            {BoxForm`SummaryItem[{"Name: ", name}], BoxForm`SummaryItem[{"ID: ", id}]}
            (*,{BoxForm`SummaryItem[{"Status: ", status}], ""}*)
        }
        (* Sometimes *)
        , BoxForm`SummaryItem[{# <> ": ", Replace[Lookup[oPrime, #], {Null -> "", None -> ""}]}] & /@ sometimesOpts

        , StandardForm
    ]
]


SQLSavepoint /: MakeBoxes[
    SQLSavepoint[
        sp_Symbol,
        opts___?BoxForm`HeldOptionQ
    ],
    StandardForm] := Module[
    
    {name, id = "", icon = summaryBoxIcon, sometimesOpts, 
        o = canonicalOptions[Join[Flatten[{opts}], Options[SQLSavepoint]]], oPrime},
    
    name = Lookup[o, "Name"]; 
    
    If[JavaObjectQ[sp] && sp =!= Null,
        id = Quiet[Replace[sp@getSavepointId[], $Failed -> ""]],
        (* else *)
        Null
    ];

    sometimesOpts = Sort[
        DeleteCases[Options[SQLSavepoint][[All, 1]], Alternatives @@ {
            "Name"
        }]
    ];

    oPrime = FilterRules[Join[{}, o], Options[SQLSavepoint]];
    
    BoxForm`ArrangeSummaryBox[SQLSavepoint,
        SQLSavepoint[sp, opts],
        icon,
        (* Always *)
        {
            {BoxForm`SummaryItem[{"Name: ", name}], BoxForm`SummaryItem[{"ID: ", id}]}
            (*,{BoxForm`SummaryItem[{"Status: ", status}], ""}*)
        }
        (* Sometimes *)
        , BoxForm`SummaryItem[{# <> ": ", Replace[Lookup[oPrime, #], {Null -> "", None -> ""}]}] & /@ sometimesOpts

        , StandardForm
    ]
]

End[] (* `SQL`Private` *)
