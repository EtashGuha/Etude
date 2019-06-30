(* Wolfram Language package *)

Package["Databases`Common`"]

PackageImport["Databases`"]
PackageImport["PacletManager`"]


(* These are exported for development purposes *)
PackageExport["$DBDebugMode"]
PackageExport["$DBShowStackTraceInDebug"]
PackageExport["$DBGenerateInternalErrorMessage"]
PackageExport["$DBErrorHandler"]
PackageExport["$DBDefaultErrorHandler"]
PackageExport["$DBDefaultErrorHandler"]
PackageExport["DBDebugEnv"]
PackageExport["$DBDebugEnv"]
PackageExport["$DBOneLevelOnlyDebugErrorHandlerInjection"]


PackageExport["DBGetErrorType"]
PackageExport["DBGetFailingFunction"]
PackageExport["DBGetFailingFunctionArguments"]

PackageExport["DBRaise"]
PackageExport["DBHandleError"]
PackageExport["DBDefError"]
PackageExport["DBDef"]
PackageExport["DBError"]
PackageExport["$DBInErrorHandler"]



If[
    ! ValueQ[$DBDebugMode],
    $DBDebugMode := DBCheckAndSetCache[
        i$DBDebugMode,
        Databases`Private`$DevelopmentMode
    ]
]

$DBShowStackTraceInDebug := False;

$DBGenerateInternalErrorMessage := $DBDebugMode;

$DBErrorHandler = None;

$DBInErrorHandler = False;

$DBOneLevelOnlyDebugErrorHandlerInjection = True;


ClearAll[DBGetErrorType, DBGetFailingFunction, DBGetFailingFunctionArguments];
DBGetErrorType[Failure[type_, ___]] := type;
DBGetFailingFunction[Failure[_, assoc_]] := assoc["FailingFunction"];
DBGetFailingFunctionArguments[Failure[_, assoc_]] := assoc["FailingFunctionArgs"];

ClearAll[DBRaise];
DBRaise[
    f_,
    type_String: "DatabaseInternalFailure",
    args_List: {},
    extra_: <||>
] :=
    DBRaise @ Failure[
        type, <|
            "MessageTemplate"   -> "Error of type `ErrorType` in internal function `Function`",
            "MessageParameters" -> <|"ErrorType" -> type, "Function" -> HoldForm[f]|>,
            "FailingFunction" :> f,
            "FailingFunctionArgs" :> args,
            extra
        |>
    ]

DBRaise[Failure[tag_, assoc: _?AssociationQ: <||>]] /; $DBShowStackTraceInDebug :=
    markEnd @ Throw[
        Failure[tag, <|assoc, "Stack" -> DBFormattedStackTrace[]|>],
        DBError
    ]

DBRaise[f_Failure] :=
    Throw[f, DBError]


ClearAll[DBHandleError];
Options[DBHandleError] = {
    "GenerateErrorMessage" :> $DBGenerateInternalErrorMessage
};

DBHandleError[
    topLevelF: _Symbol: None,
    handler: Except[_?OptionQ] : Automatic,
    opts: OptionsPattern[]
] :=
    With[{genmsg = TrueQ @ OptionValue["GenerateErrorMessage"]},
        Function[
            code,
            If[TrueQ @ $DBDebugMode, StackComplete, Identity] @ Block[
                {
                    $DBGenerateInternalErrorMessage = genmsg,
                    $DBErrorHandler = $DBErrorHandler
                }
                ,
                (* If explicit handler was passed, it will be assigned to $DBErrorHandler
                in the production mode. In the debug mode, the same will happen unless
                $DBErrorHandler has been defined in the outer scope. This is needed
                to be able to inject a different error handler in the debug mode
                *)
                If[handler =!= Automatic && Or[
                    !TrueQ[$DBDebugMode],  (* Not in debug mode *)
                    $DBErrorHandler === None, (* No $DBErrorHandler defined in the outer scope *)
                    And[ (* can't inject, only first - level calls can be injected in this mode *)
                       TrueQ @ $DBInErrorHandler,
                       TrueQ @ $DBOneLevelOnlyDebugErrorHandlerInjection
                    ]]
                    ,
                    $DBErrorHandler = handler
                ];
                (* If $DBErrorHandler is still undefined, fall back to default handler *)
                If[MatchQ[$DBErrorHandler, Automatic | None],
                    $DBErrorHandler = $DBDefaultErrorHandler
                ];
                Catch[
                    Block[{$DBInErrorHandler = True},
                        markStart[code]
                    ],
                    DBError,
                    Function[{value, tag},
                        $DBErrorHandler[topLevelF, value]
                    ]
                ]
            ],
            HoldAll
        ]
    ];

(* $DBDefaultErrorHandler =
    Function[{topLevelF, failure},
        If[TrueQ @ $DBDebugMode && TrueQ @ $DBGenerateInternalErrorMessage,
            Message[
                Databases`Databases::interr,
                DBGetErrorType[failure],
                DBGetFailingFunction[failure],
                DBGetFailingFunctionArguments[failure]
            ]
        ];
        $Failed
    ]; *)

$DBDefaultErrorHandler = Function[{topLevelF, failure}, failure];

DBDebugEnv[handler_] :=
    Function[
        code,
        Block[{
            $DBDebugMode = True,
            $DBShowStackTraceInDebug = True,
            $DBErrorHandler = handler
            },
            code
        ],
        HoldAll
    ];

(* A shortcut *)
$DBDebugEnv := DBDebugEnv[$DBDefaultErrorHandler]

badargFunction[f_Symbol] := 
    Switch[
        Attributes[f],
        _?(MemberQ[HoldAllComplete]),
            Map[HoldComplete],
        _?(MemberQ[HoldFirst | HoldAll | HoldRest]),
            Map[HoldForm],
        _,
            Identity
    ]


DBDefError[errorType_String][f_Symbol] :=
    f[args___] := DBRaise[
        f, 
        errorType, 
        badargFunction[f] @ Unevaluated[{args}]
    ]

DBDefError[errorType_String][f_Symbol[]] :=
    f[params___][args___] := DBRaise[
        f, 
        errorType, 
        {badargFunction[f] @ Unevaluated[{params}], {args}}
    ]

DBDefError[f: _Symbol | _Symbol[]] := DBDefError["badargs"][f]

DBDefError[syms: (_Symbol | _Symbol[])..] :=
    Scan[DBDefError, {syms}]
