Package["Databases`Common`"]

PackageImport["Databases`"]

PackageExport["DBSetDefaultStackTraceFormatting"]
PackageExport["DBStackFrameMakeBoxes"]
PackageExport["DBFormattedStackTrace"]
PackageExport["DBDisplayContainer"] 
PackageExport["$DBExecutableStackFrames"]
PackageExport["$DBShowStackTraceFormattingProgress"]

PackageScope["markStart"]
PackageScope["markEnd"]
PackageScope["DBFormattedStackTrace"]


$DBExecutableStackFrames = True

$DBShowStackTraceFormattingProgress = False

$stackFrameCounter = 0;
$stackSize = 1;

(* 
**  Marker functions. By using them in code, we can select the part of the stack 
**  that we really need, by inspecting the stack and only picking the part that 
**  is between these two functions (they will also be on the stack if called).
*)
markStart[code_] := code
markEnd[code_] := code


(* 
** Gets the stack and selects the relevant part of it
*)
stack[] :=
    Module[{s = Stack[_], startPos, endPos},
        startPos = If[MissingQ[#], 1, First @ # + 1 ] & @ FirstPosition[
            s,
            HoldForm[markStart[_]],
            Missing["NotFound"],
            1
        ];
        endPos = If[MissingQ[#], Length @ s, First @ # - 1 ] & @ FirstPosition[
            s,
            HoldForm[markEnd[_]],
            Missing["NotFound"],
            1
        ];
        If[endPos < startPos,
            Return[{}]
        ];
        Take[s, {startPos, endPos}]
    ]
    
    
(* ==============                 Stack formatting                 ===========*)

$defaultFormattingRules = DBGetMemoizedValue[$defaultFormattingRules, _List, {} ]

(*
**  Global registry for parts of expressions which should use some custom 
**  formatting, for the purposes of stack trace rendering.
*)
DBSetDefaultStackTraceFormatting[rule_RuleDelayed]:= Set[
    $defaultFormattingRules,
    DeleteDuplicates @ Append[$defaultFormattingRules, rule]
]
    
    
SetAttributes[simplisticExpressionRepr, HoldAllComplete]    
simplisticExpressionRepr[head_[___]] := HoldComplete[head["..."]]  
simplisticExpressionRepr[atom_]:= HoldComplete[atom] 
       
    
SetAttributes[DBDisplayContainer, HoldAll]
DBDisplayContainer[expr_] := expr


SetAttributes[getScopedSymbols, HoldAllComplete]
getScopedSymbols[Function[var:Except[Null, _Symbol], rest__]] :=  
    getScopedSymbols[Function[{var}, rest]]
    
getScopedSymbols[Function[vars:{__Symbol}, rest__]]:=
    getScopedSymbols[vars]
    
getScopedSymbols[(Module | With )[vars_List, _]]:=
    getScopedSymbols[vars]
    
getScopedSymbols[vars_List] := Map[getScopedSymbols, Unevaluated[vars]]

getScopedSymbols[Verbatim[Pattern][var_, _]] := getScopedSymbols[var]

getScopedSymbols[Set[var_Symbol, _]] := getScopedSymbols[var]

getScopedSymbols[var_Symbol] := Hold[var]

getScopedSymbols[_] := {}
    

SetAttributes[makeLongVarsReplacementRules, HoldAllComplete]
makeLongVarsReplacementRules[expr_]:= 
    Replace[
        DeleteDuplicates @ Flatten @ Cases[
            Unevaluated @ expr, 
            sc: (_Pattern | _Function | _With | _Module) :> getScopedSymbols[sc],
            {0, Infinity},
            Heads -> True
        ]
        ,
        {
            Hold[var_] /; StringContainsQ[Context[var], "`PackagePrivate`"] :> 
                With[{newsym = DBUniqueTemporary[SymbolName @ Unevaluated @ var]},
                    HoldPattern[var] :> newsym
                ],
            Hold[var_] :> (HoldPattern[var] :> var)
        }
        ,
        {1}
    ]

SetAttributes[preprocess, HoldAllComplete]
preprocess[expr_]:=
    ReplaceAll[
        HoldComplete[expr], 
        makeLongVarsReplacementRules[expr]
    ]
    

SetAttributes[executableFrameMakeBoxes, HoldAllComplete]
executableFrameMakeBoxes[frame_] :=
    Block[{
        $boxesPreprocessor = Identity, 
        $frame = Identity,
        DBStackFrameMakeBoxes = execMakeBoxes, 
        $executableFrame = True,
        $fmtLineLength = 200
        },
        Replace[
            preprocess[frame],
            HoldComplete[f_] :> First @ formatFrame[f]
        ]
    ]
    
SetAttributes[executableFrameCellPrint, HoldAllComplete]    
executableFrameCellPrint[frame_] :=
    CellPrint @ Cell[
        BoxData[executableFrameMakeBoxes[frame]], 
        "Input", 
        FontFamily -> "Monaco", FontSize -> 12, CellFrame -> True
    ]
    

SetAttributes[execMakeBoxes, HoldAllComplete]
execMakeBoxes[expr_] := MakeBoxes @ DBDisplayContainer[expr]
  
    
(* 
**  A stub to be possibly redefined via UpValues for various heads. Defines 
**  simplified / lightweight MakeBoxes - like function to be used for stack 
**  trace rendering / formatting.
*)
SetAttributes[DBStackFrameMakeBoxes, HoldAll]
DBStackFrameMakeBoxes[expr_] := 
    Replace[
        simplisticExpressionRepr[expr],
        HoldComplete[e_] :> 
            MakeBoxes @ Style[
                Framed[e, FrameStyle -> Gray, RoundingRadius -> 5], 
                ShowStringCharacters -> False
            ]
    ]


(* 
** TODO: perhaps we want to only use short names for scoped variables by default,
** and / or add an option to the stack formatter for private and package-scoped 
** non-local-variable symbols rendering 
*)
$boxesPreprocessor = ReplaceAll[{
    str_String?(StringMatchQ[__ ~~ "`" ~~ __ ~~ "$" ~~ (DigitCharacter ...)]) :>
        RuleCondition @ StringReplace[
            str,
            __ ~~ "`" ~~ name : (__ ~~ "$" ~~ DigitCharacter ...) :> name
        ],
    str_String?(StringMatchQ[__ ~~ ("`PackagePrivate`" | "`PackageScope`")  ~~__]) :> 
        RuleCondition @ StringReplace[
            str,
            (__ ~~ ("`PackagePrivate`" | "`PackageScope`") ~~ name__) :> name
        ]
}]
    

$frame = Function[
    expr,  
    Framed[
        Pane[
            Style[
                expr, 
                If[TrueQ[DatabasesUtilities`$DatabasesUtilitiesAvailable],
                    "Code",
                    "Input"
                ],
                FontFamily -> "Monaco", ShowStringCharacters -> True,
                FontSize -> 10
            ],
            ImageSize -> {UpTo[1000], Automatic}
        ]
        ,
        FrameStyle -> Lighter[Gray, 0.5]
    ]
]


$fmtLineLength = 100


$utilsHeldExprFormatter[boxPreprocessor_] := 
    Function[
        heldCode
        ,
        Block[{DatabasesUtilities`Formatting`ExpressionFormatter`PackagePrivate`$maxLineLength = $fmtLineLength},
            Replace[
                heldCode,
                Hold[se_] :> Catch[
                    DatabasesUtilities`Formatting`FullCodeFormat[
                        boxPreprocessor @ MakeBoxes[se]
                    ],
                    _,
                    Function[{value, tag},
                        DatabasesUtilities`Formatting`FullCodeFormat[
                            boxPreprocessor[
                                DatabasesUtilities`Formatting`CodeFormatterMakeBoxes[se]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    ]
    

$defaultHeldFormatter[boxPreprocessor_] := 
    Replace[
        Hold[se_] :> boxPreprocessor @ MakeBoxes[se]
    ]
    

SetAttributes[formatFrame, HoldAllComplete]

formatFrame[HoldForm[code_]] := formatFrame[code]

formatFrame[code_] := 
    $frame @ With[
        {boxes = RawBoxes @ DBMakeFormattedBoxes[
             Unevaluated[code],
             If[
                 TrueQ[DatabasesUtilities`$DatabasesUtilitiesAvailable],
                 $utilsHeldExprFormatter,
                 $defaultHeldFormatter
             ][$boxesPreprocessor],
             $defaultFormattingRules
        ]}
         ,
        If[TrueQ[$executableFrame] || !TrueQ[$DBExecutableStackFrames], 
            boxes,
            (* else *)
            Column[{
               Button[
                   Style["Print executable frame", FontFamily -> "Monaco"],
                   executableFrameCellPrint[code],
                   Method -> "Queued"
               ],
               boxes
           }]
        ]    
    ]
     
    
frameCounted = Function[code, $stackFrameCounter++; code, HoldAll]    


$stackOverallFormatter = 
    Composition[
        Framed[
            #, 
            FrameStyle -> LightGray, RoundingRadius -> 10
        ] &,
        Column
    ]
    

DBFormattedStackTrace[Failure[tag_, assoc_]] /; KeyExistsQ[assoc, "Stack"] :=
    assoc["Stack"]

DBFormattedStackTrace[] /; !TrueQ[$inStackFormatter] := DBFormattedStackTrace[stack[]]

DBFormattedStackTrace[st_List] /; !TrueQ[$inStackFormatter]:=
    With[{formattedStack =  Block[
            {
                $stackFrameCounter = 0, 
                $stackSize = Length[st], 
                $inStackFormatter = True
            }
            ,
            If[TrueQ[$DBShowStackTraceFormattingProgress],
                DBWithProgressIndicator[stackFormatterProgressIndicator[]],
                (* else *)
                Identity
            ] @ $stackOverallFormatter @ Replace[
                st,
                frame: HoldForm[f_[___]] :> frameCounted @ OpenerView[
                    { DBSymbolicHead[f], formatFrame[frame] }
                ]
                ,
                {1}
            ]
            
        ]},
        (* 
        ** Update needed to clear evaluation cache with $inStackFormatter = True 
        *)
        Update[DBFormattedStackTrace]; 
        formattedStack
    ]
    
DBFormattedStackTrace[_List]:=$Failed
        
    
stackFormatterProgressIndicator[]:=
    PrintTemporary[
        Framed[
            Row[{
                "Error encountered. Stack trace is being formatted :  ",
                ProgressIndicator[
                    Refresh[
                        Dynamic[$stackFrameCounter/$stackSize], 
                        UpdateInterval -> 1
                    ]
                ]
            }],
            FrameStyle -> Directive[LightGray],
            RoundingRadius -> 10
        ]
    ]
    
    
DBDisplayContainer /: MakeBoxes[
    c : DBDisplayContainer[inner_],
    StandardForm
] := 
    BoxForm`ArrangeSummaryBox[
        DBDisplayContainer,
        c,
        None,
        {
            BoxForm`MakeSummaryItem[
                {
                    "Type: ", 
                    Replace[
                        Unevaluated[inner], 
                        h_[___] :> SymbolName[Unevaluated @ h]
                    ]
                },
                StandardForm
            ]
        },
        {}
        ,
        StandardForm
    ]    
    
    