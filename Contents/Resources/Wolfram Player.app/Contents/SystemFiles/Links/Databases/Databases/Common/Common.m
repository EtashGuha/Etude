Package["Databases`Common`"]

PackageImport["Databases`"]


PackageExport["DBBlockExtend"]
PackageExport["DBBlockModify"]
PackageExport["DBUnevaluatedPatternCheck"]
PackageExport["DBFilterOptions"]
PackageExport["DBIterationLimiter"]
PackageExport["DBGetMemoizedValue"]
PackageExport["DBLookupFunction"]
PackageExport["DBMemberOf"]
PackageExport["DBPositionTree"]
PackageExport["DBKeyValueReverse"]
PackageExport["DBEchoUnevaluated"]
PackageExport["DBFnOrIdentity"]
PackageExport["DBEchoInputForm"]
PackageExport["DBLinkedList"]
PackageExport["DBLinkedListPattern"]
PackageExport["DBFromLinkedList"]
PackageExport["DBToLinkedList"]
PackageExport["DBDelayedPattern"]
PackageExport["DBApplyToHeldArguments"]
PackageExport["DBSymbolicHead"]
PackageExport["DBWithProgressIndicator"]
PackageExport["DBShieldExpression"]
PackageExport["DBMakeFormattedBoxes"]
PackageExport["DBUniqueTemporary"]
PackageExport["DBAssociationThrough"]
PackageExport["DBCheckAndSetCache"]



SetAttributes[DBCheckAndSetCache, HoldRest]
DBCheckAndSetCache[key_, value_, test_: Function[True]] := 
    With[
        {result = Internal`CheckCache[key]},
        If[
            FailureQ[result] || ! test[result],
            With[
                {v = value}, 
                Internal`SetCache[key, v]; 
                v
            ],
            result
        ]
    ]


(*
**  Allows to extend a given symbol with new definitions, dynamically. When the third
**  argument is set to True (the default), those definitions are guaranteed to be
**  tried before the original ones for the symbol, in most cases (exceptions are
**  pattern-free definitions, which are stored in a separate hash table and are immune
**  to manual reorderings). When the third argument is set to False, the new definitions
**  are added to the old ones, using the standard Mathematica rule-reordering mechanism.
**
**  Takes a symbol, a code (or sequence) of new definitions, and a flag which determines
**  the priority of newly added definitions. Returns a Function which would enhance
**  the original symbol dynamically with those extra definitions / rules, locally
**  for the function call.
*)

SetAttributes[DBBlockExtend, HoldAll]

DBBlockExtend[sym_, {extra__}, prependNew_: True] /; TrueQ[prependNew] :=
    DBBlockExtend[sym, {extra}, prependNew] =
    	Module[{pvals = <||>, props = {DownValues, OwnValues, SubValues, UpValues}},
    		Block[{sym},
    			extra;
    			Scan[Function[pvals[#] = #[sym]], props];
    		];
    		Function[Null,
    			Internal`InheritedBlock[{sym},
    				Scan[Function[#[sym]=Join[pvals[#], #[sym]]], props];
    				sym[##]
    			],
    			HoldAll
    		]
    	]

DBBlockExtend[sym_, {extra__}, prependNew_] :=
    Function[Null,
        Internal`InheritedBlock[{sym},
            extra;
            sym[##]
        ],
        HoldAll
    ]

 
(* 
** Helper function for wrapDefinitions. Is used to parse the r.h.s. of the rules 
** being decorated, and decorate in a special way in certain cases (e.g. when 
** the r.h.s. is wrapped in Condition explicitly or implicitly).
**
** Spec for the wrap function: should take an expression and return wrapped 
** expression, NOTE, *wrapped also in Hold*. Should normally also prevent 
** the evaluation of the passed expression.
*)
SetAttributes[wrapRHS, HoldFirst]

wrapRHS[With[dec_, Verbatim[Condition][body_, cond_]], wrap_] := 
    Module[{dummy},
        Replace[
            wrapRHS[body, wrap],
            Hold[expr_] :> Hold[dummy[dec, Condition[expr, cond]]]
        ] /. dummy -> With
    ]


wrapRHS[Verbatim[Condition][e_, cond_], wrap_] := Replace[
    wrapRHS[e, wrap],
    Hold[expr_] :> Hold[Condition[expr, cond]]
]

wrapRHS[rhs_, wrap_] := wrap[rhs]
    

(* 
** Helper function for DBBlockModify. Wraps the rules' r.h.s. with the passed 
** wrapper. The wrapper takes two arguments: a list of the arguments passed to 
** the function sym, and the actual r.h.s. that would've been normally computed.
**
** NOTE: In this function, using  var. name <args> in place of <argums> will lead 
** to var collisions with other functions in this file which use <args>, 
** when it is used. 
*)
wrapDefinitions[sym_, wrapper_] := Function[
    rules,
    Module[{ruleDelayed, dummy},
        SetAttributes[ruleDelayed, HoldAll]; (* Helper dummy symbol *)
        Replace[
            rules,
            {   (* TODO: factor out common code here *)         
                Verbatim[RuleDelayed][
                    Verbatim[HoldPattern][HoldPattern[sym][argums___]],
                    rhs_
                ] :>                     
                ReplaceAll[
                    Replace[
                        wrapRHS[rhs, Function[x, Hold[wrapper[{a}, x]], HoldAll]],
                        Hold[expr_] :> ruleDelayed[HoldPattern[dummy], expr]
                    ] /.  dummy :> sym[a:argums], (* dummy, ruleDelayed prevent var renamings by the system*)
                    ruleDelayed -> RuleDelayed
                ]
                ,
                Verbatim[RuleDelayed][Verbatim[HoldPattern][Verbatim[Condition][
                    HoldPattern[sym[argums___]],cond_]], 
                    rhs_
                ] :>                     
                ReplaceAll[
                    Replace[
                        wrapRHS[rhs, Function[x, Hold[wrapper[{a}, x]], HoldAll]],
                        Hold[expr_] :>  ruleDelayed[
                            HoldPattern[Condition[dummy, cond]], 
                            expr
                        ]
                    ] /. dummy :> sym[a:argums], (* dummy, ruleDelayed prevent var renamings by the system*)
                    ruleDelayed -> RuleDelayed
                ],
                unparseableDefinition_ :> 
                    DBRaise[
                        wrapDefinitions,
                        "unparseable_definition_for_wrapping",
                        {},
                        <|
                            "WrappedSymbol" -> sym,
                            "ProblematicDefinition" -> unparseableDefinition
                        |>
                    ]
            },
            {1}
        ]
    ]
]

DBBlockModify[sym_, wrapper_, appliedFunction_:Automatic]:=
    With[{fun = If[appliedFunction === Automatic, sym, appliedFunction]},
    	Function[Null,
    		Internal`InheritedBlock[{sym},
    			DownValues[sym] =  wrapDefinitions[sym, wrapper][DownValues[sym]]; 
    			fun[##]
    		],
    		HoldAll
    	]
    ]


SetAttributes[DBGetMemoizedValue, HoldAll]
DBGetMemoizedValue[symbol_Symbol, computedPattern_, default_: None] :=
    If[
        MatchQ[Hold[symbol] /. OwnValues[symbol], Hold[computedPattern]],
        symbol,
        (* else *)
        default
    ]



DBUnevaluatedPatternCheck[f_] := Function[arg, f[Unevaluated[arg]], HoldFirst]
DBUnevaluatedPatternCheck /: Verbatim[PatternTest][p_, pred_DBUnevaluatedPatternCheck] :=
    With[{ev = pred}, PatternTest[p, ev]]


DBFilterOptions[fun_Symbol] := Function[Sequence @@ FilterRules[{##}, Options[fun]]]
DBFilterOptions /: f_[args___, DBFilterOptions[opts___]] :=
    With[{filtered = Sequence @@ FilterRules[{opts}, Options[f]]},
        f[args, filtered]
    ]


DBIterationLimiter[limit_Integer?Positive, f_:None] :=
    Module[{ctr = 0},
        Function[arg,
            If[TrueQ[ctr++ > limit],
                DBRaise[
                    If[ f === None, DBIterationLimiter, f -> DBIterationLimiter],
                    "iteration_limit_exceeded",
                    {},
                    <| "IteratedExpressionAtLimit" -> arg |>
                ],
                (* else *)
                arg
            ]
        ]
    ]



$missingF = Function[Missing["KeyAbsent", #]]

(*
**  Similar to Lookup, but allows one to provide default function that depends on the
**  key
*)
DBLookupFunction[assoc_Association?AssociationQ, defaultFunction_: $missingF] :=
	Function[Lookup[assoc, #, defaultFunction[#]]]

(*
**  Takes a list or All, and returns a function that, when applied to an expression,
**  checks if it is a member of this list.
*)
DBMemberOf[All] := Function[True]
DBMemberOf[elems_List] :=
	DBLookupFunction[
		Association @ Thread[elems -> True],
		Function[False]
	]

(*
**  Given an expression made of lists and assocs, and a pattern, finds positions of
**  all elements matching <pattern>, and groups position list into a tree, combining
**  together positions with common paths.
*)
DBPositionTree[expr_, pattern_, restrictTo: (_List | All) : All, exceptWithin_: Except[_]] :=
	groupPositions @ DeleteCases[
		Position[
			expr,
			pattern ? (Evaluate[DBMemberOf[restrictTo]])
		],
		If[
			exceptWithin === Except[_],
			exceptWithin,
			Alternatives @@ Map[
				Append[__],
				Position[
					expr,
					exceptWithin
				]
			]
		]
	]


groupPositions[{}] := <||>
groupPositions[p: {{_Key}...}] := Flatten @ p
groupPositions[positions: {{_, ___}..}] :=
	Map[groupPositions] @ GroupBy[positions, First -> Rest]


(*
**  Reverses keys and values in an assoc. Only works for assocs with unique values.
*)
DBKeyValueReverse[assoc_Association?AssociationQ] :=
    Composition[Association, Reverse[#, {2}]&, Normal] @ assoc


DBAssociationThrough[assoc_, arg_] := Map[#[arg] &, assoc]


DBEchoUnevaluated = Function[
    label,
    Function[code, Echo[Unevaluated @ code, label, Unevaluated], HoldAll]
]


(*
** Takes a condition and a function fun, which itself is expected to return a function
** Returns a function that, when applied to expression, returns fun[expr] if cond[expr]
** yields True, and Identity otherwise
*)
DBFnOrIdentity[cond_, fun_] :=
    Function[expr, If[TrueQ[cond[expr]], fun[expr], Identity]]


DBEchoInputForm[x_, label_:""] := Echo[x, label, InputForm]
DBEchoInputForm[x_, label_, f_] := Echo[x, label, InputForm @* f]


(*
**  Linked lists are structures of the form
**
**    DBLinkedList[1, DBLinkedList[2, DBLinkedList[3, DBLinkedList[]]]]
**
**  where linkedListContainer is an inert symbol. These structures are the closest
**  WL has to immutable linked lists, and have constant-time addition and copy
**  complexity.
**
**  NOTE:  linked lists as implemented here, may not be reentrant, so if you need
**  to use them in a nested way, it may be better to use different container heads
**  for different nested linked lists.
*)
SetAttributes[DBLinkedList, HoldAllComplete]

DBLinkedListPattern[head_: DBLinkedList] := _head

DBFromLinkedList[ll_, head_: DBLinkedList] /; MatchQ[ll, DBLinkedListPattern[head]] :=
	List @@ Flatten[ll, Infinity, head]

DBToLinkedList[l_List, head_: DBLinkedList] :=
    Fold[head[#2, #1]&, head[], Reverse @ l]

DBDelayedPattern[patt_] := _?(Function[arg, MatchQ[Unevaluated @ arg, patt], HoldAll])


(*
**  Applies a function to arguments, some of which are wrapped in Hold, stripping
**  such Hold-s in the process. Example:
**
**     DBApplyToHeldArguments[
**         Function[Null, ToString[Unevaluated[#]], HoldAll]
**     ] @ Hold[2 + 2] -> "2+2"
*)
DBApplyToHeldArguments[fun_] :=
	Function[
        Null,
        Replace[
            Replace[Hold[##], Hold[x_] :>x, {1}],
            Hold[args___] :> fun[args]
        ]
    ]
    
    
SetAttributes[DBSymbolicHead, HoldAllComplete]
DBSymbolicHead[expr_] := Scan[Return, Unevaluated[expr], {-1}, Heads -> True]


DBUniqueTemporary[sym_] := Unique[sym, Temporary]

SetAttributes[DBWithProgressIndicator, HoldAll]    
DBWithProgressIndicator[indicatorCode_]:=
    Function[
        code
        ,
        With[{indicator = indicatorCode},
            With[{result = code},
                NotebookDelete[indicator];
                result
            ]
        ]  
        ,
        HoldAll
    ]
    
    
SetAttributes[DBShieldExpression, HoldAll]
DBShieldExpression[expr_, partTransformationRules : {___RuleDelayed}] :=
	Module[{result, backRules, i = 0, shieldingRules},
        shieldingRules = Replace[
            partTransformationRules,
            (patt_ :> rhs_) :>  patt :> RuleCondition[
				With[{shieldToken = StringJoin["Shielded_", ToString[++i]]},
					Sow[shieldToken :> rhs]; shieldToken
				]
			],
            {1}
        ];
		{result, backRules} = Reap[Hold[expr] /. shieldingRules];
		If[backRules =!= {}, backRules = First @ backRules];
		<|"HeldExpression" -> result, "BackRules" -> backRules|>
	]   
    
    
DBMakeFormattedBoxes[
    expr_,
    heldExpressionBoxFormatter_,
    customFormattingRules:{___RuleDelayed}
] :=
    With[{shielded = DBShieldExpression[expr, customFormattingRules]},
        ReplaceAll[
            heldExpressionBoxFormatter @ shielded["HeldExpression"],
            Replace[
                shielded["BackRules"],
                (tok_ :> res_) :> (ToString[tok, InputForm] -> res),
                {1}
            ]
        ]
    ]
            
