
BeginPackage["TypeFramework`ConstraintObjects`GeneralizeConstraint`"]

CreateGeneralizeConstraint
GeneralizeConstraintQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`Inference`ConstraintSolveState`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]




RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
GeneralizeConstraintClass = DeclareClass[
    GeneralizeConstraint,
    <|
        "active" -> (active[Self]&),
        "computeFree" -> Function[{}, computeFree[Self]],
        "solve" -> (solve[Self, ##]&),
        "judgmentForm" -> (judgmentForm[Self]&),
        "monitorForm" -> (monitorForm[Self, ##]&),
        "format" -> (format[Self,##]&),
        "unresolve" -> (unresolve[Self]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
    	"id",
        "monomorphic",
        "sigma",
        "tau"
    },
    Predicate -> GeneralizeConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ GeneralizeConstraint];
]]

CreateGeneralizeConstraint[sigma_, tau_, m_] :=
    CreateObject[GeneralizeConstraint, <|
    		"id" -> GetNextConstraintId[],
            "sigma" -> sigma,
            "tau" -> tau,
            "monomorphic" -> m
        |>]

active[self_] :=
    Values[self["free"]]

computeFree[self_] :=
    Join[
        self["sigma"]["free"]
        ,
        KeySelect[
            self["tau"]["free"]
            ,
            AnyTrue[Select[self["monomorphic"], TypeVariableQ], Function[{m}, m["id"] === #]]&
        ]
    ]

solve[self_, st_] :=
    With[{
        m = self["monomorphic"],
        tau = self["tau"]
    },
        st["contextReduction"];
    With[{
        sigma = generalizeWithPredicates[st, m, tau]
    },
        st["addScheme", self["sigma"] -> sigma];
        {CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]],{}}
    ]];
    
sameQ[a_, b_] :=
    a["sameQ", b];   
    
toGenericTVar[a_?TypeVariableQ, {idx_}] :=
    CreateTypeVariable[a["name"] <> "$" <> ToString[idx]]
toGenericTVar[a_, {idx_}] :=
    CreateTypeVariable["gen$" <> ToString[idx]];
    

generalizeWithPredicates[constraintState_?ConstraintSolveStateQ, m_, ty_] :=
    generalizeWithPredicates[
        <|
            "ConstraintSolveState" -> constraintState,
            "TypeEnvironment" -> constraintState["typeEnvironment"]
        |>,
        m,
        ty
    ];
generalizeWithPredicates[st_?AssociationQ, m_, ty_] := 
    With[{
        cst = st["ConstraintSolveState"],
        tenv = st["TypeEnvironment"]
    },
    With[{
        abstractTenv = tenv["abstracttypes"],
        preds = cst["predicatesToProve"]["get"], 
        ftvTy = Values[ty["free"]],
        ftvM = Values[Join @@ Map[#["free"]&, m]]
    },
    With[{
        a = Complement[ftvTy, ftvM, SameTest -> sameQ]
    },
    With[{
        gas = MapIndexed[toGenericTVar, a]
    },
    With[{
        subst = CreateTypeSubstitution[
            AssociationThread[a -> gas],
            "TypeEnvironment" -> tenv
        ]
    },
    With[{
        q = Map[
            Function[{pred},
                With[{
                    ftvPred = Values[pred["free"]]
                },
                    If[Intersection[ftvPred, a, SameTest -> sameQ] === {},
                        Nothing,
                        pred
                    ]
                ]
            ],
            preds
        ]
    },
    With[{
        reducedPreds = abstractTenv["contextReduce", subst["apply", q]]
    },
        CreateTypeForAll[
            gas,
            CreateTypeQualified[
                reducedPreds,
                subst["apply", ty]
            ]
        ]
    ]]]]]]];
    
generalizeWithPredicates[args___] :=
    TypeFailure[
        "InvalidArgumentsGeneralizeWithPredicates",
        "invalid arguments to generalizeWithPredicates with `1`. generalizeWithPredicates expects a state, monomorphic list, and a type to genralize",
        {args}
    ]


monitorForm[ self_, sub_, rest_] :=
	ConstraintSolveForm[<|
	   "name" -> "Generalize",
	   "sigma" -> self["sigma"],
	   "tau" -> self["tau"],
	   "monomorphic" -> self["monomorphic"],
	   "unify" -> sub,
	   "rest" -> rest
	|>]
    
(* TODO *)
judgmentForm[self_] := self["toString"]

icon := icon =  Graphics[Text[
    Style["GEN\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];

toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "GeneralizeConstraint",
        self,
        icon,
        {
            BoxForm`SummaryItem[{Pane["\[Sigma]: ",             {90, Automatic}], self["sigma"]}],
            BoxForm`SummaryItem[{Pane["\[Tau]: ",               {90, Automatic}], self["tau"]}],
            BoxForm`SummaryItem[{Pane["\[ScriptCapitalM]: ",    {90, Automatic}], self["monomorphic"]}]
        },
        {
        }, 
        fmt
    ]
    
    
toString[self_] := 
    StringJoin[
        self["sigma"]["toString"],
        " = ",
        "Generalize[{",
        #["toString"]& /@ self["monomorphic"],
        "}, ",
        self["tau"]["toString"],
        "]"
    ];

unresolve[self_] :=
    GeneralizeConstraint[<|
        "sigma" -> self["sigma"]["unresolve"],
        "monomorphic" -> (#["unresolve"]& /@ self["monomorphic"]),
        "tau" -> self["tau"]["unresolve"]
    |>]

format[self_, shortQ_:True] :=
    Row[{
        self["sigma"]["format", shortQ],
        " = ",
        "Generalize[",
        #["format", shortQ]& /@ self["monomorphic"],
        ", ",
        self["tau"]["format", shortQ],
        "]"
    }];

End[]

EndPackage[]
