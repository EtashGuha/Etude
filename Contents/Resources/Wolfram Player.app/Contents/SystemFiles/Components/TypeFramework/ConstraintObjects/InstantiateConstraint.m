
BeginPackage["TypeFramework`ConstraintObjects`InstantiateConstraint`"]

CreateInstantiateConstraint
InstantiateConstraintQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
InstantiateConstraintClass = DeclareClass[
    InstantiateConstraint,
    <|
        "active" -> (active[Self]&),
        "solve" -> (solve[Self, ##]&),
        "computeFree" -> Function[{}, computeFree[Self]],
        "judgmentForm" -> (judgmentForm[Self]&),
        "monitorForm" -> (monitorForm[Self, ##]&),
        "format" -> (format[Self,##]&),
        "unresolve" -> (unresolve[Self]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
    	"id",
        "tau",
        "rho"
    },
    Predicate -> InstantiateConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ InstantiateConstraint];
]]

CreateInstantiateConstraint[tau_, rho_] :=
    CreateObject[InstantiateConstraint, <|
    		"id" -> GetNextConstraintId[],
            "tau" -> tau,
            "rho" -> rho
    |>]


active[self_] :=
    Values[self["free"]]



computeFree[self_] :=
    Join[
        self["tau"]["free"],
        self["rho"]["free"]
    ]
    
solve[self_, st_] :=
    With[{
        scheme = st["lookupScheme", self["rho"]]
    },
    With[{
        qual = instantiateM[st, scheme]
    },
    With[{
        ps = qual["predicates"],
        tau1 = qual["type"]
    },
        Do[
            With[{
                constraint = st["appendProveConstraint", pred]
            },
                constraint["setProperty", "source" -> self]
            ],
            {pred, ps}
        ];
        With[{
            constraint = st["prependEqualConstraint", self["tau"], tau1["type"]]
        },
            constraint["setProperty", "source" -> self];
        {CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]], {constraint}}
    	]
    ]]];

instantiateM[st_, scheme_?TypeForAllQ] :=
    scheme["instantiate", "TypeEnvironment" -> st["typeEnvironment"]];
    
instantiateM[st_, args__] :=
    TypeFailure[
        "InvalidArgumentsInstantiateM",
        "invalid arguments to instantiateM with `1`. instantiateM expects a type scheme",
        {args}
    ]

judgmentForm[self_] := self["toString"]
 
 
 
 monitorForm[ self_, sub_, rest_] :=
	ConstraintSolveForm[<|
	   "name" -> "Instantiate",
	   "tau" -> self["tau"],
	   "rho" -> self["rho"],
	   "unify" -> sub,
	   "rest" -> {}
	|>]

icon := icon =  Graphics[Text[
    Style["INST\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];
      
toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "InstantiateConstraint",
        self,
        icon,
        {
            BoxForm`SummaryItem[{Pane["\[Tau]: ",       {90, Automatic}], self["tau"]}],
            BoxForm`SummaryItem[{Pane["\[Rho]: ",       {90, Automatic}], self["rho"]}]
        },
        {
        }, 
        fmt
    ]

toString[self_] := 
    StringJoin[
        self["tau"]["toString"],
        " = ",
        "Instantiate[",
            self["rho"]["toString"],
        "]"
    ];

unresolve[self_] :=
    InstantiateConstraint[<|
        "tau" -> self["tau"]["unresolve"],
        "rho" -> self["rho"]["unresolve"]
    |>]

format[self_, shortQ_:True] :=
    Row[{
        self["tau"]["format", shortQ],
        " = ",
        "Instantiate[",
	        self["rho"]["format", shortQ],
        "]"
    }];
    
End[]

EndPackage[]
