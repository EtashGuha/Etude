
BeginPackage["TypeFramework`ConstraintObjects`EqualConstraint`"]

CreateEqualConstraint
EqualConstraintQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Inference`Unify`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`Utilities`Error`"]


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
EqualConstraintClass = DeclareClass[
	EqualConstraint,
	<|
        "active" -> (active[Self]&),
        "solve" -> (solve[Self, ##]&),
        "computeFree" -> (computeFree[Self]&),
        "monitorForm" -> (monitorForm[Self, ##]&),
        "judgmentForm" -> (judgmentForm[Self]&),
        "unresolve" -> (unresolve[Self]&),
		"format" -> (format[Self,##]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"id",
		"lhs",
		"rhs"
	},
	Predicate -> EqualConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ EqualConstraint];
]]

CreateEqualConstraint[lhs_, rhs_] :=
	CreateObject[EqualConstraint, <|
		"id" -> GetNextConstraintId[],
        "lhs" -> lhs,
        "rhs" -> rhs
	|>]


active[ self_] :=
	Values[self["free"]]

computeFree[self_] :=
    Join[
        self["lhs"]["free"],
        self["rhs"]["free"]
    ]
 
(*
  solve the constraint.  If there is a type failure then we look at the fixedError property
  (default 0).
  If this is greater than 1 we error out.  Otherwise we increment it.  This means that 
  we make two attempts to solve EqualConstraints.  In ConstraintData we put the constraint 
  at the back of the equalConstraints the first time and otherConstraints the second.
*) 
solve[self_, st_] :=
    With[{
        lhs = self["lhs"],
        rhs = self["rhs"]
    },
    Module[{subs},
    	CatchTypeFailure[
            subs = TypeUnify[<| "ConstraintSolveState" -> st |>, lhs, rhs, "TypeEnvironment" -> st["typeEnvironment"]];
            {subs, {}}
            ,
            _,
            Module[{err = self["getProperty", "fixedError", 0]},
				If[ err > 1,
					ThrowTypeFailure[#1]
				];
	            self["setProperty", "fixedError" -> err+1];
	            subs = CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]];
	            {subs, {self}}
	            ]&
        ]
    ]
    ]
    
judgmentForm[self_] :=
	StyleBox[
		GridBox[{
			{RowBox[{"v"<>ToString[self["lhs"]["id"]] <>
				"\[Congruent]" <>
				"v"<>ToString[self["rhs"]["id"]]}]},
			{StyleBox[
				RowBox[{"(" <>
					ToString[self["properties"]["lookup", "source"]] <>
					")"}],
				FontSize -> Small]}
		}],
		FontFamily -> "Verdana"
	];

monitorForm[ self_, sub_, rest_] :=
	ConstraintSolveForm[<|
	   "name" -> "Equal",
	   "lhs" -> self["lhs"],
	   "rhs" -> self["rhs"],
	   "unify" -> sub,
	   "rest" -> rest
	|>]

icon := icon =  Graphics[Text[
    Style["EQ\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];
      
toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "EqualConstraint",
        self,
        icon,
        {
            BoxForm`SummaryItem[{Pane["lhs: ", {90, Automatic}], self["lhs"]}],
            BoxForm`SummaryItem[{Pane["rhs: ", {90, Automatic}], self["rhs"]}]
        },
        {
            
        }, 
        fmt
    ]


toString[self_] := 
    self["lhs"]["toString"] <>
    " \[Congruent] " <>
    self["rhs"]["toString"];

unresolve[self_] :=
    EqualConstraint[
        self["lhs"]["unresolve"],
        self["rhs"]["unresolve"]
    ]

format[self_, shortQ_:True] :=
    Row[{
        self["lhs"]["format", shortQ],
        "\[Congruent]",
        self["rhs"]["format", shortQ]
    }];

End[]

EndPackage[]
