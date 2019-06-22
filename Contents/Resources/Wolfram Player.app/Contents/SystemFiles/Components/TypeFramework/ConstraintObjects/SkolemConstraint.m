
BeginPackage["TypeFramework`ConstraintObjects`SkolemConstraint`"]

CreateSkolemConstraint
SkolemConstraintQ

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`Utilities`TypeSkolemize`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
SkolemConstraintClass = DeclareClass[
    SkolemConstraint,
    <|
        "active" -> (active[Self]&),
        "solve" -> (solve[Self, ##]&),
        "format" -> (format[Self,##]&),
        "unresolve" -> (unresolve[Self]&),
        "skolemize" -> (skolemize[Self, ##]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
    	"id",
        "tau",
        "rho",
        "monomorphic"
    },
    Predicate -> SkolemConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ SkolemConstraint];
]]


CreateSkolemConstraint[tau_, rho_, m_] :=
    CreateObject[SkolemConstraint, <|
    		"id" -> GetNextConstraintId[],
            "tau" -> tau,
            "rho" -> rho,
            "monomorphic" -> m
    |>]


skolemM[] :=
    0;

active[self_] :=
    0;


solve[self_, st_] :=
    0;

(*
    Skolemization replaces each quantified type variable with a type constant
    that unifies only with itself.
 *)
skolemize[self_, st_:None] :=
    TypeSkolemize[st, self["rho"], "TypeEnvironment" -> st];

unresolve[self_] :=
    SkolemConstraint[<|
        "tau" -> self["tau"]["unresolve"],
        "rho" -> self["rho"]["unresolve"],
        "monomorphic" -> (#["unresolve"]& /@ self["monomorphic"])
    |>]
    

icon := icon =  Graphics[Text[
    Style["SKOL\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];

toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "SkolemConstraint",
        self,
        icon,
        {
            BoxForm`SummaryItem[{Pane["\[Tau]: ",               {90, Automatic}], self["tau"]}],
            BoxForm`SummaryItem[{Pane["\[Rho]: ",               {90, Automatic}], self["rho"]}],
            BoxForm`SummaryItem[{Pane["\[ScriptCapitalM]: ",    {90, Automatic}], self["monomorphic"]}]
        },
        {
        },
        fmt
    ]


toString[self_] :=
    StringJoin[
        self["tau"]["toString"],
        " = ",
        "Skolem[{",
        #["toString"]& /@ self["monomorphic"],
        "}, ",
        self["rho"]["toString"],
        "]"
    ];


format[self_, shortQ_:True] :=
    Row[{
        self["tau"]["format", shortQ],
        " = ",
        "Skolem[{",
        #["format", shortQ]& /@ self["monomorphic"],
        "}, ",
        self["rho"]["format", shortQ],
        "]"
    }];

End[]

EndPackage[]
