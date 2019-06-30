
BeginPackage["TypeFramework`ConstraintObjects`SuccessConstraint`"]

CreateSuccessConstraint
SuccessConstraintQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



(**
 * A success constraint is one that always succeeds, is always active, and
 * can be solved to generate an empty subsitution
 *) 
RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
SuccessConstraintClass = DeclareClass[
    SuccessConstraint,
    <|
        "active" -> (active[Self]&),
        "solve" -> (solve[Self, ##]&),
        "judgmentForm" -> (judgmentForm[Self]&),
        "monitorForm" -> (monitorForm[Self, ##]&),
        "format" -> (format[Self,##]&),
        "unresolve" -> (unresolve[Self]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
    	"id"
    },
    Predicate -> SuccessConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ SuccessConstraint];
]]

CreateSuccessConstraint[] :=
    CreateObject[SuccessConstraint, <|"id" -> GetNextConstraintId[]|>];


active[self_] :=
    True;

solve[self_, st_] := (
    CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]]
)

monitorForm[ self_, sub_, rest_] :=
    ConstraintSolveForm[<|
       "name" -> "Success",
       "lhs" -> "True",
       "rhs" -> (""&),
       "unify" -> sub,
       "rest" -> rest
    |>]


judgmentForm[self_] :=
    StyleBox[
        "True",
        FontFamily -> "Verdana"
    ];

icon := icon =  Graphics[Text[
    Style["SUCC\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];

toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "SuccessConstraint",
        self,
        icon,
        {
        },
        {
        }, 
        fmt
    ]


toString[self_] :=
    "Success[]";

unresolve[self_] :=
    SuccessConstraint[]

format[self_, shortQ_:True] :=
    Row[{
        "Success[]"
    }];

End[]

EndPackage[]
