
BeginPackage["TypeFramework`ConstraintObjects`FailureConstraint`"]

CreateFailureConstraint
FailureConstraintQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`Utilities`Error`"]



(**
 * A failure constraint is one that always failus, is always active, and
 * can be solved to throw an error. The error is thrown only when one tries
 * to solve the constraint (not on creation)
 *) 
RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
FailureConstraintClass = DeclareClass[
    FailureConstraint,
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
    Predicate -> FailureConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ FailureConstraint];
]]

CreateFailureConstraint[] :=
    CreateObject[FailureConstraint, <| "id" -> GetNextConstraintId[]|>];


active[self_] :=
    True;

solve[self_, st_] := (
    TypeFailure[
        "FailureConstraint",
        "encountered a failure constraint"
    ]
)

monitorForm[ self_, sub_, rest_] :=
    ConstraintSolveForm[<|
       "name" -> "Failure",
       "lhs" -> "True",
       "rhs" -> (""&),
       "unify" -> sub,
       "rest" -> rest
    |>]


judgmentForm[self_] :=
    StyleBox[
        "$Failure",
        FontFamily -> "Verdana"
    ];

icon := icon =  Graphics[Text[
    Style["FAIL\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];

toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "FailureConstraint",
        self,
        icon,
        {
        },
        {
        }, 
        fmt
    ]


toString[self_] :=
    "Failure[]";

unresolve[self_] :=
    FailureConstraint[]

format[self_, shortQ_:True] :=
    Row[{
        "Failure[]"
    }];

End[]

EndPackage[]
