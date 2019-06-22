
BeginPackage["TypeFramework`ConstraintObjects`AssumeConstraint`"]

CreateAssumeConstraint
AssumeConstraintQ

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
AssumeConstraintClass = DeclareClass[
    AssumeConstraint,
    <|
        "active" -> (active[Self]&),
        "solve" -> (solve[Self, ##]&),
        "format" -> (format[Self,##]&),
        "unresolve" -> (unresolve[Self]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
    	"id",
        "predicate"
    },
    Predicate -> AssumeConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ AssumeConstraint];
]]


CreateAssumeConstraint[pred_] :=
    CreateObject[AssumeConstraint, <|
    	"id" -> GetNextConstraintId[],
        "predicate" -> pred
    |>]
    


active[self_] :=
    Values[self["predicate"]["free"]]
    
    
solve[self_, st_] := (
    st["assumedPredicates"]["appendTo", self["predicate"]];
    CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]]
);
        

icon := icon =  Graphics[Text[
    Style["ASS\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];
        

toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "AssumeConstraint",
        self,
        icon,
        {
            BoxForm`SummaryItem[{Pane["predicate: ", {90, Automatic}], self["predicate"]}]
        },
        {
        }, 
        fmt
    ]


toString[self_] :=
    StringJoin[
        "Assume[",
        self["predicate"]["toString"],
        "]"
    ];

unresolve[self_] :=
    AssumeConstraint[<|
        "predicate" -> self["predicate"]["unresolve"]
    |>]


format[self_, shortQ_:True] :=
    Row[{
        "Assume[",
        self["predicate"]["format", shortQ],
        "]"
    }];

End[]

EndPackage[]
