
BeginPackage["TypeFramework`ConstraintObjects`ProveConstraint`"]

CreateProveConstraint
ProveConstraintQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
ProveConstraintClass = DeclareClass[
    ProveConstraint,
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
    	"id",
        "predicate"
    },
    Predicate -> ProveConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ ProveConstraint];
]]

CreateProveConstraint[pred_] :=
    CreateObject[ProveConstraint, <|
    	"id" -> GetNextConstraintId[],
        "predicate" -> pred
    |>];


active[self_] :=
    Values[self["predicate"]["free"]]

solve[self_, st_] := (
    st["predicatesToProve"]["appendTo", self["predicate"]];
    CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]]
)




monitorForm[ self_, sub_, rest_] :=
	ConstraintSolveForm[<|
	   "name" -> "Prove",
	   "lhs" -> self["predicate"],
	   "rhs" -> (""&),
	   "unify" -> sub,
	   "rest" -> rest
	|>]


judgmentForm[self_] :=
	StyleBox[
		GridBox[{
			{RowBox[{
				Replace[self["predicate"]["test"], MemberQ[AbstractType[s_String]] :> s],
				RowBox[("v" <> ToString[#["id"]] &)/@ self["predicate"]["types"]]}]},
			{StyleBox[
				RowBox[{"(" <>
					ToString[self["properties"]["lookup", "source"]] <>
					")"}],
				FontSize -> Small]}
		}],
		FontFamily -> "Verdana"
	];

icon := icon =  Graphics[Text[
    Style["PRV\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];

toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "ProveConstraint",
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
        "Prove[",
        self["predicate"]["toString"],
        "]"
    ];


unresolve[self_] :=
    ProveConstraint[<|
        "predicate" -> self["predicate"]["unresolve"]
    |>]
    
format[self_, shortQ_:True] :=
    Row[{
        "Prove[",
        self["predicate"]["format", shortQ],
        "]"
    }];

End[]

EndPackage[]
