
BeginPackage["TypeFramework`ConstraintObjects`ConstraintBase`"]

BaseConstraintClass
BaseConstraintQ

GetNextConstraintId

RegisterConstraint

Begin["`Private`"]


Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



$NextId

If[!ValueQ[$NextId],
	$NextId = 1
]

GetNextConstraintId[] :=
	$NextId++



(**************************************************
 **************************************************
 * Base Constraint
 **************************************************
 **************************************************)


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
BaseConstraintClass = DeclareClass[
    BaseConstraint,
    <|
        "initialize" -> Function[{},
            If[Self["properties"] === Undefined,
                Self["setProperties", CreateReference[<||>]]
            ]
        ],
        "format" -> (""&),
        "active" -> (undefinedActive[Self]&),
        "free" -> (free[Self]&),
        "unresolve" -> (undefinedUnresolve[Self]&),
        "computeFree" -> (undefinedComputeFree[Self]&),
        "solve" -> (undefinedSolve[Self]&),
        "monitorForm" -> (baseMonitorForm[Self, ##]&),
        "judgmentForm" -> (""&),
        "toString" -> Function[{}, ""],
        "toBoxes" -> Function[{fmt}, ""]
    |>,
    {
    	"freeCache" -> Null,
        "properties"
    },
(*
 Set a predicate with an internal name that will not be called.
 This class is not designed to be instantiated.
*)
    Predicate -> nullBaseConstraintQ,
    Extends -> {
		ClassPropertiesTrait
	}
];
]]



(*
  Add functionality for TypeBaseClassQ
*)
If[!AssociationQ[$constraintObjects],
    $constraintObjects = <||>
]

RegisterConstraint[ name_] :=
	AssociateTo[$constraintObjects, name -> True]
	
BaseConstraintQ[ obj_] :=
	ObjectInstanceQ[obj] && KeyExistsQ[$constraintObjects, obj["_class"]]


free[self_] :=
	(
	If[self["freeCache"] === Null,
		self["setFreeCache", self["computeFree"]]];
	self["freeCache"]
	)


undefinedActive[self_] :=
    TypeFailure[
        "ConstraintActiveUndefined",
        "the \"active\" method for the class `1` has not been defined",
        SymbolName[self["_class"]]
    ];
    
undefinedUnresolve[self_] :=
    TypeFailure[
        "ConstraintUnresolveUndefined",
        "the \"unresolve\" method for the class `1` has not been defined",
        SymbolName[self["_class"]]
    ];
    
undefinedComputeFree[self_] :=
    TypeFailure[
        "ConstraintFreeUndefined",
        "the \"computeFree\" method for the class `1` has not been defined",
        SymbolName[self["_class"]]
    ];

undefinedSolve[self_] :=
    TypeFailure[
        "ConstraintSolveUndefined",
        "the \"solve\" method for the class `1` has not been defined",
        SymbolName[self["_class"]]
    ];


baseMonitorForm[self_, sol_, res_] :=
    ConstraintSolveForm[
        Append[
            <|
                "name" -> SymbolName[self["_class"]]
            |>,
            self["_state"]["get"]
        ]
    ]


(**************************************************
 **************************************************
 * ConstraintSolveForm
 **************************************************
 **************************************************)


iconConstraintSolveForm := iconConstraintSolveForm = Graphics[Text[
	Style["CSolveForm",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]

getClosed[ args_] :=
	Module[{work},
		work = Lookup[args,  "rest", {}];
		Map[ BoxForm`SummaryItem[{"", #["format"]}]&, work]
	]


formatArg[ "name", val_] :=
	val
	
formatArg[ _, val_List] :=
	Map[#["format"]&, val]
	
formatArg[ _, val_] :=
	val["format"]


getOpen[ args_] :=
	Module[ {work},
		work = KeyDrop[args, "rest"];
		KeyValueMap[ BoxForm`SummaryItem[ {#1 <> ": ",formatArg[#1, #2]}]&, work]
	]



ConstraintSolveForm /: MakeBoxes[ConstraintSolveForm[args_], fmt_] :=
    With[{open = getOpen[args], closed = getClosed[args]},
        BoxForm`ArrangeSummaryBox[
            "ConstraintSolveForm",
            ConstraintSolveForm[args],
            iconConstraintSolveForm,
            open,
            closed
            , 
            fmt
        ]
    ]


End[]

EndPackage[]

