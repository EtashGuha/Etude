
BeginPackage["TypeFramework`TypeObjects`AbstractType`"]


CreateAbstractType


Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`Inference`Unify`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
DeclareClass[
	AbstractType,
	<|
		"sameQ" -> Function[{other},
			AbstractTypeQ[other] &&
			Self["typename"] === other["typename"]
		],
        "unresolve" -> (unresolve[Self]&),
        "canonicalize" -> (canonicalize[Self, ##]&),
		"addInstance" -> (addInstance[Self, ##]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"id",
		"kind",
	    "typename" -> "",
	    "arguments" -> {},
	    "members" -> {},
	    "supers" -> {},
	    "instances" -> {},
	    "default" -> Undefined
	},
	Predicate -> AbstractTypeQ
];
]]


Options[CreateAbstractType] = {"Deriving" -> {}, "Default" -> Undefined}


CreateAbstractType[ name_, args_, members_, kind_, opts0:OptionsPattern[]] :=
	CreateAbstractType[AbstractType[ name, args, members, kind, opts0]]

CreateAbstractType[ AbstractType[ name_, args_, members_, kind_, opts0:OptionsPattern[]]] :=
	Module[{supers, default, opts = Association[opts0]},
		supers = Flatten[{Lookup[opts, "Deriving", {}]}];
        default = Lookup[opts, "Default", Undefined];
		CreateObject[
			AbstractType,
			<|
				"id" -> GetNextTypeId[],
				"kind" -> kind,
				"typename" -> name,
                "arguments" -> args,
                "supers" -> CreateReference[supers],
                "instances" -> CreateReference[{}],
                "members" -> members,
                "default" -> default
			|>
		]
	]




genAbstractTypeArg[a_, {idx_}] :=
    Switch[a,
        _?StringQ,
            CreateTypeVariable[a],
        _?TypeVariableQ,
            CreateTypeVariable[a["name"]],
        _,
            TypeFailure[
                "CreateAbstractType",
                "one or more of the args `1` used to create a type class is invalid",
                a
            ]
    ];

(**************************************************)
 
unresolve[ self_] :=
	AbstractType[ self["typename"]] 

(**************************************************)
   
canonicalize[self_] :=
    canonicalize[self, CreateReference[1]];
canonicalize[self_, idx_] :=
    TypeFailure[
        "CanonicalizeAbstractType",
        "canonicalization of abstract type `1` is not currently implemented",
        self
    ]
    
    
(**************************************************)

addInstance[self_, inst_?TypeQualifiedQ] :=
  With[{
    instsRef = self["instances"]
  },
  With[{
    insts = instsRef["toList"]
  },
    If[instanceOverlapsQ[insts, inst],
        TypeFailure[
            "AbstractTypeAddInstance",
            "because of overlap in `1` the type instances `2` cannot be added to the class `3`",
            Select[insts, instanceOverlapsQ[#, inst]&],
            inst,
            self
        ],
        instsRef["appendTo", inst]
    ]
  ]];
addInstance[self_, other___] :=
  TypeFailure[
    "AbstractTypeAddInstance",
    "invalid argument `1` was used as instance for `2`",
    {other},
    self
  ];

(* TODO: possibly check for super classes when checking overlaps *)
instanceOverlapsQ[insts_, qual_?TypeQualifiedQ] :=
    instanceOverlapsQ[insts, qual["type"]]
instanceOverlapsQ[insts_, pred_?TypePredicateQ] :=
    If[insts === {},
        False,
        AnyTrue[
            Map[#["type"]&, insts],
            predUnifiableQ[#, pred]&
        ]
    ]
instanceOverlapsQ[pred1_?TypePredicateQ, pred2_?TypePredicateQ] :=
    predUnifiableQ[pred1, pred2]

predUnifiableQ[pred1_?TypePredicateQ, pred2_?TypePredicateQ] :=
    (pred1["test"] === pred2["test"]) && 
    TypeUnifiableManyQ[pred1["types"], pred2["types"]]

(**************************************************)

icon := icon = Graphics[Text[
	Style["ABS\nTYP",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
        
toBoxes[t_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"AbstractType",
		t,
  		icon,
		{
            BoxForm`MakeSummaryItem[{Pane["id: ",                   {90, Automatic}], t["id"]}, fmt],
            BoxForm`MakeSummaryItem[{Pane["name: ",                 {90, Automatic}], t["typename"]}, fmt],
            BoxForm`MakeSummaryItem[{Pane["arguments: ",            {90, Automatic}], t["arguments"]}, fmt],
            BoxForm`MakeSummaryItem[{Pane["supers: ",               {90, Automatic}], t["supers"]["toList"]}, fmt],
            BoxForm`MakeSummaryItem[{Pane["instances: ",            {90, Automatic}], Column[t["instances"]["toList"]]}, fmt],
            If[t["default"] =!= Undefined,
                BoxForm`MakeSummaryItem[{Pane["default: ",          {90, Automatic}], t["default"]}, fmt],
                Nothing
            ]
        },
        {
            BoxForm`MakeSummaryItem[{Pane["members: ",  {90, Automatic}], t["members"]}, fmt]
        },
  		fmt
  	]


toString[typ_] := "AbstractType[" <> typ["typename"] <> "]"

End[]

EndPackage[]
