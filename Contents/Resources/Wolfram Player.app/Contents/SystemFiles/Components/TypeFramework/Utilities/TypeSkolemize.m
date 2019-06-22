
BeginPackage["TypeFramework`Utilities`TypeSkolemize`"]

TypeSkolemize;

Begin["`Private`"] 

Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`TypeObjects`Kind`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeSequence`"]



defaultTypeSkolemizeOptions = <|
    "Deep" -> True,
    "TypeEnvironment" -> Undefined
|>;

Options[TypeSkolemize] = Normal[defaultTypeSkolemizeOptions];


makeState[iopts:OptionsPattern[TypeSkolemize]] :=
    With[{
        opts = Association[iopts]
    },
        <|
            "TypeEnvironment" -> Lookup[opts, "TypeEnvironment", Undefined]
        |>
    ]
    
(*
    Skolemization replaces each quantified type variable with a type constant
    that unifies only with itself.
 *)


TypeSkolemize[t_?TypeForAllQ, opts:OptionsPattern[TypeSkolemize]] :=
    skolemize[makeState[opts], t, Join[defaultTypeSkolemizeOptions, Association[opts]]];

skolemize[st_, ts_?ListQ, opts_?AssociationQ] :=
    skolemize[st, #, opts]& /@ ts;

skolemize[st_, t_?TypeQualifiedQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeQualified[
            t["predicates"],
            skolemize[st, t["type"], opts]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    
skolemize[st_, t_?TypePredicateQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeQualified[
            skolemize[st, t["types"], opts],
            t["test"]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    

skolemize[st_, t_?TypeApplicationQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeApplication[
            skolemize[st, t["type"], opts],
            skolemize[st, t["arguments"], opts]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];

skolemize[st_, t_?TypeForAllQ, opts_?AssociationQ] :=
    With[{
        args = t["variables"]
    },
    With[{
        skolArgs = makeSkolemVariable[st, #]& /@ args
    },
    With[{
        sub = CreateTypeSubstitution[
            AssociationThread[args -> skolArgs],
            "TypeEnvironment" -> st["TypeEnvironment"]
        ]
    },
    With[{
        body0 = sub["apply", t["type"]]
    },
    With[{
        body = If[opts["Deep"],
            skolemize[st, body0, opts],
            body0
        ]
    },
    With[{
        res = CreateTypeForAll[
            skolArgs,
            body
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ]]]]]];

skolemize[st_, t_?TypeArrowQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeArrow[
            skolemize[st, t["arguments"], opts],
            skolemize[st, t["result"], opts]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
skolemize[st_, t_?TypeSequenceQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeSequence[
            skolemize[st, t["type"], opts],
            If[ t["binding"] === None, 
            		None,
            		skolemize[st, t["binding"], opts]
            ],
            {
                t["min"],
                t["max"]
            }
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    
skolemize[st_, t_?TypeLiteralQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeLiteral[
            t["value"],
            skolemize[st, t["type"], opts]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    
skolemize[st_, t_?TypeVariableQ, opts_?AssociationQ] :=
    t;
skolemize[st_, t_?TypeConstructorQ, opts_?AssociationQ] :=
    t;
 
skolemize[st_, t_, opts_?AssociationQ] :=
    TypeFailure[
        "Skolemize",
        "invalid usage of the skolemize function. Skolemize cannot be performed on `1`",
        t
    ]
       
makeSkolemVariable[st_, t_?TypeVariableQ] :=
    With[{
        new = t["clone"]
    },
        new["setVariablename", "_" <> new["variablename"]];
        new["setKind", CreateSkolemKind[new["kind"]]];
        new
    ];
End[];

EndPackage[];