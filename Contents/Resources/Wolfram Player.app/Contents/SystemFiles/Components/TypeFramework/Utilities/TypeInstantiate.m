
BeginPackage["TypeFramework`Utilities`TypeInstantiate`"]

TypeInstantiate;

Begin["`Private`"] 

Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeSequence`"]


defaultTypeInstantiateOptions = <|
    "TypeEnvironment" -> Undefined,
    "Deep" -> False
|>;

Options[TypeInstantiate] = Normal[defaultTypeInstantiateOptions];

makeState[iopts:OptionsPattern[TypeInstantiate]] :=
    With[{
        opts = Association[iopts]
    },
	    <|
	        "TypeEnvironment" -> Lookup[opts, "TypeEnvironment", Undefined]
	    |>
    ]

TypeInstantiate[t_?TypeForAllQ, opts:OptionsPattern[TypeInstantiate]] :=
    instantiate[makeState[opts], t, Join[defaultTypeInstantiateOptions, Association[opts]]];


instantiate[st_, t_?TypeForAllQ, opts_?AssociationQ] :=
    With[{
        fresh = #["clone"]& /@ t["variables"]
    },
        Scan[
            #["setProperty", "MetaVariable" -> True]&,
            fresh
        ];
    With[{
        subsExt = CreateTypeSubstitution[
            AssociationThread[t["variables"] -> fresh],
            "TypeEnvironment" -> st["TypeEnvironment"]
        ]
    },
    With[{
        body = subsExt["apply", t["type"]]
    },
        If[opts["Deep"],
            instantiate0[st, body, opts],
            body
        ]
    ]]];
    
instantiate0[st_, ts_?ListQ, opts_?AssociationQ] :=
    instantiate0[st, #, opts]& /@ ts;

instantiate0[st_, t_?TypeVariableQ, opts_?AssociationQ] :=
    t;
    
instantiate0[st_, t_?TypeConstructorQ, opts_?AssociationQ] :=
    t;
    
instantiate0[st_, t_?TypeApplicationQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeApplication[
            instantiate0[st, t["type"], opts],
            instantiate0[st, t["arguments"], opts]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    
instantiate0[st_, t_?TypeArrowQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeArrow[
            instantiate0[st, t["arguments"], opts],
            instantiate0[st, t["result"], opts]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    
instantiate0[st_, t_?TypeSequenceQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeSequence[
            instantiate0[st, t["type"], opts],
            If[ t["binding"] === None, 
            		None, 
            		instantiate0[st, t["binding"], opts]],
            {
                t["min"],
                t["max"]
            }
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    
instantiate0[st_, t_?TypeLiteralQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypeLiteral[
            t["value"],
            instantiate0[st, t["type"], opts]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    
instantiate0[st_, t_?TypeQualifiedQ, opts_?AssociationQ] :=
    With[{
        preds = instantiate0[st, t["predicates"], opts],
        body = instantiate0[st, t["type"], opts]
    },
    With[{
        res = If[TypeQualifiedQ[body],
            CreateTypeQualified[
                Join[preds, body["predicates"]],
                body["type"]
            ],
            CreateTypeQualified[
                preds,
                body
            ]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ]];
    
instantiate0[st_, t_?TypePredicateQ, opts_?AssociationQ] :=
    With[{
        res = CreateTypePredicate[
            instantiate0[st, t["types"], opts],
            t["test"]
        ]
    },
        res["setProperties", t["properties"]["clone"]];
        res
    ];
    
instantiate0[st_, t_?TypeForAllQ, opts_?AssociationQ] :=
    instantiate[st, t, opts];

End[];

EndPackage[];
