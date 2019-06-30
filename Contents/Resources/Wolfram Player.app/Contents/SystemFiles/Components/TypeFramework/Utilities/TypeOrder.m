
BeginPackage["TypeFramework`Utilities`TypeOrder`"]

TypeSort;
TypeOrdering;
TypeOrder;

Begin["`Private`"] 

Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`Inference`TypeInferenceState`"]
Needs["TypeFramework`Utilities`TypeToPattern`"]

(* private imports *)
Needs["TypeFramework`Utilities`TypeSubsumesQ`"]



subsumesQ := subsumesQ =
    TypeFramework`Utilities`TypeSubsumesQ`Private`subsumesQ;

defaultTypeOrderOptions = <|
    Method -> "Subsumption"
|>;

Options[TypeOrder] =
Options[TypeOrdering] = 
Options[TypeSort] = Normal[defaultTypeOrderOptions];

TypeSort[tenv_?TypeInferenceStateQ, ts_?ListQ, opts0:OptionsPattern[TypeSort]] :=
    TypeSort[tenv["typeEnvironment"], ts, opts0];
TypeSort[declEnv_?TypeEnvironmentQ, ts_?ListQ, opts0:OptionsPattern[TypeSort]] :=
    With[{
        opts = Join[defaultTypeOrderOptions, Association[opts0]]
    },
    With[{
        srt = Quiet[Check[
            Sort[
                ts,
                With[{
                    order = TypeOrder[declEnv, ##, Sequence @@ Normal[opts]]
                },
                    If[FailureQ[order],
                        TypeFailure[
			                "TypeSort",
			                "encountered an Indeterminate ordering while comparing `1` and `2`",
			                ##
			            ],
                        order
                    ]
                ]&
            ],
            $Failed
        ]]
    },
        If[FailureQ[srt],
            TypeFailure[
                "TypeSort",
                "encountered an Indeterminate ordering while sorting `1`",
                ts
            ],
            srt
        ]
    ]];
    
TypeOrdering[tenv_?TypeInferenceStateQ, ts_?ListQ, opts0:OptionsPattern[TypeOrdering]] :=
    TypeOrdering[tenv["typeEnvironment"], ts, opts0];
TypeOrdering[declEnv_?TypeEnvironmentQ, ts_?ListQ, opts0:OptionsPattern[TypeOrdering]] :=
    With[{
        opts = Join[defaultTypeOrderOptions, Association[opts0]]
    },
    With[{
        order = Quiet[Check[
            Ordering[
                ts,
                All,
                With[{
                    order = TypeOrder[declEnv, ##, Sequence @@ Normal[opts]]
                },
                    If[FailureQ[order],
                        TypeFailure[
                            "TypeOrdering",
                            "encountered an Indeterminate ordering while comparing `1` and `2`",
                            ##
                        ],
                        order
                    ]
                ]&
            ]
            ,
            $Failed
        ]] 
    },
        If[FailureQ[order],
            TypeFailure[
                "TypeOrdering",
                "encountered an Indeterminate or $Failed ordering while ordering `1`",
                ts
            ],
            order
        ]
    ]];

TypeOrder[tenv_?TypeInferenceStateQ, t1_, t2_, opts0:OptionsPattern[TypeOrder]] :=
    TypeOrder[tenv["typeEnvironment"], t1, t2, opts0];
TypeOrder[declEnv_?TypeEnvironmentQ, t1_, t2_, opts0:OptionsPattern[TypeOrder]] :=
    With[{
        opts = Join[defaultTypeOrderOptions, Association[opts0]]
    },
    With[{
        method = Lookup[opts, Method]
    },
        Switch[method,
            "Subsumption",
                compareSubsumption[declEnv, t1, t2],
            Pattern | "Pattern",
                comparePattern[declEnv, t1, t2],
            _,
                TypeFailure[
                    "TypeOrderMethod",
			        "the method `1` is not currently implemented for the TypeOrder function",
			        method
                ]
        ]
    ]];

typeSubsumesQ[declEnv_, t1_, t2_] :=
    CatchTypeFailure[
        TypeSubstitutionQ[subsumesQ[declEnv, {}, t1, t2]],
        _,
        ($Failed)&
    ];
    
compareSubsumption[declEnv_, t1_, t2_] :=
    With[{
        s1 = typeSubsumesQ[declEnv, t1, t2],
        s2 = typeSubsumesQ[declEnv, t2, t1]
    },
        Which[
            TrueQ[s1] && TrueQ[s2],
                0,
            TrueQ[s1],
                -1,
            TrueQ[s2],
                1,
            s1 === False || s2 === False,
                Indeterminate,
            FailureQ[s1] || FailureQ[s2],
                $Failed
        ]
    ];
    
comparePattern[st_, t1_, t2_] :=
    With[{
        ty1 = TypeToPattern[t1],
        ty2 = TypeToPattern[t2]
    },
        patternOrder[ty1, ty2]
    ];

patternOrder[patt1_, patt2_] := 
    Switch[Internal`ComparePatterns[patt1, patt2],
        "Specific", -1,
        "Identical" | "Equivalent", 0,
        "Incomparable",
            Switch[Internal`ComparePatterns[patt2, patt1],
                "Specific", 1,
                "Identical" | "Equivalent", 0,
                "Incomparable" | "Disjoint", Indeterminate,
                _, $Failed
            ],
        _, $Failed
    ];

End[];

EndPackage[];
