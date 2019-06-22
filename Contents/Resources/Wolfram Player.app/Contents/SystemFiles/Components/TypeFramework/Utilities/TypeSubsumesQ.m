
BeginPackage["TypeFramework`Utilities`TypeSubsumesQ`"]

TypeSubsumesQ;

Begin["`Private`"] 

Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`Inference`TypeInferenceState`"]
Needs["TypeFramework`Inference`Unify`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`Utilities`TypeSkolemize`"]

Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]

ClearAll[TypeSubsumesQ];
ClearAll[subsumesQ];

(* debugPrint = Print *)

TypeSubsumesQ[tenv_?TypeInferenceStateQ, t1_, t2_] :=
    TypeSubsumesQ[tenv["typeEnvironment"], t1, t2];
TypeSubsumesQ[declEnv_?TypeEnvironmentQ, t1_, t2_] :=
    TypeSubstitutionQ[CatchTypeFailure[
        subsumesQ[declEnv, {}, t1, t2],
        _,
        (debugPrint["TypeSubsumesQ:: ", ##]; $Failed)&
    ]];

mkFailure[v1_, v2_] := TypeFailure[
	"TypeSubsumesQ",
	"`v1` does not subsume `v2`",
	v1,
	v2
];

(* The ordering of these patterns matters to ensure top subsumes itself *)
subsumesQ[env_, assumedPredicates_, v1_?TypeBottomQ, v2_] := CreateTypeSubstitution["TypeEnvironment" -> env];
subsumesQ[env_, assumedPredicates_, v1_, v2_?TypeTopQ] := CreateTypeSubstitution["TypeEnvironment" -> env];
subsumesQ[env_, assumedPredicates_, v1_?TypeTopQ, v2_] := mkFailure[v1, v2];
subsumesQ[env_, assumedPredicates_, v1_, v2_?TypeBottomQ] := mkFailure[v1, v2];


subsumesQ[env_, assumedPredicates_, t1_, t2_?TypeForAllQ] :=
    With[{
        pr = TypeSkolemize[t2]
    },
    With[{
        qual = pr["type"]
    },
        Intersection[
            pr["variables"],
            Values[t1["free"]],
            SameTest -> sameQ
        ] === {} &&
        subsumesQ[env, Join[assumedPredicates, qual["predicates"]], t1, qual["type"]]
    ]];
subsumesQ[env_, assumedPredicates_, t1_?TypeForAllQ, t2_?TypeObjectQ] :=
    With[{
        inst = t1["instantiate", "Deep" -> False] (* A qualified type *)
    },
        If[entailsQ[env, assumedPredicates, inst["predicates"]],
            subsumesQ[env, assumedPredicates, inst["type"], t2],
            TypeFailure[
		        "TypeSubsumesQ",
		        "the predicates `1` are not enatailed by the assumed predicates `2`",
		        inst["predicates"],
		        assumedPredicates
		    ]
        ]
    ];
    
 subsumesQ[env_, assumedPredicates_, t1_?TypeQualifiedQ, t2_?TypeObjectQ] :=
 	subsumesQ[ env, assumedPredicates, CreateTypeForAll[{}, t1], t2]
    
subsumesQ[env_, assumedPredicates_, t1_?TypeObjectQ, t2_?TypeQualifiedQ] :=
 	subsumesQ[ env, assumedPredicates, t1, CreateTypeForAll[{}, t2]]
    
    
subsumesQ[env_, assumedPredicates_, t1_?TypeArrowQ, t2_?TypeArrowQ] :=
    subsumesQ[
        env,
        assumedPredicates,
        Append[t1["arguments"], t1["result"]],
        Append[t2["arguments"], t2["result"]]
    ]; 
subsumesQ[env_, assumedPredicates_, t1_?TypeApplicationQ, t2_?TypeApplicationQ] :=
    With[{
        subst = subsumesQ[env, assumedPredicates, t1["type"], t2["type"]]
    },
	    subsumesQ[
	        env,
	        assumedPredicates,
	        subst["apply", t1["arguments"]],
	        subst["apply", t2["arguments"]]
	    ]
    ]; 
    
subsumesQ[env_, assumedPredicates_, t1_?TypeLiteralQ, t2_] :=
    subsumesQ[env, assumedPredicates, t1["type"], t2];
    
subsumesQ[env_, assumedPredicates_, t1_, t2_?TypeLiteralQ] :=
    subsumesQ[env, assumedPredicates, t2, t1["type"]];
    
subsumesQ[env_, assumedPredicates_, t1_?TypeObjectQ, t2_?TypeObjectQ] := (
    TypeUnify[t1, t2, "TypeEnvironment" -> env]
);

  
subsumesQ[env_, assumedPredicates_, {}, {}] :=
    CreateTypeSubstitution["TypeEnvironment" -> env];
subsumesQ[env_, assumedPredicates_, {}, e:{__}] :=
    TypeFailure[
        "TypeSubsumesQ",
        "the length of the arguments `1` did not match",
        e
    ]
subsumesQ[env_, assumedPredicates_, e:{__}, {}] :=
    TypeFailure[
        "TypeSubsumesQ",
        "the length of the arguments `1` did not match",
        e
    ]
subsumesQ[env_, assumedPredicates_, {t1_, t1s___}, {t2_, t2s___}] :=
    With[{
        subst = subsumesQ[env, assumedPredicates, t1, t2]
    },
        subsumesQ[
            env,
            assumedPredicates,
            subst["apply", {t1s}],
            subst["apply", {t2s}]
        ]
    ];
    
    
sameQ[t1_, t2_] := TrueQ[t1["sameQ", t2]];

entailsQ[env_, assumedPreds_, preds_?ListQ] :=
    AllTrue[preds, entailsQ[env, assumedPreds, #]&];
entailsQ[env_, assumedPreds_, pred_] :=
    With[{
        absEnv = env["abstracttypes"]
    },
        absEnv["predicatesEntailQ", assumedPreds, pred]
    ];

End[];

EndPackage[];
