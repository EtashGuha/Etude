
BeginPackage["TypeFramework`Inference`Unify`"]

TypeUnify;
TypeUnifyCatch
TypeUnifyMany;
TypeUnifiableQ;
TypeUnifiableManyQ;

FindGeneralUnifierCatch
FindGeneralUnifier


Begin["`Private`"];

Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeSequence`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeProjection`"]
Needs["TypeFramework`TypeObjects`TypeEvaluate`"]
Needs["TypeFramework`Inference`Substitution`"]

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

defaultTypeUnifyOptions = <|
    "TypeEnvironment" -> Undefined
|>;

Options[TypeUnify] = Normal[defaultTypeUnifyOptions];
Options[TypeUnifiableQ] = Normal[defaultTypeUnifyOptions];
Options[TypeUnifyMany] = Normal[defaultTypeUnifyOptions];
Options[TypeUnifiableManyQ] = Normal[defaultTypeUnifyOptions];

makeState[iopts:OptionsPattern[TypeUnify]] :=
    With[{
        opts = Association[iopts]
    },
        <|
            "TypeEnvironment" -> Lookup[opts, "TypeEnvironment", Undefined]
        |>
    ]



(*
 A version of TypeUnify that returns Null if it can't unify rather than throwing an exception.
*)
TypeUnifyCatch[ args___] :=
	CatchTypeFailure[
		TypeUnify[args],
		_,
	    Null&
	]
	


TypeUnify[a_, b_, opts:OptionsPattern[]] :=
    TypeUnify[<||>, a, b, opts]


(*
  Raise a TypeErrror[ Failure[tag, ...]].  If this is an occurs check the tag is
  "TypeUnificationOccursCheck" otherwise it is "TypeUnify".  This is to distinguish 
  between the different errors which might need different handling.  AlternativeConstraint 
  does this.
*)

makeFailure[ st_, a_, b_, ex:TypeError[Failure["TypeUnificationOccursCheck", __]]] :=
	TypeFailure["TypeUnificationOccursCheck",
    		"failed to unify `1` and `2` because of `3`",
   			a,
   			b,
   			{ex}	 
    	]

makeFailure[ st_, a_, b_, ex_] :=
	TypeFailure["TypeUnify",
    		"failed to unify `1` and `2` because of `3`",
   			a,
   			b,
   			{ex}	 
    	]

TypeUnify[st0_, a_?TypeObjectQ, b_?TypeObjectQ, opts:OptionsPattern[]] :=
	CatchTypeFailure[
		With[{
		   st = Join[makeState[opts], st0]
		},
		With[{
		   ef = unify[st, a, b]
		},
		   ef
		]],
		_,
		makeFailure[st0, a, b, #]&
	];


TypeUnify[ st_, a_, b_, opts:OptionsPattern[]] :=
	ThrowException[{"Unknown arguments to TypeUnify", {a,b}}]


TypeUnifiableQ[a_, b_, opts:OptionsPattern[]] :=
    TypeUnifiableQ[<||>, a, b]

TypeUnifiableQ[st0_, a_, b_, opts:OptionsPattern[]] :=
    With[{
       st = Join[makeState[opts], st0]
    },
	With[{
		sub = CatchException[
	        TypeUnify[st, a, b],
	        {{_, Null&}}
	    ]
	},
		TypeSubstitutionQ[sub]
    ]];


TypeUnifyMany[a_, b_, opts:OptionsPattern[]] :=
    TypeUnifyMany[<||>, a, b, opts]

TypeUnifyMany[st0_, a_?ListQ, b_?ListQ, opts:OptionsPattern[]] :=
    With[{
       st = Join[makeState[opts], st0]
    },
    With[{
       ef = unifyMany[st, a, b]
    },
       ef
    ]];

TypeUnifyMany[ st_, a_, b_, opts:OptionsPattern[]] :=
    ThrowException[{"Unknown arguments to TypeUnifyMany", {a,b}}]

TypeUnifiableManyQ[a_, b_, opts:OptionsPattern[]] :=
    TypeUnifiableManyQ[<||>, a, b, opts]

TypeUnifiableManyQ[st0_, a_, b_, opts:OptionsPattern[]] :=
    With[{
       st = Join[makeState[opts], st0]
    },
	    CatchException[
	        With[{sub = TypeUnifyMany[st, a, b, "TypeEnvironment" -> st["TypeEnvironment"]]},
	            TypeSubstitutionQ[sub]
	        ],
	        {{_, False&}}
	    ]
    ];


unifyMany[st0_, {}, {}, opts:OptionsPattern[]] :=
    With[{
       st = Join[makeState[opts], st0]
    },
        CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]]
    ];





(*
   Sequence
*)



computeLength[ t_, min_, max_, givenLen_] :=
	Which[
		min > givenLen,
			TypeFailure[
        		"TypeSequence Length",
        		"the number of arguments `1` given to type `2` do not satisfy the minimum length `3`",
       		 givenLen, t, min
    		],
		max =!= Infinity && max < givenLen,
			TypeFailure[
        		"TypeSequence Length",
        		"the number of arguments `1` given to type `2 do not satisfy the maximum length `3`",
       		 givenLen, t, min
    		],
    	True,
			givenLen]

seqVars[t_?TypeObjectQ, ii_] :=
	t

unifyMany[st_, {t_?TypeSequenceQ, tfixed___}, t2_List] :=
	unifyMany[st, t2, {t, tfixed}]

unifyMany[st_, t1_List, {t_?TypeSequenceQ, tfixed___}] :=
	Module[ {var},
		var = t["binding"];
		unifySequence[st, var, t1, t, {tfixed}]
	]

unifySequence[st_, var:(_?TypeVariableQ|None), t1_List, t_?TypeSequenceQ, tfixed_List] :=
	With[{
		min = t["min"],
		max = t["max"]
	},
	With[{
		len = computeLength[t, min, max, Length[t1]-Length[tfixed]]
	},
    With[{
        tvars = Table[
            seqVars[t["type"], ii],
            {
                ii,
                1,
                len
            }
        ]
    },
    With[{
        sub1 = unifyMany[st, t1, Join[tvars, tfixed]]
    },
    	If[var === None,
    		sub1,
			sub1["compose", bind[st, var -> SequenceSubstitution[t1]]]]
    ]]]]






unifyMany[st_, {v1_, v1s___}, {v2_, v2s___}] :=
    With[{
        sub1 = unify[st, v1, v2]
    },
    With[{
        sub2 = unifyMany[
            st,
            sub1["apply", {v1s}],
            sub1["apply", {v2s}]
        ]
    },
        sub2["compose", sub1]
    ]];


unifyMany[st_, {}, t:{__}] :=
    TypeFailure[
        "TypeUnificationListSizeMismatch",
        "the type `1` did not match the expected lhs (TODO:: make this error make more sense)",
        t
    ]

unifyMany[st_, t:{__}, {}] :=
    TypeFailure[
        "TypeUnificationListSizeMismatch",
        "the type `1` did not match the expected lhs (TODO:: make this error make more sense)",
        t
    ]


unify[st_, v1_?TypeApplicationQ, v2_?TypeApplicationQ] :=
    With[{
        sub1 = unify[st, v1["type"], v2["type"]]
    },
    With[{
        sub2 = unifyMany[st, sub1["apply", v1["arguments"]], sub1["apply", v2["arguments"]]]
    },
        sub2["compose", sub1]
    ]];



unify[st_, v1_?TypeArrowQ, v2_?TypeArrowQ] :=
    With[{
        sub1 = unifyMany[st, v1["arguments"], v2["arguments"]]
    },
    With[{
        sub2 = unify[st, sub1["apply", v1["result"]], sub1["apply", v2["result"]]]
    },
        sub2["compose", sub1]
    ]];

unify[st_, v1_?TypePredicateQ, v2_?TypePredicateQ] :=
    If[v1["underlyingAbstractType"] === v2["underlyingAbstractType"],
        unifyMany[st, v1["types"], v2["types"]],
        TypeFailure[
            "TypeUnificationPredicates",
            "the predicates types `1` and `2` cannot be unified",
            v1["unresolve"],
            v2["unresolve"]
        ]
    ];

unify[st_, v_, v_] :=
    CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]]


unify[st_, v1_?(TypeIsA["Error"]), v2_] :=
    bind[st, v2 -> v1]
    
unify[st_, v1_, v2_?(TypeIsA["Error"])] :=
    bind[st, v1 -> v2]
    
unify[st_, v1_?TypeBottomQ, v2_?TypeTopQ] :=
    TypeFailure[
        "TypeUnification",
        "cannot unify bottom `1` and top `2` types",
        v1["unresolve"],
        v2["unresolve"]
    ];
    
unify[st_, v1_?TypeTopQ, v2_?TypeBottomQ] :=
    TypeFailure[
        "TypeUnification",
        "cannot unify bottom `1` and top `2` types",
        v1["unresolve"],
        v2["unresolve"]
    ];
    
unify[st_, v1_?TypeBottomQ, v2_] := CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]];
unify[st_, v1_?TypeTopQ, v2_] := CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]];
unify[st_, v1_, v2_?TypeBottomQ] := CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]];
unify[st_, v1_, v2_?TypeTopQ] := CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]];


unify[st_, v1_?TypeEvaluateQ, v2_?TypeLiteralQ] :=
	unify[ st, v2, v1]
	
unify[st_, v1_?TypeLiteralQ, v2_?TypeEvaluateQ] :=
	v2["solve", st["TypeEnvironment"], v1]



unify[st_, v1_?TypeLiteralQ, v2_?TypeLiteralQ] := 
    With[{
        val1 = v1["value"],
        val2 = v2["value"]
    },
	    Which[
	       TypeObjectQ[val1] && TypeObjectQ[val2],
	           unifyMany[st, {val1, v1["type"]}, {val2, v2["type"]}],
           TypeObjectQ[val1] && TypeTopQ[val1],
               unify[st, v1["type"], v2["type"]],
           TypeObjectQ[val2] && TypeTopQ[val2],
               unify[st, v1["type"], v2["type"]],
	       v1["value"] === v2["value"],
	           unify[st, v1["type"], v2["type"]],
	       True,
	           TypeFailure[
		           "TypeUnificationLiteral",
		           "the literal types `1` and `2` cannot be unified",
		           v1["unresolve"],
		           v2["unresolve"]
	           ]
	    ]
    ];

canUnifyProjection[t_] := 
	TypeProjectionQ[t] &&
	TypeApplicationQ[t["type"]] &&
	(IntegerQ[t["value"]] || (TypeLiteralQ[t["value"]] && IntegerQ[t["value"]["value"]]));

notTypeProjection[t_] :=
    TypeObjectQ[t] && !TypeProjectionQ[t]

unify[st_, v1_?canUnifyProjection, v2_?notTypeProjection] :=
	unify[st, v1["project"], v2]
unify[st_, v1_?notTypeProjection, v2_?canUnifyProjection] :=
	unify[st, v1, v2["project"]];
unify[st_, v1_?TypeProjectionQ, v2_?TypeProjectionQ] :=
	Which[
		TypeLiteralQ[v1["value"]] && TypeLiteralQ[v2["value"]],
			If[!v1["value"]["sameQ", v2["value"]],
			   Return[
			   	TypeFailure[
		           "TypeUnificationProjection",
		           "the type projection `1` and `2` cannot be unified",
		           v1["unresolve"],
		           v2["unresolve"]
	           	]
	           ]
			];
			Which[
				TypeApplicationQ[v1["type"]] && TypeApplicationQ[v2["type"]],
					unify[st, v1["project"], v2["project"]],
				TypeApplicationQ[v1["type"]],
					unify[st, v1["project"], v2["type"]],
				TypeApplicationQ[v2["type"]],
					unify[st, v1["type"], v2["project"]],
				True,
					unify[st, v1["type"], v2["type"]]
			],
		TypeLiteralQ[v1["value"]],
			unifyMany[st, {v1["value"], v1["type"]}, {v2["value"], v2["type"]}],
		TypeLiteralQ[v2["value"]],
			unifyMany[st, {v1["value"], v1["type"]}, {v2["value"], v2["type"]}],
		True,
			unifyMany[st, {v1["value"], v1["type"]}, {v2["value"], v2["type"]}]
	];
	
(*
unify[st_, v1_?TypeLiteralQ, v2_?TypeConstructorQ] := (
    v1["setProperty", "valueValidQ" -> False]; 
    unify[st, v1["type"], v2]
);
unify[st_, v1_?TypeConstructorQ, v2_?TypeLiteralQ] := (
    v2["setProperty", "valueValidQ" -> False]; 
    unify[st, v1, v2["type"]]
);
*)

unify[st_, v1_?TypeConstructorQ, v2_?TypeConstructorQ] :=
    If[v1["sameQ", v2],
        CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]],
        TypeFailure[
	        "TypeUnificationConstructors",
	        "the constructor types `1` and `2` cannot be unified",
	        v1["unresolve"],
	        v2["unresolve"]
	    ]
    ];


(* will not unify with anything except themselves and meta type variables *)
unifySkolem[st_, v1_?TypeVariableQ, v2_?TypeVariableQ] :=
    Which[
        v1["skolemVariableQ"] && v2["skolemVariableQ"],
            If[v1["sameQ", v2],
                CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]],
                TypeFailure[
                    "TypeUnificationSkolem",
                    "the skolem variables `1` and `2` are not equal",
                    v1["unresolve"],
                    v2["unresolve"]
                ]
            ],
        (v1["skolemVariableQ"] && v2["metaVariableQ"]) ||
        (v2["skolemVariableQ"] && v1["metaVariableQ"]),
            CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]],
        True,
            TypeFailure[
                "TypeUnificationSkolem",
                "expecting the two type variables `1` and `2` to have skolem kinds",
                v1["unresolve"],
                v2["unresolve"]
            ]
    ];
unifySkolem[st_, v1_?TypeVariableQ, v2_] :=
    bind[st, v1 -> v2];
unifySkolem[st_, v1_, v2_] :=
    TypeFailure[
        "TypeUnificationSkolem",
        "expecting the two types `1` and `2` to be type variables",
        v1,
        v2
    ];

unify[st_, v1_?TypeVariableQ, v2_] :=
    If[v1["skolemVariableQ"],
        unifySkolem[st, v1, v2],
        bind[st, v1 -> v2]
    ]

unify[st_, v1_, v2_?TypeVariableQ] :=
    If[v2["skolemVariableQ"],
        If[TypeVariableQ[v1] && v1["skolemVariableQ"],
            unifySkolem[st, v2, v1],
            TypeFailure[
		        "TypeUnificationSkolem",
		        "expecting the left hand side `1` of skolem unification with `2` to be a skolem variable",
		        v1,
		        v2
		    ]
        ],
        bind[st, v2 -> v1]
    ]




bind[st_, v1_ -> v1_] :=
    CreateTypeSubstitution["TypeEnvironment" -> st["TypeEnvironment"]]

bind[st_, v1_?TypeVariableQ -> v2_] :=
    If[occursCheck[v1, v2],
        TypeFailure[
            "TypeUnificationOccursCheck",
            "the variable `1` occurs within `2` which results in an infinite type",
            v1,
            v2
        ],
        CreateTypeSubstitution[v1 -> v2, "TypeEnvironment" -> st["TypeEnvironment"]]
    ]

bind[st_, v1_ -> v2_] :=
    TypeFailure[
        "TypeUnificationBind",
        "expected the lhs to be a type variable while binding `1` to `2`",
        v1,
        v2
    ];

unify[st_, v1_, v2_] :=
    TypeFailure[
	        "TypeUnification Failure",
	        "the types `1` and `2` cannot be unified",
	        v1,
	        v2
	    ]

occursCheck[a_?TypeVariableQ, SequenceSubstitution[t_]] :=
	Module[{free = Values[Join @@ Map[#["free"]&, t]]},
		AnyTrue[free, #["sameQ", a]&]
	]
	

occursCheck[a_?TypeVariableQ, t_] :=
	If[False && t["isNamedApplication", "TypeJoin"],
		f[];
		False
		,
		AnyTrue[Values[t["free"]], #["sameQ", a]&]]


occursCheck[___] :=
	False

(*
  FindGeneralUnifier functionality.
  This returns a type that unifies with both input types.  
  eg for  {Complex[Real]}-> var1 and {Complex[Real]}->NA[var2,var3]
  it returns {Complex[Real]}-> var4,  where var4 is a new type variable.
  This result can be unified with both arguments.
*)

FindGeneralUnifierCatch[ args___] :=
	CatchTypeFailure[
		FindGeneralUnifier[args],
	    _,
	    Null&
	]
	
FindGeneralUnifier[a_, b_, opts:OptionsPattern[]] :=
    FindGeneralUnifier[<||>, a, b, opts]



FindGeneralUnifier[st0_, a_?TypeObjectQ, b_?TypeObjectQ, opts:OptionsPattern[]] :=
	With[{
	   st = Join[makeState[opts], st0]
	},
	With[{
	   ef = findGeneral[st, a, b]
	},
	   ef
	]];


findGeneral[st_, v1_?TypeConstructorQ, v2_?TypeConstructorQ] :=
    If[v1["sameQ", v2],
        v1,
        TypeFailure[
	        "TypeUnificationConstructors",
	        "the constructor types `1` and `2` cannot be unified",
	        v1["unresolve"],
	        v2["unresolve"]
	    ]
    ]

findGeneral[st_, v1_?TypeApplicationQ, v2_?TypeApplicationQ] :=
    With[{
        type = findGeneral[st, v1["type"], v2["type"]]
    },
    With[{
        args = findGeneralMany[st, v1["arguments"], v2["arguments"]]
    },
        CreateTypeApplication[ type, args]
    ]]

findGeneral[st_, v1_?TypeArrowQ, v2_?TypeArrowQ] :=
    With[{
        args = findGeneralMany[st, v1["arguments"], v2["arguments"]]
    },
    With[{
        result = findGeneral[st, v1["result"], v2["result"]]
    },
        CreateTypeArrow[args, result]
    ]]

findGeneral[st_, v1_?TypeVariableQ, v2_] :=
    CreateTypeVariable["common"]

findGeneral[st_, v1_, v2_?TypeVariableQ] :=
    CreateTypeVariable["common"]

findGeneral[st_, v1_?TypeLiteralQ, v2_?TypeLiteralQ] := 
    If[
		v1["value"] === v2["value"],
			CreateTypeLiteral[ v1["value"], findGeneral[st, v1["type"], v2["type"]]],
			findGeneralFailure[st, v1, v2]
    ]


findGeneral[st_, v1_, v2_] :=
    findGeneralFailure[st, v1, v2]

findGeneralFailure[st_, v1_, v2_] :=
    TypeFailure[
	        "TypeCommonUnification Failure",
	        "the types `1` and `2` cannot be unified",
	        v1,
	        v2
	    ]




findGeneralMany[st_, l1_, l2_] :=
	If[Length[l1] =!= Length[l2],
    	TypeFailure[
        	"TypeCommonUnificationListSizeMismatch",
        	"the list of types `1` and `2` do not have the same length",
        	l1, l2
    	]
    	,
    	MapThread[ findGeneral[st,#1, #2]&, {l1, l2}]
	]





End[];

EndPackage[];
