
BeginPackage["TypeFramework`Basic`LambdaMExpr`"]

GenerateLambdaMExpr;

Begin["`Private`"]

Needs["CompileAST`Export`FromMExpr`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Base`"]
Needs["CompileAST`Class`Symbol`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Inference`TypeInferenceState`"]
Needs["CompileUtilities`Error`Exceptions`"]

dispatch = <|
    "System`Integer" -> generateInteger,
    "System`Real" -> generateReal,
    "System`True" -> generateBoolean,
    "System`False" -> generateBoolean,
    "System`Symbol" -> generateSymbol,
    "System`If" -> generateIf,
    "System`Module" -> generateModule,
    "System`Pattern" -> generatePattern,
    "System`Blank" -> generateBlank,
    "System`Typed" -> generateTyped,
    "System`Set" -> generateSet,
    "System`Equal" -> generateCall,
    "System`Less" -> generateCall,
    "System`LessEqual" -> generateCall,
    "System`Greater" -> generateCall,
    "System`GreaterEqual" -> generateCall,
    "System`Plus" -> generateCall,
    "System`Times" -> generateCall,
    "System`Subtract" -> generateCall,
    "System`Divide" -> generateCall
|>;

GenerateLambdaMExpr[state_, mexpr_?MExprQ] :=
    generate0[state, mexpr]; 
   
   
generate[st_?TypeInferenceStateQ,  mexpr_] :=
    If[st["hasProperty", "LanguageExtensions"],
        With[{
            langExt = st["getProperty", "LanguageExtensions"],
            hd = mexpr["head"]["fullName"]
        },
            If[KeyExistsQ[langExt, hd],
                langExt[hd][st, mexpr],
                generate0[st, mexpr]
            ]
        ],
        generate0[st, mexpr]
    ];

generate0[st_, mexpr_?MExprSymbolQ] :=
    generateSymbol[st, mexpr];
generate0[st_, mexpr_?MExprLiteralQ] :=
    With[{
        d = Which[
            mexpr["sameQ", True],
                dispatch["System`True"],
            mexpr["sameQ", False],
                dispatch["System`False"],
            True,
                Lookup[dispatch, mexpr["head"]["fullName"], errorDispatch]
        ]
    },
        d[st, mexpr]
    ];
generate0[st_, mexpr_?MExprNormalQ] :=
    With[{
        headName = mexpr["head"]["fullName"]
    },
    With[{
        d = Lookup[dispatch, headName, errorDispatch]
    },
        d[st, mexpr]
    ]];

generateInteger[st_, mexpr_] :=
    With[{
        tv = CreateTypeVariable[ "IntLit$" <> mexpr["toString"]]
	},
	With[{
	    pred = CreateTypePredicate[tv, MemberQ["Number"]]
	},
	    st["appendProveConstraint", pred, <|"source" -> mexpr|>];
	    tv
	]];

generateReal[st_, mexpr_] :=
    With[{
        tv = CreateTypeVariable[ "RealLit$" <> mexpr["toString"]]
    },
    With[{
        pred = CreateTypePredicate[tv, MemberQ["RealFloatingPoint"]]
    },
        st["appendProveConstraint", pred, <|"source" -> mexpr|>];
        tv
    ]];


generateBoolean[st_, mexpr_] :=
    With[{
        tv = CreateTypeVariable[ "BooleanLit$" <> mexpr["toString"]],
        ty = st["resolveType", TypeSpecifier["Boolean"]]
    },
        st["appendEqualConstraint", tv, ty, <|"source" -> mexpr|>];
        tv
    ];

idOfSymbol[mexpr_] := If[mexpr["context"] === "System`",
    ReleaseHold[mexpr["data"]],
    mexpr["toString"]
];

generateSymbol[st_, mexpr_] :=
    With[{
        tv = CreateTypeVariable["Var$" <> mexpr["toString"]]
    },
        st["addAssumption", idOfSymbol[mexpr] -> {tv}, <|"source" -> mexpr|>];
        tv
    ];

generateIf[st_, mexpr_] :=
    With[{
        condTy = generate[st, mexpr["part", 1]],
        trueTy = generate[st, mexpr["part", 2]],
        falseTy = generate[st, mexpr["part", 3]],
        tv = CreateTypeVariable["If"]
    },
        st["appendEqualConstraint", condTy, st["resolveType", TypeSpecifier["Boolean"]], <|"source" -> mexpr|>];
        st["appendEqualConstraint", trueTy, tv, <|"source" -> mexpr|>];
        st["appendEqualConstraint", falseTy, tv, <|"source" -> mexpr|>];
        tv
    ];


(* Assume Module is of the form
 * Module[{var1=val1, ...., varn=valn}, ...]
 * i.e. all variables are assigned in the
 * first argument of module
 *)
generateModule[st_, mexpr_] :=
    With[{
        tv = CreateTypeVariable["Module"],
        defs = mexpr["part", 1],
        body = mexpr["part", 2]
    },
    With[{
        lhss = Table[
            defs["part", ii]["part", 1],
            {ii, defs["length"]}
        ],
        rhss = Table[
            generate[st, defs["part", ii]["part", 2]],
            {ii, defs["length"]}
        ]
    },
    With[{
        oldAssumptions = st["cloneAssumptions"],
        resTy = generate[st, body]
    },
    With[{
        bodyAssumptions = st["cloneAssumptions"]
    },
        st["dropAssumptions", idOfSymbol /@ lhss];
        st["mergeAssumptions", oldAssumptions];
        st["appendEqualConstraint", tv, resTy, <|"source" -> mexpr|>];
        MapThread[
            Function[{lhs, rhs},
                Do[
                    st["appendImplicitInstanceConstraint", var, rhs, st["monomorphicSet"]],
                    {var, bodyAssumptions["lookup", idOfSymbol[lhs], {}]}
                ]
            ],
            {lhss, rhss}
        ];
        tv
    ]]]];

generateFunction[st_, mexpr_] :=
    0;

generateCompoundExpression[st_, mexpr_] :=
    0;

generateCall[st_, mexpr_] :=
    Module[{funT, varsT, funTy, tv},
        funT = generate[st, mexpr["head"]];
        varsT = generateMany[st, mexpr["arguments"]];
        tv = CreateTypeVariable[mexpr["head"]["toString"] <> "$$" <> ToString[mexpr["id"]]];
        funTy = st["resolveType", TypeSpecifier[ varsT -> tv]];
        st["appendEqualConstraint", funT, funTy, <|"source" -> mexpr|>];
        tv
    ];

generateTyped[st_, mexpr_] :=
    With[{
        expr = mexpr["part", 1],
        tyExpr = mexpr["part", 2]
    },
    With[{
        ty = st["resolveType", ReleaseHold[FromMExpr[tyExpr]]]
    },
        ty
    ]];

generateSet[st_, mexpr_] :=
    With[{
        lhs = mexpr["part", 1],
        rhs = mexpr["part", 2]
    },
    With[{
        tv = CreateTypeVariable[lhs["toString"]],
        ty = generate[st, rhs]
    },
        st["appendEqualConstraint", tv, ty, <|"source" -> mexpr|>];
        tv
    ]];

generateUnbound[st_, mexpr_] :=
    With[{
        tv = CreateTypeVariable[mexpr["toString"] <> "$" <> ToString[mexpr["id"]]]
    },
        tv
    ];

generateBlank[st_, mexpr_] :=
    If[mexpr["length"] === 0,
        generateUnbound[st, mexpr],
        With[{
            ty = st["resolveType", TypeSpecifier[mexpr["part", 1]["toString"]]]
        },
            ty
        ]
    ];


generatePattern[st_, mexpr_] :=
    With[{
        name = If[mexpr["length"] == 1, "pattern", mexpr["part", 1]["toString"]]
    },
    With[{
        tv = CreateTypeVariable[name],
        ty = generate[st, mexpr["part", Max[mexpr["length"], 2]]]
    },
        st["appendEqualConstraint", tv, ty, <|"source" -> mexpr|>];
        tv
    ]];

errorDispatch[st_, mexpr_] :=
    ThrowException[TypeInferenceException[{"Unknown argument to infer mexpr ", {mexpr}}]]

generateMany[st_, p_?ListQ] :=
    Map[generate[st, #]&, p]

End[]

EndPackage[]
