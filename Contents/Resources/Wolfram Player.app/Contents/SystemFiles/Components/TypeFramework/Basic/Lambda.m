(* Wolfram Language Package *)

(* Created by the Wolfram Workbench 01-Jun-2017 *)

BeginPackage["TypeFramework`Basic`Lambda`"]
(* Exported symbols added here with SymbolName::usage *) 

GenerateLambda

Lambda
Variable
App
IfCond
Let
Assign



Begin["`Private`"]

Needs["TypeFramework`"]
Needs["TypeFramework`Inference`TypeInferenceState`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`Inference`PatternInferenceState`"]

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

ClearAll[desugar]


GenerateLambda[state_, prog_] :=
    generate[ state, desugar[prog]] 
	
desugar[i_Integer] :=
    Constant[i];
desugar[r_Real] :=
    Constant[r];
desugar[r_Complex] :=
    Constant[r];
desugar[b:(True | False)] :=
    Constant[b];
desugar[s_Symbol] :=
    Variable[SymbolName[s]];
desugar[Lambda[arg_, body_]] :=
    desugar[Lambda[{arg}, body]];
desugar[Lambda[args_?ListQ, body_]] :=
    Lambda[desugar[args], desugar[body]];
desugar[Let[arg_, body_]] :=
    Let[desugar[arg], desugar[body]];
desugar[Let[{}, body_]] :=
    desugar[body];
desugar[Let[{a_, args___}, body_]] :=
    Let[desugar[a], desugar[Let[{args}, body]]];
desugar[Assign[lhs_, rhs_]] :=
    Assign[desugar[lhs], desugar[rhs]];
desugar[App[f_, arg_]] :=
    desugar[App[f, {arg}]];
desugar[App[f_, args_?ListQ]] :=
    App[desugar[f], desugar[args]];
desugar[IfCond[cond_, then_, else_]] :=
    IfCond[desugar[cond], desugar[then], desugar[else]];
desugar[lst_?ListQ] := desugar /@ lst
desugar[x_] := x;

    
free[_Constant] :=
    {};
free[st_, v:Variable[name_]] :=
    {v};
free[Lambda[args_?ListQ, body_]] :=
    Complement[free[body], free[args]];
free[Let[arg_, body_]] :=
    Complement[free[body], free[arg]];
free[Assign[lhs_, rhs_]] :=
    Join[free[lhs], free[rhs]];
free[App[f_, args_]] :=
    Join[free[f], free[args]];
free[IfCond[cond_, then_, else_]] :=
    Join[free[cond], free[then], free[else]];
free[lst_?ListQ] := Catenate[free /@ lst];

getFunctionDefinitions[App[f_, args_]] :=
    {};
getFunctionDefinitions[Lambda[args_, body_]] :=
    getFunctionDefinitions[body];
getFunctionDefinitions[Assign[lhs_, rhs_Lambda]] :=
    Join[{lhs -> rhs}, getFunctionDefinitions[rhs]];
getFunctionDefinitions[Assign[lhs_, rhs_]] :=
    getFunctionDefinitions[rhs];
getFunctionDefinitions[IfCond[cond_, then_, else_]] :=
    Join[getFunctionDefinitions[then], getFunctionDefinitions[else]];
getFunctionDefinitions[x_] := {};
    
dependencyGraph[funs_] :=
    With[{
        vertices = Keys[funs]
    },
    With[{edges = Catenate[Table[
            With[{
                freeVars = free[Lookup[funs, vert]]
            },
            With[{
                following = Lookup[funs, freeVars, Nothing]
            },
                Map[DirectedEdge[vert, #]&, following]
            ]],
            {vert, vertices}
        ]]
    },
        Graph[vertices, edges]
    ]];
    
bindingGroup[prog_] :=
    With[{
        functions = Association[getFunctionDefinitions[prog]]
    },
    With[{
        depGraph = dependencyGraph[functions]
    },
    With[{
        components = ConnectedComponents[depGraph]
    },
        components
    ]]];

generate[st_?TypeInferenceStateQ,  e_] :=
    If[st["hasProperty", "LanguageExtensions"],
        With[{
            langExt = st["getProperty", "LanguageExtensions"],
            hd = Head[e]
        },
            If[KeyExistsQ[langExt, hd],
                langExt[hd][st, e],
                generate0[st, e]
            ]
        ],
        generate0[st, e]
    ];
    
generate0[st_?TypeInferenceStateQ, c:Constant[int_Integer]] :=
  With[{
    tv = CreateTypeVariable[ "IntLit$" <> ToString[int]],
    ty = st["resolveType", TypeSpecifier["Integer"]]
  },
  	st["appendEqualConstraint", tv, ty, <| "source" -> c |>];
  	ty
  ];


generate0[st_?TypeInferenceStateQ, c:Constant[val_Real]] :=
  With[{
    tv = CreateTypeVariable[ "RealLit$" <> ToString[val]],
    ty = st["resolveType", TypeSpecifier["Real"]]
  },
  	st["appendEqualConstraint", tv, ty, <| "source" -> c |>];
  	ty
  ]


generate0[st_?TypeInferenceStateQ, c:Constant[val:(True|False)]] :=
  With[{
    tv = CreateTypeVariable[ "BooleanLit$" <> ToString[val]],
    ty = st["resolveType", TypeSpecifier["Boolean"]]
  },
  	st["appendEqualConstraint", tv, ty, <| "source" -> c |>];
  	ty
  ]


generate0[st_?TypeInferenceStateQ, e:Variable[var_]] :=
  With[{
    tv = CreateTypeVariable["Var$" <> ToString[var]]
  },
  	If[st["isScoped", var],
	  	st["addAssumption",var -> {tv}, <|"source" -> e|>]
	  	, (* Else *)
	  	st["appendLookupConstraint", var, tv, st["monomorphicSet"]["toList"], <|"source" -> e|>]
	];
  	tv
  ];


  
generate0[st_?TypeInferenceStateQ, e:Lambda[args_List, body_]] :=
  Module[{patst, argTypes, bodyType, funType, ran, dom, b2, oldMono, oldScoped},
    patst = CreatePatternInferenceState[generatePattern];
    argTypes = patst["generate", args];
    dom = patst["patternVariables"];
    ran = patst["patternTypes"];
    oldMono = st["cloneMonomorphicSet"];
    oldScoped = st["pushScoped", dom];
    bodyType = generate[st, body];
    st["popScoped", oldScoped];
    st["setMonomorphicSet", oldMono];
    b2 = CreateTypeVariable["Lambda"];
    funType = CreateTypeArrow[ argTypes, bodyType];
    MapThread[ 
    	Function[ {var, type},
    		If[st["hasAssumption", var],
                Do[
                    st["appendEqualConstraint", type, ass, <|"source" -> (var -> {ass})|>],
                    {ass, st["lookupAssumptions", var]}
                ];
    			st["dropAssumption", var]
    	    ]
    	], 
    	{dom, ran}
    ];
    st["appendEqualConstraint", b2, funType, <|"source" -> e|>];
    b2
  ]


generate0[st_?TypeInferenceStateQ, e:Let[Assign[Variable[x_], e1_], e2_]] :=
    Module[{e1T, e2T, assumps, tv, oldScoped},
    	oldScoped = st["pushScoped", {x}];
    	e2T = generate[st, e2];
    	assumps = st["lookupAssumptions", x];
    	st["popScoped", oldScoped];
    	st["dropAssumption", x];
        tv = CreateTypeVariable["Let"];
        Do[ 
        	e1T = generate[st, e1];
        	st["appendEqualConstraint", assum, e1T, <|"source" -> e|>]
        	,
            {assum, assumps}
        ];
        st["appendEqualConstraint", tv, e2T, <|"source" -> e|>];
        tv
    ]
    
generate0[st_?TypeInferenceStateQ, e:App[f_, vars_List]] :=
    Module[{funT, varsT, funTy, tv},
        funT = generate[st, f];
        varsT = generateMany[st, vars];
        tv = CreateTypeVariable["AppRes"];
        funTy = st["resolveType", TypeSpecifier[ varsT -> tv]];
        st["appendEqualConstraint", funT, funTy, <|"source" -> e|>];
        tv
    ]


generate0[st_?TypeInferenceStateQ, e:IfCond[cond_, truebr_, falsebr_]] :=
  Module[{condTy, trueTy, falseTy, tv},
		condTy = generate[st, cond];
		trueTy = generate[st, truebr];
		falseTy = generate[st, falsebr];
		tv = CreateTypeVariable["IfCond"];
		st["appendEqualConstraint",
			condTy,
			st["resolveType", TypeSpecifier["Boolean"]],
			<|"source" -> e|>
		];
		st["appendEqualConstraint", trueTy, tv, <|"source" -> e|>];
		st["appendEqualConstraint", falseTy, tv, <|"source" -> e|>];
		tv
  ];



generate0[st_?TypeInferenceStateQ,  e_] :=
    ThrowException[TypeInferenceException[{"Unknown argument to generate", {e}}]]
    
generate0[args___] :=
    ThrowException[TypeInferenceException[{"Unknown argument to generate", {args}}]]




generateMany[ st_?TypeInferenceStateQ, p_?ListQ] := (
	Map[ generate[st, #]&, p]
)



(*
  Pattern Inference
*)


generatePattern[ st_?PatternInferenceStateQ, Variable[var_]] :=
	With[{
    	tv = CreateTypeVariable["PVar$" <> ToString[var]]
  	},
		st["addBinding",var, tv];
		tv
  	]; 

 
generatePattern[st_?PatternInferenceStateQ, e_] :=
    If[st["hasProperty", "LanguageExtensions"],
        With[{
            langExt = st["getProperty", "LanguageExtensions"],
            hd = Head[e]
        },
            If[!KeyExistsQ[langExt, hd],
                ThrowException[TypeInferenceException[{"Unknown argument to generatePattern", {e}}]]
            ];
            langExt[hd][st, e]
        ],
        ThrowException[TypeInferenceException[{"Unknown argument to generatePattern", {e}}]]
    ];
    
generatePattern[ args___] :=
	ThrowException[TypeInferenceException[{"Unknown argument to generatePattern", {args}}]]


End[]

EndPackage[]

