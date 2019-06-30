BeginPackage["Compile`AST`Macro`Builtin`Math`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`Literal`"]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, PreIncrement,
			PreIncrement[a_] -> Module[{val$$ = a},
				a = val$$ + 1]
		];
		
		RegisterMacro[env, Increment,
			Increment[a_] -> Module[{val$$ = a},
				a = val$$ + 1;
				val$$
			]
		];
		
		RegisterMacro[env, PreDecrement,
			PreDecrement[a_] -> Module[{val$$ = a},
				a = val$$ - 1]
		];
		
		RegisterMacro[env, Decrement,
			Decrement[a_] -> Module[{val$$ = a},
				a = val$$ - 1;
				val$$
			]
		];
		
		RegisterMacro[env, AddTo,
			AddTo[a_, v_] -> Module[{val$$ = a},
				a = val$$ + v]
		];
		
		RegisterMacro[env, SubtractFrom,
			SubtractFrom[a_, v_] -> Module[{val$$ = a},
				a = val$$ - v]
		];
		
		RegisterMacro[env, TimesBy,
			TimesBy[a_, v_] -> Module[{val$$ = a},
				a = val$$ * v]
		];
		
		RegisterMacro[env, DivideBy,
			DivideBy[a_, v_] -> Module[{val$$ = a},
				a = val$$ / v]
		];
		
		RegisterMacro[env, Mod,
			Mod[a_, b_, d_] -> Mod[a, b] + d
		];
		
		RegisterMacro[env, Power,
			Power[_, 0] -> 1,
			Power[a_, 1] -> a,
			Power[a_, 2] -> Module[{arg$$ = a}, arg$$*arg$$],
			Power[a_, -1] -> Divide[1, a],
			Power[E, x_] -> Exp[x]
		];
		
		RegisterMacro[env, Minus,
			Minus[0] -> 0,
			Minus[Minus[a_]] -> a
		];
		
		negativeConstantQ[x_] :=
			If[ MExprLiteralQ[x], x["data"] < 0, False];
		
		makeNegative[x_] :=
			If[MExprLiteralQ[x], CreateMExprLiteral @@ {-x["data"]}, Minus[x]];
			
		RegisterMacro[env, Plus,
			Plus[] -> 0,
			Plus[a_] -> a,
			Plus[a_, 0] -> a,
			Plus[0, a_] -> a,
			Plus[Minus[a_], Minus[b_]] -> Minus[Plus[a,b]],
			Plus[Minus[a_], b_] -> Subtract[b,a],
			Plus[a_, Minus[b_]] -> Subtract[a,b],
			Plus[a_?negativeConstantQ, b_?negativeConstantQ] -> 
					Minus[Plus[Compile`Internal`MacroEvaluate[makeNegative[a]],Compile`Internal`MacroEvaluate[makeNegative[b]]]],
			Plus[a_?negativeConstantQ, b_] -> 
					Subtract[b,Compile`Internal`MacroEvaluate[makeNegative[a]]],
			Plus[a_, b_?negativeConstantQ] -> 
					Subtract[a,Compile`Internal`MacroEvaluate[makeNegative[b]]],
			Plus[a_, b_, c__] -> Plus[a, Plus[b, c]]
		];
	
		RegisterMacro[env, Times,
			Times[] -> 1,
			Times[a_] -> a,
			Times[-1, a_] -> Minus[a],
			Times[a_, 1] -> a,
			Times[1, a_] -> a,
			Times[a_, Power[b_, -1]] -> Divide[a, b],
			Times[a_, b_, c__] -> Times[a, Times[b, c]]
		];
		
		RegisterMacro[env, Subtract,
			Subtract[a_, 0] -> a,
			Subtract[0, a_] -> Minus[a]
		];
		
		RegisterMacro[env, Floor,
			Floor[x_Integer] -> x
		];
		
		RegisterMacro[env, Ceiling,
			Ceiling[x_Integer] -> x
		];
				
		RegisterMacro[env, Conjugate,
			Conjugate[a_Integer] -> a,
			Conjugate[a_Real] -> a,
			Conjugate[Complex[a_, b_]] -> a - b
		];
		
		RegisterMacro[env, I,
			I -> Complex[0,1]
		];

	]

RegisterCallback["SetupMacros", setupMacros]


End[]
EndPackage[]
