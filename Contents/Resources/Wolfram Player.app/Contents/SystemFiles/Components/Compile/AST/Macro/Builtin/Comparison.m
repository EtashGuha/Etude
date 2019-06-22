BeginPackage["Compile`AST`Macro`Builtin`Comparison`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]



setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},

		RegisterMacro[env, Greater,
			Greater[a_, b_, c__] ->
				And[ Greater[a,b], Greater[b,c]]
		];
		
		RegisterMacro[env, GreaterEqual,
			GreaterEqual[a_, b_, c__] ->
				And[ GreaterEqual[a,b], GreaterEqual[b,c]]
		];
		
		RegisterMacro[env, Less,
			Less[a_, b_, c__] ->
				And[ Less[a,b], Less[b,c]]
		];
		
		RegisterMacro[env, LessEqual,
			LessEqual[a_, b_, c__] ->
				And[ LessEqual[a,b], LessEqual[b,c]]
		];
		
		RegisterMacro[env, Inequality,
			Inequality[a_, op_, b_] ->
				op[a,b],
			Inequality[a_, op1_, b_, op2_, c__] ->
				And[ op1[a,b], Inequality[b, op2, c]]
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
