BeginPackage["Compile`AST`Macro`Builtin`Numerical`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
				
		RegisterMacro[env, Between,
			Between[a_, {min_, max_}] -> min <= a && a <= max
		];
		
	]


RegisterCallback["SetupMacros", setupMacros]


End[]
EndPackage[]
