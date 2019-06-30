BeginPackage["Compile`AST`Macro`Builtin`Composition`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		
		RegisterMacro[env, Composition,
			Composition[f_, g_] -> Function[{a}, f[g[a]]],
			Composition[f_, g_, rest__] -> Composition[f, Composition[g, rest]]
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
