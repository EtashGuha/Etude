BeginPackage["Compile`AST`Macro`Builtin`Range`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Range,
			Range[i_] ->
				Native`Range[1,i],
			Range[i_, j_] ->
				Native`Range[i, j],
			Range[i_, j_, k_] ->
				Native`Range[i, j, k]
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
