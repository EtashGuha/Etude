BeginPackage["Compile`AST`Macro`Builtin`Function`"]

Begin["`Private`"]

Needs["Compile`AST`Transform`Beta`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Function,
			f_Function[arg_] -> Apply[f, {arg}],
			f_Function[arg1_, arg2_] -> Apply[f, {arg1, arg2}]
		];
		
		RegisterMacro[env, Apply,
			Apply[f_Function, vals_] ->
				Compile`Internal`MacroEvaluate[MExprBetaReduce[f, vals]]
		];
	]


RegisterCallback["SetupMacros", setupMacros]


End[]
EndPackage[]
