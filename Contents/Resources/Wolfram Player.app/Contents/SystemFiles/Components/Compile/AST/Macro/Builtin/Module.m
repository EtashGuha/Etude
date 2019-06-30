BeginPackage["Compile`AST`Macro`Builtin`Module`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Module,
			Module[{}, body_] -> (body) (* you have to put parens here because the result of the module may be captured *)
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
