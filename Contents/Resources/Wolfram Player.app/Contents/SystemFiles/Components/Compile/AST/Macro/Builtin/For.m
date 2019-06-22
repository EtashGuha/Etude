BeginPackage["Compile`AST`Macro`Builtin`For`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, For,
			For[init_, cond_, incr_, body_] ->
				Module[{},
					init;
					While[cond,
						body;
						incr
					];
				]
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
