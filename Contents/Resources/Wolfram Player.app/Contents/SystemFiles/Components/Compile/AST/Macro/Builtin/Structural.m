BeginPackage["Compile`AST`Macro`Builtin`Structural`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]


setupMacros[st_] := Module[{
	env = st["macroEnvironment"]
},
		RegisterMacro[env, AppendTo,
			AppendTo[sym_Symbol, elem_] ->
				(sym = Append[sym, elem])
		];
		
		RegisterMacro[env, PrependTo,
			PrependTo[sym_Symbol, elem_] ->
				(sym = Prepend[sym, elem])
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
