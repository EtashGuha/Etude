BeginPackage["Compile`AST`Macro`Builtin`Types`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["TypeFramework`"] (* for MetaData *)


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Native`BitCast,
			Native`BitCast[a_, b_String] ->
				Native`BitCast[a, TypeSpecifier[b]],
			Native`BitCast[a_, b_String[args___]] ->
				Native`BitCast[a, TypeSpecifier[b[args]]]
		];
        RegisterMacro[env, Native`ReinterpretCast,
            Native`ReinterpretCast[a_, b_String] ->
                Native`ReinterpretCast[a, TypeSpecifier[b]],
            Native`ReinterpretCast[a_, b_String[args___]] ->
                Native`ReinterpretCast[a, TypeSpecifier[b[args]]]
        ];
		RegisterMacro[env, Compile`TypeJoin,
			Compile`TypeJoin[s_Type, t_Type] -> (
				Native`DeclareVariable[Typed[a, s]] = Undefined;
				Native`DeclareVariable[Typed[b, t]] = Undefined;
				MetaData[<| "MacroFixed" -> True |>]@Compile`TypeJoin[a, b]
			),
			Compile`TypeJoin[s_, t_Type] -> (
				Native`DeclareVariable[Typed[b, t]] = Undefined;
				MetaData[<| "MacroFixed" -> True |>]@Compile`TypeJoin[s, b]
			),
			Compile`TypeJoin[s_Type, t_] -> (
				Native`DeclareVariable[Typed[a, s]] = Undefined;
				MetaData[<| "MacroFixed" -> True |>]@Compile`TypeJoin[a, t]
			),
			Compile`TypeJoin[s_String, t_String] -> 
				Compile`TypeJoin[TypeSpecifier[s], TypeSpecifier[t]],
			Compile`TypeJoin[s_String, t_] -> 
				Compile`TypeJoin[TypeSpecifier[s], t],
			Compile`TypeJoin[s_, t_String] -> 
				Compile`TypeJoin[s, TypeSpecifier[t]]
		];
	]

RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
