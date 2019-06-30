BeginPackage["Compile`AST`Macro`Builtin`Logical`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},

		RegisterMacro[env, TrueQ,
			TrueQ[True] -> True,
			TrueQ[False] -> False
		];
		RegisterMacro[env, Not,
			Not[True] -> False,
			Not[False] -> True
		];
		RegisterMacro[env, And,
			And[False, ___] -> False,
			And[e_] -> TrueQ[e],
			And[True, rest__] ->
				And[rest],
			And[e_, rest__] ->
				If[e,
					And[rest],
					False
				]
		];
		RegisterMacro[env, Or,
			Or[True, ___] -> True,
			Or[e_] -> TrueQ[e],
			Or[False, rest__] ->
				Or[rest],
			Or[e_, rest__] ->
				If[e,
					True,
					Or[rest]
				]
		];
		RegisterMacro[env, Compile`EagerAnd,
			Compile`EagerAnd[e_] -> e,
			Compile`EagerAnd[a1_, a2_, rest__] ->
				Compile`EagerAnd[a1, Compile`EagerAnd[a2, rest]]
		];
		RegisterMacro[env, Compile`EagerOr,
			Compile`EagerOr[e_] -> e,
			Compile`EagerOr[a1_, a2_, rest__] ->
				Compile`EagerOr[a1, Compile`EagerOr[a2, rest]]
		];
		RegisterMacro[env, Nand,
			Nand[x_, y_] ->
				! x || ! y
		];
		RegisterMacro[env, Nor,
			Nor[x_, y_] ->
				! x && ! y
		];
		RegisterMacro[env, Xor,
			Xor[x_, y_] ->
				(! x || ! y) && (x || y)
		];
		RegisterMacro[env, Xnor,
			Xnor[x_, y_] ->
				(! x || y) && (x || ! y)
		];
		RegisterMacro[env, Boole,
			Boole[True] -> 1,
			Boole[False] -> 0,
			Boole[x_] -> If[x, 1, 0]
		];
		RegisterMacro[env, Unitize,
			Unitize[0] -> 0
		];
	]


RegisterCallback["SetupMacros", setupMacros]


End[]
EndPackage[]
