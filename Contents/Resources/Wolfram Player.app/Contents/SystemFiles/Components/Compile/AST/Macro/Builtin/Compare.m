BeginPackage["Compile`AST`Macro`Builtin`Compare`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]



mexprSameQ[x_, y_] :=
	With[{res = x["sameQ", y]},
		CreateMExprLiteral[res]
	]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		
		RegisterMacro[env, Equal,
			Equal[True, False] -> False,
			Equal[False, True] -> False, 
			Equal[Automatic, _String] -> False,
			Equal[_String, Automatic] -> False,
			Equal[x_Integer, y_Integer] -> 
				Compile`Internal`MacroEvaluate[
					mexprSameQ[x, y]
				],
			Equal[x_String, y_String] -> 
				Compile`Internal`MacroEvaluate[
					mexprSameQ[x, y]
				],
			Equal[a_, a_] -> True,
			Equal[Greater[args__], True] -> Greater[args],
			Equal[GreaterEqual[args__], True] -> GreaterEqual[args],
			Equal[Greater[args__], False] -> LessEqual[args],
			Equal[GreaterEqual[args__], False] -> Less[args],
			Equal[Less[args__], True] -> Less[args],
			Equal[LessEqual[args__], True] -> LessEqual[args],
			Equal[Less[args__], False] -> GreaterEqual[args],
			Equal[LessEqual[args__], False] -> Greater[args]
		];
		
		RegisterMacro[env, SameQ,
			SameQ[True, False] -> False,
			SameQ[False, True] -> False, 
			SameQ[Automatic, _String] -> False,
			SameQ[_String, Automatic] -> False,
			SameQ[x_Integer, y_Integer] -> 
				Compile`Internal`MacroEvaluate[
					mexprSameQ[x, y]
				],
			SameQ[x_String, y_String] -> 
				Compile`Internal`MacroEvaluate[
					mexprSameQ[x, y]
				],
			SameQ[a_, a_] -> True,
			SameQ[Greater[args__], True] -> Greater[args],
			SameQ[GreaterEqual[args__], True] -> GreaterEqual[args],
			SameQ[Greater[args__], False] -> LessEqual[args],
			SameQ[GreaterEqual[args__], False] -> Less[args],
			SameQ[Less[args__], True] -> Less[args],
			SameQ[LessEqual[args__], True] -> LessEqual[args],
			SameQ[Less[args__], False] -> GreaterEqual[args],
			SameQ[LessEqual[args__], False] -> Greater[args]
		];
		
		RegisterMacro[env, Unequal,
			Unequal[args__] -> Not[Equal[args]]
		];
		
		RegisterMacro[env, UnsameQ,
			UnsameQ[args__] -> Not[SameQ[args]]
		];
		
		RegisterMacro[env, Greater,
			Greater[x_Integer, y_Integer] ->
				Compile`Internal`MacroEvaluate[
					greaterQ[x, y]
				]
		];
		
		RegisterMacro[env, GreaterEqual,
			GreaterEqual[x_Integer, y_Integer] ->
				Compile`Internal`MacroEvaluate[
					greaterEqualQ[x, y]
				]
		];
		
		RegisterMacro[env, Less,
			Less[x_Integer, y_Integer] ->
				Compile`Internal`MacroEvaluate[
					lessQ[x, y]
				]
		];
		
		RegisterMacro[env, LessEqual,
			LessEqual[x_Integer, y_Integer] ->
				Compile`Internal`MacroEvaluate[
					lessEqualQ[x, y]
				]
		];
	]

greaterQ[x_, y_] := With[{res = x["data"] > y["data"]}, CreateMExprLiteral[res]]

greaterEqualQ[x_, y_] := With[{res = x["data"] >= y["data"]}, CreateMExprLiteral[res]]
lessQ[x_, y_] := With[{res = x["data"] < y["data"]}, CreateMExprLiteral[res]]
lessEqualQ[x_, y_] := With[{res = x["data"] <= y["data"]}, CreateMExprLiteral[res]]



RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
