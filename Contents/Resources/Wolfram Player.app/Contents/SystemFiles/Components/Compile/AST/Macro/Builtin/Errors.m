BeginPackage["Compile`AST`Macro`Builtin`Errors`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["Compile`API`RuntimeErrors`"]
Needs["CompileAST`Create`Construct`"]

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Native`ErrorCode,
			Native`ErrorCode[text_String] ->
				Compile`Internal`MacroEvaluate[
					convertToCode[text]
				]
		];
	]

convertToCode[text_] :=
	Module[ {str = "Unknown"},
		If[text["literalQ"] && StringQ[text["data"]],
			str = text["data"]];
		CreateMExprLiteral[ErrorCodeFromErrorText[str]]
	]


RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
