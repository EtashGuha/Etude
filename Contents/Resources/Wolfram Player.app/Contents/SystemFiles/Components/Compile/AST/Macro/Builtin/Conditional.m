BeginPackage["Compile`AST`Macro`Builtin`Conditional`"]

Begin["`Private`"]

Needs["Compile`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, If,
			If[_, a_, a_] -> a,
			If[True, a_, ___] -> a,
			If[False, _, b_] -> b,
			If[a_, True, False] -> a,
			If[a_, False, True] -> Not[a]
		(*	,
			If[cond_, then_, else_] ->
				Module[{res$$},
					Compile`Internal`If[cond,
						res$$ = Module[{},
							then
						],
						res$$ = Module[{},
							else
						]
					]
				]*)
		];
		
		RegisterMacro[env, Switch,
			
			Switch[ ] -> 
				Compile`Internal`MacroEvaluate[
					switchZeroArgs[]
				],
			
			Switch[_] -> Null,
			
			Switch[arg_, tests__] -> 
				Compile`Internal`MacroEvaluate[
					convertSwitch[arg, tests]
				]
			
			
		];
		
		RegisterMacro[env, Which,
			Which[] -> Null
		];
	]


isBlank[ arg_]:=
	arg["normalQ"] && arg["length"] === 0 && arg["hasHead", CreateMExprSymbol[Blank]]

isAlternatives[ arg_]:=
	arg["normalQ"] && arg["length"] > 0 && arg["hasHead", CreateMExprSymbol[Alternatives]]



getTest[argSym_, val_] :=
	Which[ 
		isBlank[val],
			CreateMExprSymbol[True]
		,
		isAlternatives[val],
			Module[{args = val["arguments"]},
				args = Map[ CreateMExpr[ SameQ, {argSym, #}]&, args];
				CreateMExpr[ Compile`EagerOr, args]
			]
		,
		True,
			CreateMExpr[ SameQ, {argSym, val}]]


convertSwitch[ arg_, branchesIn__] :=
	Module[{branches = {branchesIn}, tests, exprs, argSym, whichArgs, setExpr, ef},
		If[OddQ[Length[branches]],
			ThrowException[CompilerException["Switch called with wrong number of arguments.", Length[branches]]]];
		{tests, exprs} = Transpose[ Partition[branches, 2]];
		argSym = CreateMExprSymbol[argSym];
		tests = Map[ getTest[argSym, #]&, tests];
		whichArgs = Flatten[Transpose[{tests, exprs}]];
		ef = CreateMExpr[ Which, whichArgs];
		ef["setProperty", "originalExpr" -> CreateMExpr[ Switch, {arg, branchesIn}]];
		setExpr = CreateMExpr[ Set, {argSym, arg}];
		setExpr = CreateMExpr[ List, {setExpr}];
		ef = CreateMExpr[ Module, {setExpr, ef}];
		ef
	]


switchZeroArgs[] :=
	ThrowException[CompilerException["Switch called with 0 arguments."]]



RegisterCallback["SetupMacros", setupMacros]

End[]

EndPackage[]
