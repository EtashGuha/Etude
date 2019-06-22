BeginPackage["Compile`AST`Macro`Expand`"]

MacroExpandPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Symbol`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`"]
Needs["CompileAST`Class`MExprAtomQ`"]



$recursionLimit = 5

macroExpand[mexpr_, opts:OptionsPattern[]] :=
	macroExpand[mexpr, <| opts |>]
	
macroExpand[mexpr_, opts_?AssociationQ] :=
	macroExpandWork[getEnvironment[opts], mexpr, opts]
	
macroExpandOnce[mexpr_, opts:OptionsPattern[]] :=
	macroExpandOnce[mexpr, <| opts |>]
	
macroExpandOnce[mexpr_, opts_?AssociationQ] :=
	macroExpandOnceWork[getEnvironment[opts], mexpr, opts]
	

getEnvironment[opts_] :=
	Module[ {env},
		env = Lookup[ opts, "MacroEnvironment", Null];
		If[!MacroEnvironmentQ[env], 
			env = $DefaultMacroEnvironment
		];
		env
	]

macroExpandWork[env_, mexpr_, opts_?AssociationQ] :=
	Module[{
		recur = 0,
		maxRecur = Lookup[ opts, "MacroRecursionLimit", $recursionLimit],
		prev = mexpr,
		curr
	},
		While[recur++ < maxRecur,
			curr = macroExpandOnceWork[env, prev, opts];
			If[testMacroFixed[curr, prev],
				Break[]
			];
			curr["removeProperty", "MacroExpandAgain"];
			prev = curr;
		];
		curr
	]

macroExpandOnceWork[env_, mexpr0_, opts_?AssociationQ] :=
	With[{
		mexpr = mexpr0["clone"]
	},
	With[{
		new = expand1[env, mexpr, opts]
	},
		If[mexpr["properties"]["length"] > 0,
			Do[
				If[!new["hasProperty", prop],
					new["setProperty", prop -> mexpr["getProperty", prop]];
				],
				{prop, mexpr["properties"]["keys"]}
			]
		];
		new
	]]


ClearAll[expand1]
expand1[env_, mexpr_?MExprLiteralQ, opts_] :=
	mexpr
	
expand1[env_, mexpr_?MExprSymbolQ, opts_] :=
	With[{sym = mexpr["symbol"]},
		If[env["hasRules", sym],
			With[{replacements = env["getRules", sym]},
				applyReplacements[ env, replacements, mexpr, opts]
			],
			mexpr
		]
	]

expand1[env_, mexpr_?MExprNormalQ, opts_] :=
	With[{
	   addFixed = testAddMacroFixed[#, mexpr]&
	},
		Which[
			mexpr["hasProperty", "MacroFixed"],
				mexpr,
			MExprSymbolQ[mexpr["head"]] && mexpr["hasHead", Compile`Internal`MacroOptions],
				With[{
				   arg = mexpr["part", 1]
				},
					addFixed@
					If[MExprAtomQ[arg],
				   		With[{opt = Lookup[opts, arg["data"]]},
				   		   If[MissingQ[opt],
				   		      CreateMExpr["Nothing"],
				   		      CreateMExpr[opt]
				   		   ]
				   		],
					    ThrowException[{"Invalid usage for MacroOptions using value ", mexpr}]
					]
				],
			MExprSymbolQ[mexpr["head"]] && mexpr["hasHead", Compile`Internal`MacroEvaluate],
				With[{
				   f = mexpr["part", 1]["getHead"],
				   args = expand1[env, #, opts]& /@ mexpr["part", 1]["arguments"]
				},
                    expand1[env, Apply[f, args], opts]
				],
			MExprSymbolQ[mexpr["head"]] && mexpr["hasHead", Compile`Internal`MacroEvaluateHeld], (* does not evaluate the arguments *)
				With[{
				   f = mexpr["part", 1]["getHead"],
				   args = mexpr["part", 1]["arguments"]
				},
					expand1[env, Apply[f, args], opts]
				],
			MExprSymbolQ[mexpr["head"]],
				With[{
					hd0 = mexpr["head"]["symbol"]
				},
					addFixed@
					If[env["hasRules", hd0],
						With[{
							new = applyReplacements[ env, env["getRules", hd0], mexpr, opts]
						},
							If[MExprNormalQ[new],
								expandParts[env, new, opts],
								new
							]
						],
						expandParts[env, mexpr, opts]
					]
				],
			True, (* The head is a normal *)
			(* We get the left most expression, and 
			 * if it is a symbol then we expand on 
			 * that, otherwise we do not match any
			 * macro
			 *)
				With[{
					firstHd = getFirstHead[mexpr]
				},
					addFixed@
					If[FailureQ[firstHd],
						(* No patterns would apply, since the first
						 * head is a literal
						 *)
						expandParts[env, mexpr, opts],
						If[env["hasRules", firstHd],
							With[{
								new = applyReplacements[env, env["getRules", firstHd], mexpr, opts]
							},
								If[MExprNormalQ[new] && testMacroFixed[new, mexpr],
									expandParts[env, new, opts],
									new
								]
							],
							expandParts[env, mexpr, opts]
						]
					]
				]
		]
	];
	
expand1[env_, arg_, opts_] :=
	ThrowException[{"Invalid macro expansion ", arg}]

	
applyReplacements[env_, replacements_, mexpr_, opts_] :=
	(
	replacements["reset"];
	With[{
		new = replacements["replace", mexpr, opts]
	},
		If[
			MExprSymbolQ[new["head"]] && 
				new["hasHead", Compile`Internal`MacroRecursionExpand] &&
				new["length"] === 1,
				expand1[env, new["part",1], opts],
				new]
	]
	)


SetAttributes[timeIt, HoldAllComplete]
timeIt[e_] := With[{t = AbsoluteTiming[e]}, Print[{Unevaluated[e], First[t]}]; Last[t]]
	
expandParts[env_, mexpr_, opts_] :=
	With[{
		hd = expand1[env, mexpr["head"], opts],
		args = expand1[env, #, opts]& /@ mexpr["arguments"]
	},
		Module[{newMExpr},
			newMExpr = CreateMExprNormal[hd, args];
			(* If `newMExpr` does NOT have a property that `mexpr` DOES have,
			   copy that from `mexpr` to `newMExpr` *)
			If[mexpr["properties"]["length"] > 0,
				Do[
					If[!newMExpr["hasProperty", prop],
						newMExpr["setProperty", prop -> mexpr["getProperty", prop]];
					],
					{prop, mexpr["properties"]["keys"]}
				]
			];
			newMExpr
		]
	]

testMacroFixed[new_, old_] := (
	If[new["hasProperty", "MacroExpandAgain"],
		Return[False]
	];
	If[new["getProperty", "MacroFixed", False],
		True,
		new["sameQ", old]
	]
)
	

testAddMacroFixed[ mexprN_, mexpr_] :=
(
	If[mexprN["hasProperty", "MacroExpandAgain"],
		mexprN["removeProperty", "MacroExpandAgain"];
		Return[mexprN]
	];
	If[mexprN["sameQ", mexpr],
		mexprN["setProperty", "MacroFixed" -> True]
	];
	mexprN
)	
	

getFirstHead[_?MExprLiteralQ] := 
	$Failed
getFirstHead[mexpr_?MExprSymbolQ] :=
	mexpr["symbol"]
getFirstHead[mexpr_?MExprNormalQ] :=
	getFirstHead[mexpr["head"]]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"MacroExpand",
	"Expands the MExpr AST based on the builtin macros. Unlike other passes, this pass returns a new AST."
];

MacroExpandPass = CreateMExprPass[<|
	"information" -> info,
	"runPass" -> macroExpand
|>];

RegisterPass[MacroExpandPass]
]]

End[]

EndPackage[]
