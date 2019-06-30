
BeginPackage["CompileAST`PatternMatching`PatternObjects`ConditionalPattern`"]

ConditionalPattern
IsConditionTrue

Begin["`Private`"] 

Needs["CompileAST`Export`FromMExpr`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
ConditionalPatternClass = DeclareClass[
	ConditionalPattern,
	<|
		"matchPattern" -> Function[{mexpr}, matchMExpr[Self, mexpr]],
		"initialize" -> Function[{}, initialize[Self]]
	|>,
	{
	   "matcher",
	   "pattern",
	   "nestedPattern",
	   "condition"	
	},
    Predicate -> ConditionalPatternQ
]
]]

initialize[ self_] := 
	With[{
		matcher = self["matcher"],
		pattern = self["pattern"]
	},
		self["setNestedPattern",
			matcher["processPattern", pattern["part", 1]]
		];
		self["setCondition", pattern["part", 2]]
	]

matchMExpr[ self_, mexpr_] :=
	With[{
		f = ReleaseHold[
			FromMExpr[self["condition"]]
		]
	},
	With[{
		app = Apply[f, {mexpr}]
	},
		Which[
			IsConditionTrue[app],
				self["matcher"]["matchRecursive",  mexpr, self["nestedPattern"]],
			True,
				Null
		]
	]]

IsConditionTrue[app_] :=
	TrueQ[app] || (MExprLiteralQ[app] && app["sameQ", True])

End[]

EndPackage[]
