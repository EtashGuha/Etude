
BeginPackage["CompileAST`PatternMatching`PatternObjects`SingleBlank`"]

SingleBlank

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
SingleBlankClass = DeclareClass[
	SingleBlank,
	<|
		"matchPattern" -> Function[{mexpr}, matchMExpr[Self, mexpr]],
		"initialize" -> Function[{}, initialize[Self]]
	|>,
	{
	   "matcher",
	   "pattern",
	   "head"	
	},
    Predicate -> SingleBlankQ
]
]]

initialize[ self_] :=
	Module[ {len, pattern, head},
		pattern = self["pattern"];
		len = pattern["length"];
		If[ len > 1, 
			ThrowException[{"SingleBlank Length", len}]
		];
		head = If[len === 1,
			pattern["part", 1],
			Null
		];
		self["setHead", head]
	]

matchMExpr[ self_, mexpr_] :=
	Module[ {head},
		head = self["head"];
		If[ head === Null, 
			mexpr,
			If[head["sameQ", mexpr["head"]],
				mexpr,
				Null
			]
		]
	]


End[]

EndPackage[]

