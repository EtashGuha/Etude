
BeginPackage["CompileAST`PatternMatching`PatternObjects`Verbatim`"]

VerbatimPattern

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
VerbatimClass = DeclareClass[
	VerbatimPattern,
	<|
		"matchPattern" -> Function[{mexpr}, matchMExpr[Self, mexpr]],
		"initialize" -> Function[{}, initialize[Self]]
	|>,
	{
	   "matcher",
	   "data",
	   "pattern"
	},
    Predicate -> VerbatimPatternQ
]
]]

initialize[ self_] :=
	Module[ {len, pattern},
		pattern = self["pattern"];
		len = pattern["length"];
		If[ len =!= 1, 
			ThrowException[{"Verbatim Length", len}]
		];
		self["setData", pattern["part",1]];
	]

matchMExpr[ self_, mexpr_] :=
	If[self["data"]["sameQ", mexpr],
		mexpr,
		Null
	]


End[]

EndPackage[]

