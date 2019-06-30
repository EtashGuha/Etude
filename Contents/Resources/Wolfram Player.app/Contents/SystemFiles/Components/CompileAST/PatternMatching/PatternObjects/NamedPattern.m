
BeginPackage["CompileAST`PatternMatching`PatternObjects`NamedPattern`"]

NamedPattern

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
NamedPatternClass = DeclareClass[
	NamedPattern,
	<|
		"matchPattern" -> Function[{mexpr}, matchMExpr[Self, mexpr]],
		"initialize" -> Function[{}, initialize[Self]]
	|>,
	{
	   "matcher",
	   "pattern",
	   "name",
	   "index",
	   "nestedPattern"
	},
    Predicate -> NamedPatternQ
]
]]

initialize[ self_] :=
	Module[ {matcher, pattern, name, nestedPattern, index},
		matcher = self["matcher"];
		pattern = self["pattern"];
		If[ pattern["length"] =!= 2, 
			ThrowException[{"NamedPattern Length", pattern["length"]}]
		];
		name = pattern["part",1];
		nestedPattern = pattern["part",2];
		self["setName", name];
		self["setNestedPattern", nestedPattern];
		index = matcher["getPatternIndex", name];
		self["setIndex", index];
		matcher["processPattern", nestedPattern];
	]

matchMExpr[ self_, mexpr_] :=
	Module[ {ef},
		ef = self["matcher"]["matchRecursive",  mexpr, self["nestedPattern"]];
		If[ ef === Null,
			Null,
			self["matcher"]["setBinding", self["index"], ef]
		]
	]


End[]

EndPackage[]

