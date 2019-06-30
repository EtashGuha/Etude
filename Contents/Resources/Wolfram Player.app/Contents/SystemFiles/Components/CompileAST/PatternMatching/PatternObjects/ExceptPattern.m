
BeginPackage["CompileAST`PatternMatching`PatternObjects`ExceptPattern`"]

ExceptPattern
ExceptPatternQ

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
ExceptPatternClass = DeclareClass[
	ExceptPattern,
	<|
		"matchPattern" -> Function[{mexpr}, matchMExpr[Self, mexpr]],
		"initialize" -> Function[{}, initialize[Self]]
	|>,
	{
	   "matcher",
	   "pattern",
	   "exceptionPattern",
	   "nestedPattern"
	},
    Predicate -> ExceptPatternQ
]
]]

initialize[ self_] := 
	With[{
		matcher = self["matcher"],
		pattern = self["pattern"]
	},
		If[pattern["length"] ==  2,
			self["setNestedPattern",
				matcher["processPattern", pattern["part", 2]]
			],
			self["setNestedPattern",
				Null
			]
		];
		self["setExceptionPattern",
		   	matcher["processPattern", pattern["part", 1]]
		]
	]

matchMExpr[ self_, mexpr_] :=
	Module[ {matcher, nestedPattern, exceptionPattern, ef},
	    matcher = self["matcher"];
	    nestedPattern = self["nestedPattern"];
	    If[nestedPattern =!= Null,
			ef = matcher["matchRecursive",  mexpr, nestedPattern];
			If[ ef === Null,
				Return[ Null]
			]
	    ];
	    exceptionPattern = self["exceptionPattern"];
	    ef = matcher["matchRecursive",  mexpr, exceptionPattern];
		If[ ef =!= Null,
			Return[ Null]
		];
		mexpr
	]


End[]

EndPackage[]

