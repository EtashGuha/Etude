
BeginPackage["CompileAST`PatternMatching`Replace`"]

MExprReplace
MExprReplaceAll
CreateMExprReplacer
MExprReplacer


Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileAST`PatternMatching`Matcher`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`Base`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`PatternMatching`ComparePattern`"] (* for MExprSortPatterns *)


RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprReplacerClass = DeclareClass[
	MExprReplacer,
	<|
		"replace" -> (replace[Self, ##]&),
		"replaceAll" -> (replaceAll[Self, ##]&),
		"reset" -> (resetReplacer[Self]&),
		"join" -> (join[Self, ##]&),
		"toString" -> (toString[Self, ##]&),
		"toBoxes" -> (toBoxes[Self, ##]&)
	|>,
	{
		"rules",
		"matchData"
	},
    Predicate -> MExprReplacerQ
]
]]


toString[self_] := StringJoin[
	"MExprReplacer[<<<",
	   ToString[Length[self["rules"]]],
	">>]"
]


icon := icon = Graphics[
	Text[
		Style["Replacer", GrayLevel[0.7], Bold, 0.9*CurrentValue["FontCapHeight"] / AbsoluteCurrentValue[Magnification]]
	],
	$FormatingGraphicsOptions
];   


toBoxes[self_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"MExprReplacer",
		self,
  		icon,
		{
			BoxForm`SummaryItem[{"rules: ", self["rules"]}],
			BoxForm`SummaryItem[{"matchData: ",
				Pane[
					Grid[
						self["matchData"],
						Frame -> All,
						Alignment -> {{Left, Decimal}, {Bottom}}
					]
				]
			}]
		}
	  	,
		{}, 
  		fmt
  	]



initializeReplacer[ self_] :=
	Module[ {rules, matchData},
		rules = self["rules"];
		If[!rules["isList"],
			rules = CreateMExprNormal[ CreateMExpr[List], {rules}]
		];
		rules = MExprSortPatterns[rules["arguments"]];
		matchData = Map[ createMatchers, rules];
		self["setMatchData", matchData];
	]

join[self_, other_?MExprReplacerQ] := (
	self["setMatchData", 
		Join[
			self["getMatchData"],
			other["getMatchData"]
		]
	];
	self
)
	
join[self_, args___] :=
	ThrowException[{"Replace: cannot join ", self, " with ", {args}}]

createMatchers[rule_] :=
	Module[ {lhs, rhs, matcher},
		If[ !isRule[rule], 
			ThrowException[{"Replace: rule not found: ", rule["toString"]}]
		];
		lhs = rule["part",1];
		rhs = rule["part",2];
		
		
		lhs["setProperties", lhs["properties"]["join", rule["properties"]]];
		rhs["setProperties", rhs["properties"]["join", rule["properties"]]];
		
		matcher = CreateMExprMatcher[lhs];
		{matcher, rhs}
	]
	
resetReplacer[ self_] :=
	(Scan[ #["reset"]&, self["matchData"]];self)

isRule[mexpr_] :=
	mexpr["length"] === 2 && (mexpr["hasHead", Rule] || mexpr["hasHead", RuleDelayed])

replace[self_, mexpr_, opts_:<||>] :=
	Module[ {matcher, rhs, ef = Null},
		If[ListQ[self["matchData"]],
			Do[
				{matcher, rhs} = data;
				matcher["reset"];
				If[matchConstraintQ[matcher, Append[opts, "MetaData" -> mexpr["getProperties"]]],
					ef = matcher["matchSingle", mexpr];
					If[ef =!= Null,
						ef = matcher["bindings"]["substitute", rhs];
						Return[ef]
					]
				]
				,
				{data, self["matchData"]}
			]
		];
		If[ ef === Null,
			ef = mexpr
		];
		ef
	]

matchConstraintQ[matcher_, opts_] := 
	If[matcher["pattern"]["hasProperty", "Constraint"] === False,
		True,
		With[{
			constraints = matcher["pattern"]["getProperty", "Constraint"]
		},
			TrueQ[constraints[opts]]
		]
	];

(*
 ReplaceAll
*)

RegisterCallback["DeclareCompileASTClass", Function[{st},
DeclareClass[
	ReplaceAllVisitor,
	<|
		"visitSymbol" -> Function[{mexpr}, replaceMExpr[Self, mexpr]],
		"visitInteger" -> Function[{mexpr}, replaceMExpr[Self, mexpr]],
		"visitString" -> Function[{mexpr}, replaceMExpr[Self, mexpr]],
		"visitReal" -> Function[{mexpr}, replaceMExpr[Self, mexpr]],
		"visitBoolean" -> Function[{mexpr}, replaceMExpr[Self, mexpr]],
		"visitNormal" -> Function[{mexpr}, replaceMExprNormal[Self, mexpr]]
	|>,
	{
  		"replacer",
  		"opts"
  	},
	Extends -> {
		MExprMapVisitorClass
	}
]
]]


sameInstanceQ[e1_, e2_] :=
	e1["id"] === e2["id"]

replaceMExprNormal[ self_, mexpr_] :=
	Module[ {ef},
		ef = self["replacer"]["replace", mexpr, self["opts"]];
		self["setResult", ef];
		If[sameInstanceQ[ef, mexpr],
			self["processNormal", mexpr]
		];
		False
	]

replaceMExpr[ self_, mexpr_] :=
	Module[ {ef},
		ef = self["replacer"]["replace", mexpr, self["opts"]];
		self["setResult", ef]
	]



replaceAll[self_, mexpr_, opts_:<||>] :=
	Module[ {vst},
		vst = CreateObject[ ReplaceAllVisitor, <|"replacer" -> self, "opts" -> opts|>];
		mexpr["accept", vst];
		vst["result"]
	];

CreateMExprReplacer[rules_] :=
	CatchException[
		With[{
			obj = CreateObject[MExprReplacer, <|"rules" -> rules|>]
		},
			initializeReplacer[obj];
			obj
		]
		,
		{{_, CreateFailure}}
	]

ClearAll[MExprReplace]
MExprReplace[ mexpr_, pat_ -> trgt_] :=
	With[{hd = CreateMExprSymbol[Rule], args = {pat, trgt}},
		MExprReplace[mexpr, CreateMExprNormal[hd, args]]
	]
MExprReplace[ mexpr_, rules_?ListQ] :=
	With[{hd = CreateMExprSymbol[List], args = rules},
		MExprReplace[mexpr, CreateMExprNormal[hd, args]]
	]
MExprReplace[ mexpr_, rules0_?MExprQ] :=
	Module[{replacer, rules},
		CatchException[
			rules = If[rules0["isList"],
				rules0,
				With[{hd = CreateMExprSymbol[List], args = {rules0}},
					CreateMExprNormal[hd, args]
				]
			];
			replacer = CreateMExprReplacer[rules];
			replacer["replace", mexpr]
			,
			{{_, CreateFailure}}
		]
	];
	
MExprReplaceAll[ mexpr_, rules_] :=
	Module[ {replacer},
		CatchException[
			replacer = CreateMExprReplacer[rules];
			replacer["replaceAll", mexpr]
			,
			{{_, CreateFailure}}
		]
	];

End[]

EndPackage[] 
