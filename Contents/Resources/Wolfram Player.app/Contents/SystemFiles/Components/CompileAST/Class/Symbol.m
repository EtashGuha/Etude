
BeginPackage["CompileAST`Class`Symbol`"]

MExprSymbol;
MExprSymbolClass;
MExprSymbolQ;


Begin["`Private`"] 

Needs["CompileAST`Class`Base`"]
Needs["CompileAST`Create`State`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Expect`"]
Needs["CompileAST`Class`Operators`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprSymbolClass = DeclareClass[
	MExprSymbol,
	<|
		"setName" -> Function[{name},
			ExpectThat["The symbol is not protected.", Self["protected"]
				]["named", Self["toString"]
				]["isFalse"];
			SetData[Self["name"], name];
			Self
		],
		"length" -> Function[{},
			0
		],
		"sameQ" -> Function[{other0},
			(MExprSymbolQ[other0] && Self["id"] === other0["id"]) ||
			With[{other = CoerceMExpr[other0]},
				MExprSymbolQ[other] &&
				Self["data"] === other["data"] &&
				Self["name"] === other["name"]
			]
		],
		"atomQ" -> Function[{}, True],
		"symbolQ" -> Function[{}, True],
		"fullName" -> Function[{},
			Self["context"] <> Self["sourceName"]
		],
		"lexicalName" -> Function[{},
			Self["context"] <> Self["name"]
		],
		"hasHead" -> Function[{val},
			val === Symbol ||
			If[MExprSymbolQ[val],
				val["symbol"] === Symbol,
				False
			]
		],
		"symbol" -> Function[{},
			ReleaseHold[Self["data"]]
		],
		"clone" -> (clone[Self, ##]&),
		"accept" -> Function[{vst}, acceptSymbol[Self, vst]],
		"isOperator" -> Function[{},
			isOperator[Self]
		],
		"toString" -> Function[{},
			toString[Self]
		],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"data", 
		"context",
		"_head" -> Symbol,
		"name", (**< this may get rewritten to be unique for each binding environment unless protected is true *)
		"protected" -> False,
		"sourceName" (**< name of the symbol as it appears in the source *)
	},
	Extends -> {
		MExprClass
	},
    Predicate -> MExprSymbolQ
];
RegisterMExpr[ MExprSymbol];
]]

acceptSymbol[obj_, vst_] :=
	vst["visitSymbol", obj]


clone[self_] :=
	With[{
		cln = self["_clone"]
	},
		cln["setId", CreateMExprState[]["getId"]["increment"]];
		cln["setProtected", False];
		cln["setProperties", self["clonedProperties"]];
		cln["setData", self["data"]];
		cln["setContext", self["context"]];
		cln["set_head", self["_head"]];
		cln["setName", self["name"]];
		cln["setSourceName", self["sourceName"]];
		cln
	]

clone[self_, env_] :=
	clone[self]

isOperator[self_] :=
	MemberQ[$ExprOperators, self["name"]]
	
toString[mexpr_] :=
	Module[{context, name},
        context = mexpr["context"];
		name = If[ MemberQ[ $ContextPath, context],
        	"",
        	context
      	];
      	StringJoin[name, mexpr["name"]]
	]
	
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := icon = Graphics[
	Text[
		Style["SYM\nEXPR", GrayLevel[0.7], Bold, 0.9*CurrentValue["FontCapHeight"] / AbsoluteCurrentValue[Magnification]]
	],
	$FormatingGraphicsOptions
];   


toBoxes[mexpr_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"MExprSymbol",
		mexpr,
  		icon,
	  	Join[
	  		{
	  			BoxForm`SummaryItem[{"id: ", mexpr["id"]}],
	  			BoxForm`SummaryItem[{"name: ", mexpr["toString"]}]
	  		},
	  		If[mexpr["type"] === Undefined,
	  			{},
	  			{
	  				BoxForm`SummaryItem[{"type: ",  mexpr["type"]["unresolve"]}]
	  			}
	  		],
	  		If[mexpr["protected"] === True,
	  			{
	  				BoxForm`SummaryItem[{"protected: ",  True}]
	  			},
	  			{
	  				BoxForm`SummaryItem[{"sourceName: ", mexpr["sourceName"]}]
	  			}
	  		],
	  		If[mexpr["span"] === Undefined,
	  			{},
	  			{
	  				BoxForm`SummaryItem[{"span: ", mexpr["span"]["toString"]}]
	  			}
	  		]
	  	],
		{}, 
  		fmt
  	]

End[]

EndPackage[]
