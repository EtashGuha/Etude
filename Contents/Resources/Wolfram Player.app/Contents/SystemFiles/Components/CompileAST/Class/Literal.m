BeginPackage["CompileAST`Class`Literal`"]

MExprLiteral;
MExprLiteralClass;
MExprLiteralQ;

Begin["`Private`"] 

Needs["CompileAST`Class`Base`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Create`State`"]


RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprLiteralClass = DeclareClass[
	MExprLiteral,
	<|
		(**< This is the best way to not generate accessors at this point *)
		"data" -> Function[{},
			Self["_data"]	
		],
		"length" -> Function[{},
			0
		],
		"sameQ" -> Function[{other0},
			(MExprLiteralQ[other0] && Self["id"] === other0["id"]) ||
			With[{other = CoerceMExpr[other0]},
				MExprLiteralQ[other] &&
				Self["_head"] === other["_head"] && 
				Self["data"] === other["data"]
			]
		],
		"clone" -> (clone[Self, ##]&),
		"accept" -> Function[{vst}, acceptLiteral[Self, vst]],
		"atomQ" -> Function[{}, True],
		"literalQ" -> Function[{}, True],
		"hasHead" -> Function[{val},
			Self["head"]["sameQ", val]
		],
		"toString" -> Function[{},
			toString[Self]
		],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"_data"
	},
	Extends -> {
		MExprClass
	},
    Predicate -> MExprLiteralQ
];
RegisterMExpr[ MExprLiteral];
]]


clone[self_] :=
	With[{cln = self["_clone"]},
		cln["setId", CreateMExprState[]["getId"]["increment"]];
		cln["set_data", self["_data"]];
		cln["setProperties", self["clonedProperties"]];
		cln
	]


(*
  TODO have errors for not finding a method
*)
acceptLiteral[obj_, vst_] :=
	Module[{data, head},
		data = obj["data"];
		head = Head[data];
		Switch[
			head,
			Symbol, If[ data === True || data === False,
				          vst["visitBoolean", obj], Null],
			Integer, vst["visitInteger", obj],
			Real, vst["visitReal", obj],
			String, vst["visitString", obj],
			_, Null
		]
	]
	

toString[mexpr_] :=
	With[{str = ToString[mexpr["data"], InputForm, PageWidth -> Infinity]},
		str
	]


(**************************************************)
(**************************************************)
(**************************************************)

icon := icon = Graphics[
	Text[
		Style["LIT\nEXPR", GrayLevel[0.7], Bold, 0.9*CurrentValue["FontCapHeight"] / AbsoluteCurrentValue[Magnification]]
	],
	$FormatingGraphicsOptions
];   


toBoxes[mexpr_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"MExprLiteral",
		mexpr,
  		icon,
	  	Join[
	  		{
	  			BoxForm`SummaryItem[{"id: ", mexpr["id"]}],
	  			BoxForm`SummaryItem[{"value: ", mexpr["toString"]}]
	  		},
	  		If[mexpr["type"] === Undefined,
	  			{},
	  			{
	  				BoxForm`SummaryItem[{"type: ",  mexpr["type"]["unresolve"]}]
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
