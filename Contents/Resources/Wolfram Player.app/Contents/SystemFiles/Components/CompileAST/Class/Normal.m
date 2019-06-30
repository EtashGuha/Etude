
BeginPackage["CompileAST`Class`Normal`"]

MExprNormal;
MExprNormalClass;
MExprNormalQ;


Begin["`Private`"] 

Needs["CompileAST`Class`Base`"]
Needs["CompileAST`Language`ShortNames`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Create`State`"]


SymbolQ = Developer`SymbolQ

sameQ[self_, other0_] :=
	(MExprNormalQ[other0] && (
			self["id"] === other0["id"] ||
			Internal`HashSameQ[self, other0]
		)
	) ||
	With[{other = CoerceMExpr[other0]},
		MExprNormalQ[other] &&
		self["length"] === other["length"] &&
		self["head"]["sameQ", other["head"]] &&
		TrueQ[
			AllTrue[
				Transpose[{self["arguments"], other["arguments"]}],
				#[[1]]["sameQ", #[[2]]]&
			]
		]
	]

RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprNormalClass = DeclareClass[
	MExprNormal,
	<|
		"normalQ" -> Function[{}, True],
		"length" -> Function[{}, Length[Self["arguments"]]],
		"part" -> Function[{idx},
			Which[
				idx < 0,
					Null,
				idx === 0,
					Self["head"],
				idx <= Self["length"],
					Self["arguments"][[idx]],
				True,
					Null
			]
		],
		"prependArgument" -> Function[{elem},
			Self["setArguments", Prepend[Self["arguments"], elem]];
			Self
		],
		"appendArgument" -> Function[{elem},
			Self["setArguments", Append[Self["arguments"], elem]];
			Self
		],
		"setArguments" -> Function[{val},
			SetData[Self["arguments"], val];
			Self
		],
		"setPart" -> (setPart[Self, ##]&),
		"hasHead" -> Function[{h},
			Self["head"]["sameQ", h]
		],
		"isList" -> Function[{},
			Self["hasHead", List]
		],
		"sameQ" -> (sameQ[Self, ##]&),
		"clone" -> (clone[Self, ##]&),
		"toString" -> Function[{},
			toString[Self]
		],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		],
		"accept" -> Function[{vst}, acceptNormal[Self, vst]]
	|>,
	{
		"arguments"
	},
	Extends -> {
		MExprClass
	},
    Predicate -> MExprNormalQ
];
RegisterMExpr[ MExprNormal];
]]

setPart[self_, idx_?IntegerQ -> val_] := 
	setPart[self, idx, val]
setPart[self_, idx_?IntegerQ, val_] := (
	Which[
		idx < 0,
			self,
		idx === 0,
			self["setHead", val],
		idx <= self["length"],
			self["setArguments",
				ReplacePart[self["arguments"], idx -> CoerceMExpr[val]]
			],
		True,
			self
	];
	self
)


acceptNormal[obj_, vst_] :=
	Module[{},
		If[ !vst["visitNormal", obj],
			Return[]
		];
		If[TrueQ[vst["getVisitHeadsQ"]],
		  obj["head"]["accept", vst]
		];
		Scan[ #["accept",vst]&, obj["arguments"]]
	]

clone[mexpr_, ___] :=
	clone[mexpr]

clone[mexpr_] :=
	With[{
		cln = mexpr["_clone"]
	},
		cln["setId", CreateMExprState[]["getId"]["increment"]];
		cln["setProperties", mexpr["clonedProperties"]];
		cln["setHead", mexpr["head"]["clone"]];
		cln["setArguments", #["clone"]& /@ mexpr["arguments"]];
		cln
	]

toString[mexpr_] := StringJoin[
	With[{hd = mexpr["getHead"],
		  args = Map[#["toString"]&, mexpr["arguments"]]
		 },
		Which[
			SymbolQ[hd] && hd === List,
				{"{", Riffle[args, ","], "}"},
			SymbolQ[hd] && hd === Pattern && (*< Capture the pattern a_Integer or Pattern[a, Blank[...]] *)
			Length[args] === 2 && mexpr["part", 2]["head"]["sameQ", Blank],
				args,
			SymbolQ[hd] && hd === Blank,
				Switch[Length[args],
					0,
						"_",
					1,
						{"_", args},
					_,
						Riffle[args, "_"]
				],
			SymbolQ[hd] && hd === CompoundExpression,
				Riffle[args, ";\n"],
			SymbolQ[hd] && KeyExistsQ[$SystemShortNames, hd] && mexpr["length"] > 1,
				Riffle[args, " " <> $SystemShortNames[hd] <> " "],
			SymbolQ[hd] && hd === Part,
				{
					First[args],
					"\[LeftDoubleBracket]",
					Riffle[Rest[args], ", "],
					"\[RightDoubleBracket]"
				},
			SymbolQ[hd] && hd === Association,
				{
					"\[LeftAssociation] ", 
					Riffle[args, ", "],
					" \[RightAssociation]"
				},
			True, (* FullForm *)
				{
					mexpr["head"]["toString"],
					"[",
					Riffle[args, ", "],
					"]"
				}
		]
	]
]
	
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := icon = Graphics[
	Text[
		Style["NRM\nEXPR", GrayLevel[0.7], Bold, 0.9*CurrentValue["FontCapHeight"] / AbsoluteCurrentValue[Magnification]]
	],
	$FormatingGraphicsOptions
];   


toBoxes[mexpr_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"MExprNormal",
		mexpr,
  		icon,
	  	Join[
	  		{
	  			BoxForm`SummaryItem[{"id: ", mexpr["id"]}],
	  			With[{str = mexpr["toString"]},
	  				BoxForm`SummaryItem[{"value: ", If[StringLength[str] > 100, "\n" <> str, str]}]
	  			]
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
