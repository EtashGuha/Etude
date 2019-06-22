BeginPackage["CompileAST`Export`ShowMExpr`"]

ShowMExpr;

Begin["`Private`"] 

Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Symbol`"]
Needs["CompileAST`Class`Base`"]
Needs["CompileUtilities`Format`"]


Options[ShowMExpr] = {"ShowProperties" -> False}
Options[iShowExpr] = {"ShowProperties" -> False}
Options[tooltipData] = {"ShowProperties" -> False}

ShowMExpr[mexpr_, opts:OptionsPattern[]] :=
	Module[ {res, st},
		res = iShowMExpr[st, mexpr, opts];
		RawBoxes[res]
	]

iShowMExpr[st_, mexpr_?MExprLiteralQ, opts:OptionsPattern[]] :=
	With[{box = RawBoxes[mexpr["data"]]},
		ToBoxes[
			Tooltip[
				Mouseover[
					Framed[
						box,
						(* This avoids jitter on mouse over *)
						FrameStyle -> None
					],
					Framed[
						box,
						Background -> LightYellow
					], 
					ImageSize -> All,
					Alignment -> Center,
					FrameMargins -> 1
				],
				tooltipData[st, mexpr, opts]
			]
		]
	]
iShowMExpr[st_, mexpr_?MExprSymbolQ, opts:OptionsPattern[]] :=
	Module[{context, name},
        context = mexpr["context"];
		name = If[ MemberQ[ $ContextPath, context],
        	"",
        	context
      	];
      	name = StringJoin[name, mexpr["name"]];
		With[{box = RawBoxes[name]},
			ToBoxes[
				Tooltip[
					Mouseover[
						Framed[
							box,
							(* This avoids jitter on mouse over *)
							FrameStyle -> None
						],
						Framed[
							box,
							Background -> LightYellow
						], 
						ImageSize -> All,
						Alignment -> Center,
						FrameMargins -> 1
					],
					tooltipData[st, mexpr, opts]
				]
			]
		]
	]
iShowMExpr[st_, mexpr_?MExprNormalQ, opts:OptionsPattern[]] :=
	Module[{head, args},
		head = iShowMExpr[st, mexpr["head"], opts];
		args = iShowMExpr[st, #, opts]& /@ mexpr["arguments"];
		RowBox[Flatten[{
			head,
			"[",
			Riffle[args, ", "],
			"]"
		}]]
	]
    


tooltipData[st_, mexpr_?MExprLiteralQ, OptionsPattern[]] :=
	CompileInformationPanel[
		"Literal",
		{
			"ExprType" -> "Literal",
			"Id" -> mexpr["id"],
			"Data" -> mexpr["data"],
			"Type" -> If[mexpr["type"] === Undefined,
				Nothing,
				mexpr["type"]
			]
		}
	]
tooltipData[st_, mexpr_?MExprSymbolQ, OptionsPattern[]] :=
	With[{context = mexpr["context"]},
		CompileInformationPanel[
			"Symbol",
			{
				"ExprType" -> "Symbol",
				"Id" -> mexpr["id"],
				"FullName" -> StringJoin[
					If[MemberQ[$ContextPath, context], "", context],
					mexpr["name"]
				],
				"Type" -> If[mexpr["type"] === Undefined,
					Nothing,
					mexpr["type"]
				],
				"Context" -> context,
				"SourceName" -> mexpr["sourceName"],
				"ProtectedQ" -> mexpr["protected"],
				Sequence @@ Table[
					If[MExprQ[mexpr["getProperty", prop]],
					   Nothing, (* Avoid recursion *)
					   camelCase[prop] -> mexpr["getProperty", prop]
					],
					{prop, mexpr["properties"]["keys"]}
				]
			}
		]
	]

camelCase[str_String] :=
	ToUpperCase[StringTake[str, 1]] <> StringDrop[str, 1]

End[]

EndPackage[]
