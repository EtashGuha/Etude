
BeginPackage["CompileAST`Create`Construct`"]


CreateMExpr;
CreateMExprLiteral;
CreateMExprNormal;
CreateMExprSymbol;

Begin["`Private`"] 


Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Symbol`"]
Needs["CompileAST`Create`State`"]
Needs["CompileAST`Class`Base`"]


ClearAll[oCreateMExpr];
SetAttributes[#, HoldAllComplete]& /@ {
	CreateMExpr,
	oCreateMExpr,
	CreateMExprLiteral,
	CreateMExprNormal,
	CreateMExprSymbol,
	createMExprSymbolImmutable
};

CreateMExpr[prog_] :=
	With[{st = CreateMExprState[]},
		oCreateMExpr[st, prog]
	]
	
CreateMExpr[hd_, args_] :=
	CreateMExprNormal[hd, args]

(* I think this is unused-remove? *)
oCreateMExpr[st_, hd_, args_] :=
	CreateMExprNormal[st, hd, args]
	
oCreateMExpr[st_MExprState, mexpr:(True | False)] :=
	CreateMExprLiteral[st, mexpr]
oCreateMExpr[st_MExprState, mexpr_Symbol] :=
	CreateMExprSymbol[st, mexpr]
	
oCreateMExpr[st_MExprState, mexpr:(f_[args___])] :=
	CreateMExprNormal[st, mexpr]
oCreateMExpr[st_MExprState, mexpr_] :=
	CreateMExprLiteral[st, mexpr]

(*
 TODO pattern excludes Complex, Rational
*)
CreateMExprNormal[expr_] :=
	With[{st = CreateMExprState[]},
		CreateMExprNormal[st, expr]
	]
CreateMExprNormal[hd_, args_] :=
	With[{st = CreateMExprState[]},
		CreateMExprNormal[st, hd, args]
	]
CreateMExprNormal[st_MExprState, hd_, args_] :=
	CreateObject[
		MExprNormal,
		<|
			"id" -> st["getId"]["increment"],
			"_head" -> CoerceMExpr[hd],
			"arguments" -> Apply[
				List,
				Map[
					Function[elem,
						If[MExprQ[elem],
							elem,
							oCreateMExpr[st, elem]
						],
						{HoldAllComplete}
					],
					Unevaluated[args]
				]
			]
		|>
	]
CreateMExprNormal[st_MExprState, expr:f_[args___]] :=
	Module[{id, argD, hd, mexpr},
		id = st["getId"]["increment"];
		hd = oCreateMExpr[st, f];
		argD = Map[
			Function[elem,
				oCreateMExpr[st, elem],
				{HoldAllComplete}
			],
			HoldComplete[args]
		];
		argD = List @@ argD;
		mexpr = CreateObject[
			MExprNormal,
			<|
				"id" -> id,
				"_head" -> hd,
				"arguments" -> argD
			|>
		];
		setSourceProperties[mexpr, expr];
		mexpr
	]
CreateMExprSymbol[] :=
	With[{st = CreateMExprState[], sym = Unique["sym", Temporary]},
		CreateMExprSymbol[st, sym]
	]
CreateMExprSymbol[expr_] :=
	With[{st = CreateMExprState[]},
		CreateMExprSymbol[st, expr]
	]
	
createMExprSymbolImmutable[st_MExprState, expr_Symbol] :=
	With[{name = SymbolName[Unevaluated[expr]]},
		If[!st["symbolCache"]["keyExistsQ", name],
			Module[{mexpr},
				mexpr = CreateObject[
					MExprSymbol,
					<|
						"id" -> st["getId"]["increment"],
						"data" -> HoldComplete[expr],
						"context" -> Context[expr],
						"name" -> SymbolName[Unevaluated[expr]], (**< this may not get rewritten *)
						"protected" -> True,
						"sourceName" -> SymbolName[Unevaluated[expr]] (**< name of the symbol as it appears in the source *)
					|>
				];
				st["symbolCache"]["associateTo", name -> mexpr];
			];
		];
		st["symbolCache"]["lookup", name]
	]
CreateMExprSymbol[st_MExprState, expr_Symbol] :=
	With[{ctx = Context[expr]},
		If[ctx === "System`" && MemberQ[Attributes[expr], Protected],
			createMExprSymbolImmutable[st, expr],
			Module[{mexpr},
				mexpr = CreateObject[
					MExprSymbol,
					<|
						"id" -> st["getId"]["increment"],
						"data" -> HoldComplete[expr],
						"context" -> Context[expr],
						"name" -> SymbolName[Unevaluated[expr]], (**< this may get rewritten to be unique for each binding environment *)
						"sourceName" -> SymbolName[Unevaluated[expr]] (**< name of the symbol as it appears in the source *)
					|>
				];
				mexpr
			]
		]
	]

 (* We want to create a unique literal, since it avoids cloning when
  * performing the type inference
  *)
$EnableLiteralCaching = False
CreateMExprLiteral[expr_] :=
	With[{st = CreateMExprState[]},
		CreateMExprLiteral[st, expr]
	]
CreateMExprLiteral[st_MExprState, expr_] := (
	If[$EnableLiteralCaching === False || !st["literalCache"]["keyExistsQ", expr],
		Module[{mexpr},
			mexpr = CreateObject[
				MExprLiteral,
				<|
					"id" -> st["getId"]["increment"],
					"_data" -> expr,
					"_head" -> Head[expr]
				|>
			];
			st["literalCache"]["associateTo", expr -> mexpr];
		];
	];
	st["literalCache"]["lookup", expr]
)

SetAttributes[setSourceProperties, {HoldRest}];
setSourceProperties[mexpr_?MExprQ, expr_] := Module[{tagData = RuntimeTools`ToTag[Unevaluated@expr]},
	If[tagData =!= 0,
		mexpr["setProperty", "debugTag" -> tagData[[1]]];
		mexpr["setProperty", "sourceLine" -> tagData[[2]]];
		mexpr["setProperty", "sourceFilePath" -> tagData[[4]]];
	];
];


	

End[]

EndPackage[]
