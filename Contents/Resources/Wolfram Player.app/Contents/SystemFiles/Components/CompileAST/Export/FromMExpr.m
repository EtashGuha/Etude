BeginPackage["CompileAST`Export`FromMExpr`"]

FromMExpr;

FindMExpr;

Begin["`Private`"] 


Needs["CompileAST`Class`Base`"]
Needs["CompileAST`Class`MExprAtomQ`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Symbol`"]



FromMExpr[mexpr_] :=
	fromMExpr[mexpr]

	
fromMExpr[mexpr_?MExprNormalQ] :=
	With[{
		args = fromMExpr /@ mexpr["arguments"],
		hd = fromMExpr[mexpr["head"]]
	},
		With[{e = hd @@ args},
			With[{ef = HoldComplete[e]},
				Delete[
					ef,
					Table[{1,i,0}, {i,0,Length[e]}]
				]
			]
		]
	]
fromMExpr[mexpr_?MExprSymbolQ] :=
	ToExpression[mexpr["context"] <> mexpr["name"], InputForm, HoldComplete]

fromMExpr[mexpr_?MExprLiteralQ] :=
	With[{a = mexpr["data"]},
		HoldComplete[a]
	]


FindMExpr[ mexpr_, id_] :=
	find[mexpr, id]
find[mexpr_?MExprAtomQ, id_] :=
	If[mexpr["id"] === id,
		mexpr,
		$Failed
	]
find[mexpr_?MExprNormalQ, id_] :=
	Module[{res},
		Which[
			mexpr["id"] === id,
				mexpr,
			MExprQ[res = find[mexpr["head"], id]],
				res,
			True,
				Module[{ii = 1},
					While[ii <= mexpr["length"] && !MExprQ[res],
						res = find[mexpr, mexpr["part", ii]];
						ii++
					];
					res
				]
		]
	]
		

	
End[]

EndPackage[]
