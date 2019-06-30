BeginPackage["Compile`AST`Transform`Replace`"]


Begin["`Private`"] 

Needs["CompileAST`Class`Base`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Symbol`"]



MExprReplace[mexpr_, rhs0_ -> lhs0_] :=
	With[
		{
			rhs = CoerceMExpr[rhs0],
			lhs = CoerceMExpr[lhs0]
		},
		repl[mexpr, rhs -> lhs]
	]
	
repl[mexpr_?MExprNormalQ, rhs_ -> lhs_] :=
	With[{new = mexpr["clone"]},
		new["setHead", repl[mexpr["head"], rhs -> lhs]];
		new["setElements", repl[#, rhs -> lhs]& /@ mexpr["elements"]];
		new
	]
	
repl[mexpr_?MExprSymbolQ, rhs_?MExprSymbolQ -> lhs_] :=
	If[mexpr["sameQ", rhs],
		lhs["clone"],
		mexpr
	]
repl[mexpr_?MExprLiteralQ, rhs_?MExprLiteralQ -> lhs_] :=
	If[mexpr["sameQ", rhs],
		lhs["clone"],
		mexpr
	]
repl[mexpr_, _] := mexpr
	

End[]

EndPackage[]
