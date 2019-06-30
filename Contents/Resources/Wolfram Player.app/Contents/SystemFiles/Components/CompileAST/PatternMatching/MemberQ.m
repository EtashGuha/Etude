
BeginPackage["CompileAST`PatternMatching`MemberQ`"]

MExprMemberQ;


Begin["`Private`"] 

Needs["CompileAST`PatternMatching`Matcher`"]
Needs["CompileAST`Class`Symbol`"]


MExprMemberQ[expr_][lst_] :=
	MExprMemberQ[expr, lst]
MExprMemberQ[lst_, pat_] :=
	If[MExprSymbolQ[lst] || !lst["hasHead", List],
		False,
		AnyTrue[lst["arguments"], MExprMatchQ[pat]]
	]

End[]

EndPackage[]
