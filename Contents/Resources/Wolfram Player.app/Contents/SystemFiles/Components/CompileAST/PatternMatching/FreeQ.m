
BeginPackage["CompileAST`PatternMatching`FreeQ`"]

MExprFreeQ;


Begin["`Private`"] 


Needs["CompileAST`Utilities`Set`"]



(**< This is does not follow the semantics of FreeQ, but is good for now *)
MExprFreeQ[expr_][lst_] :=
	!MExprContainsQ[expr][lst]
MExprFreeQ[lst_, expr_] :=
	!MExprContainsQ[lst, expr]

End[]

EndPackage[]

