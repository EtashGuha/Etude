BeginPackage["CompileAST`Class`MExprAtomQ`"]

MExprAtomQ;

Begin["`Private`"] 

Needs["CompileAST`Class`Symbol`"]
Needs["CompileAST`Class`Literal`"]

MExprAtomQ[mexpr___] := MExprSymbolQ[mexpr] || MExprLiteralQ[mexpr]

End[]

EndPackage[]
