

BeginPackage["CompileAST`Utilities`Set`"]

MExprContainsQ;
MExprComplement;
MExprUnion;

Begin["`Private`"]

sameQ[a_, b_] := a["sameQ", b];

    
sameNameQ[a_, b_] :=
    (a["id"] === b["id"]) || (
    a["symbolQ"] && b["symbolQ"] &&
    a["fullName"] === b["fullName"] &&
    a["name"] == b["name"])
    
MExprContainsQ[lst_, elem_] :=
    MExprContainsQ[lst, elem, SameTest -> sameQ];
MExprContainsQ[lst_, elem_, SameTest -> sameTest_] :=
	TrueQ[AnyTrue[lst, sameTest]]


MExprComplement[all_, nil_] :=
    MExprComplement[all, nil, SameTest -> sameQ];
MExprComplement[all_, nil_, SameTest -> sameTest_] :=
	Select[all,
		Function[{elem},
			NoneTrue[nil, sameTest]
		]
	]
	
MExprUnion[a_, b_] :=
    MExprUnion[a, b, SameTest -> sameQ];
MExprUnion[a_, b_, SameTest -> sameTest_] :=
	Join[
		a,
		MExprComplement[b, a, SameTest -> sameTest]
	]
	
End[]

EndPackage[]