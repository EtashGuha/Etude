BeginPackage["TypeFramework`Utilities`AlphaEquivalentQ`"]

AlphaEquivalentQ

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]


AlphaEquivalentQ[ty1_?TypeObjectQ, ty2_?TypeObjectQ] :=
	alphaEquivalentQ[{ty1, {}}, {ty2, {}}];

alphaEquivalentQ[{ty1_?TypeForAllQ, bvs1_List}, {ty2_?TypeForAllQ, bvs2_List}] :=
	alphaEquivalentQ[
		{ty1["type"], Append[bvs1, ty1["variables"]]},
		{ty2["type"], Append[bvs2, ty2["variables"]]}
	];

alphaEquivalentQ[{ty1_?TypeQualifiedQ, bvs1_List}, {ty2_?TypeQualifiedQ, bvs2_List}] :=
	alphaEquivalentQ[{ty1["predicates"], bvs1}, {ty2["predicates"], bvs2}] &&
	alphaEquivalentQ[{ty1["type"], bvs1}, {ty2["type"], bvs2}];

alphaEquivalentQ[{ty1_?TypePredicateQ, bvs1_List}, {ty2_?TypePredicateQ, bvs2_List}] :=
	ty1["test"] == ty2["test"] &&
	alphaEquivalentQ[{ty1["types"], bvs1}, {ty2["types"], bvs2}]

alphaEquivalentQ[{ty1_?TypeArrowQ, bvs1_List}, {ty2_?TypeArrowQ, bvs2_List}] :=
	alphaEquivalentQ[{ty1["arguments"], bvs1}, {ty2["arguments"], bvs2}] &&
	alphaEquivalentQ[{ty1["result"], bvs1}, {ty2["result"], bvs2}];

alphaEquivalentQ[{ty1_?TypeApplicationQ, bvs1_List}, {ty2_?TypeApplicationQ, bvs2_List}] :=
	alphaEquivalentQ[{ty1["type"], bvs1}, {ty2["type"], bvs2}] &&
	alphaEquivalentQ[{ty1["arguments"], bvs1}, {ty2["arguments"], bvs2}];

alphaEquivalentQ[{ty1_?TypeVariableQ, bvs1_List}, {ty2_?TypeVariableQ, bvs2_List}] :=
	With[{
		indices1 = Position[bvs1, _?(ty1["sameQ", #]&)],
		indices2 = Position[bvs2, _?(ty2["sameQ", #]&)]
	},
		Which[
			Length[indices1] == 0 && Length[indices2] == 0, ty1["sameQ", ty2],
			Length[indices1] >  0 && Length[indices2] >  0, Last[indices1] == Last[indices2],
			True, False
		]
	];

alphaEquivalentQ[{ty1_?TypeLiteralQ, bvs1_List}, {ty2_?TypeLiteralQ, bvs2_List}] :=
    ty1["value"] === ty2["value"] &&
    alphaEquivalentQ[{ty1["type"], bvs1}, {ty2["type"], bvs2}];
    
alphaEquivalentQ[{ty1_?TypeConstructorQ, bvs1_List}, {ty2_?TypeConstructorQ, bvs2_List}] :=
	ty1["sameQ", ty2];

alphaEquivalentQ[{tys1_List, bvs1_List}, {tys2_List, bvs2_List}] :=
	And@@MapThread[mkAlphaEquivalentQ[bvs1, bvs2], {tys1, tys2}];

alphaEquivalentQ[___] := False

mkAlphaEquivalentQ[bvs1_List, bvs2_List] := alphaEquivalentQ[{#1, bvs1}, {#2, bvs2}]&;

End[]

EndPackage[]