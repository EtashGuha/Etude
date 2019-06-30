BeginPackage["GraphStore`SPARQL`JoinAcross`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`ArrayAssociation`"];

Begin["`Private`"];

SPARQLJoinAcross[OrderlessPatternSequence[{}, _]] := {};
SPARQLJoinAcross[OrderlessPatternSequence[{<||>}, sol_]] := sol;
SPARQLJoinAcross[l1_List, l2_List] := If[SameQ @@ Keys[l1] && SameQ @@ Keys[l2],
	With[
		{i = Intersection[First[Keys[l1]], First[Keys[l2]]]},
		If[i === {},
			Flatten[Outer[Join, l1, l2], 1],
			JoinAcross[l1, l2, Key /@ i]
		]
	],
	First[Reap[
		Function[e1,
			Function[e2,
				If[compatibleQ[{e1, e2}],
					Sow[Join[e1, e2]]
				]
			] /@ l2
		] /@ l1
	][[2]], {}]
];
SPARQLJoinAcross[x___, a_ArrayAssociation, y___] := SPARQLJoinAcross[x, Normal[a], y];

compatibleQ[sols : {___?AssociationQ}] := SameQ @@ KeyIntersection[sols]

End[];
EndPackage[];
