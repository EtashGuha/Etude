(* optimization based on: *)
(* Matching RDF Graphs *)
(* Jeremy J. Carroll *)
(* https://pdfs.semanticscholar.org/8f1c/16deb1c32c552da3f7f6e462e193f5389e74.pdf *)

BeginPackage["GraphStore`RDF`Isomorphism`", {"GraphStore`", "GraphStore`RDF`"}];
Begin["`Private`"];

FindRDFGraphIsomorphism[args___] := With[{res = Catch[iFindRDFGraphIsomorphism[args], $failTag]}, res /; res =!= $failTag];
IsomorphicRDFStoreQ[args___] := With[{res = Catch[iIsomorphicRDFStoreQ[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* find graph isomorphism *)

clear[iFindRDFGraphIsomorphism];

iFindRDFGraphIsomorphism[g1_, g2_] := iFindRDFGraphIsomorphism[g1, g2, 1];
iFindRDFGraphIsomorphism[g1_, g2_, 0] := {};
iFindRDFGraphIsomorphism[g1_, g2_, All] := iFindRDFGraphIsomorphism[g1, g2, Infinity];
iFindRDFGraphIsomorphism[g1_List, g2_List, n_] /; MemberQ[g1, RDFBlankNode[], {2}] || MemberQ[g2, RDFBlankNode[], {2}] := Module[
	{res, genbn = {}},
	res = iFindRDFGraphIsomorphism[
		Replace[g1, RDFBlankNode[] :> Last[AppendTo[genbn, RDFBlankNode[CreateUUID["b-"]]]], {2}],
		Replace[g2, RDFBlankNode[] :> Last[AppendTo[genbn, RDFBlankNode[CreateUUID["b-"]]]], {2}],
		n
	];
	res = res // KeyDrop[genbn] // DeleteCases[#, Alternatives @@ genbn, {2}] &;
	res
];
iFindRDFGraphIsomorphism[g1_List, g2_List, n : _Integer?NonNegative | Infinity] := Catch[
	If[Length[g1] =!= Length[g2],
		Throw[{}, $tag]
	];
	If[CountDistinct[Join @@ g1] =!= CountDistinct[Join @@ g2],
		Throw[{}, $tag]
	];
	If[CountDistinct[Cases[g1, _RDFBlankNode, {2}]] =!= CountDistinct[Cases[g2, _RDFBlankNode, {2}]],
		Throw[{}, $tag]
	];
	If[Length[g1] === 0,
		Throw[{<||>}, $tag]
	];
	If[Count[g1, _RDFBlankNode, {2}] === 0,
		If[ContainsExactly[g1, g2],
			Throw[{<||>}, $tag],
			Throw[{}, $tag]
		]
	];
	Module[{
		e1 = g1,
		e2 = g2,
		c1 = <|{} -> DeleteDuplicates[Cases[g1, _RDFBlankNode, {2}]]|>,
		c2 = <|{} -> DeleteDuplicates[Cases[g2, _RDFBlankNode, {2}]]|>,
		isoList = {}
	},
		While[
			With[{
				c1new = reclassifyVertices[g1, c1],
				c2new = reclassifyVertices[g2, c2]
			},
				If[c1new === c1 && c2new === c2,
					Break[],
					{c1, c2} = {c1new, c2new}
				];
				If[! ContainsExactly[Keys[c1], Keys[c2]],
					Throw[{}, $tag]
				];
				True
			]
		];
		Product[i!, {i, Length /@ Values[c1]}] // Replace[x_ /; x > 10^7 :> fail[]];
		Function[iso,
			If[ContainsExactly[Replace[e1, iso, {2}], e2],
				AppendTo[isoList, iso];
				If[Length[isoList] === n,
					Throw[isoList, $tag]
				]
			]
		] /@ Join @@@ Tuples[
			Function[class,
				Function[perm2,
					AssociationThread[c1[class], perm2]
				] /@ Replace[Permutations[c2[class]], _Permutations :> fail[]]
			] /@ Keys[c1]
		];
		isoList
	],
	$tag
];

clear[reclassifyVertices];
reclassifyVertices[g_, c_] := Join @@ KeyValueMap[
	Function[{class, nodes},
		If[MatchQ[nodes, {_}],
			<|class -> nodes|>,
			GroupBy[
				nodes,
				Join[class, neighborClassCounts[g, KeyDrop[c, {class}], #]] &
			] // Replace[
				_?(Length[#] === 1 &) :> <|class -> nodes|>
			]
		]
	],
	c
];

clear[neighborClassCounts];
neighborClassCounts[g_, classes_, n_] := Flatten[
	With[
		{el = g},
		Function[patt,
			Sort[Cases[el, patt :> If[MatchQ[neighbor, _RDFBlankNode], classSize[classes, neighbor], neighbor]]]
		] /@ Permutations[RDFTriple[n, _, neighbor_]]
	],
	1
];

clear[classSize];
classSize[classes_, node_] := Length[SelectFirst[classes, MemberQ[node]]];

(* end find graph isomorphism *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* isomorphic store *)

clear[iIsomorphicRDFStoreQ];
iIsomorphicRDFStoreQ[] := True;
iIsomorphicRDFStoreQ[RDFStore[_List, _?AssociationQ]] := True;
iIsomorphicRDFStoreQ[RDFStore[default1_List, named1_?AssociationQ], RDFStore[default2_List, named2_?AssociationQ]] := And[
	iIsomorphicRDFGraphQ[default1, default2],
	Length[named1] === Length[named2],
	isomorphicListQ[Keys[named1], Keys[named2]],
	AllTrue[
		Keys[named1],
		With[
			{iso = First[findListIsomorphism[Keys[named1], Keys[named2]]]},
			Function[g,
				iIsomorphicRDFGraphQ[named1[g], named2[g /. iso]]
			]
		]
	]
];
iIsomorphicRDFStoreQ[x___] := AllTrue[
	Partition[{x}, 2, 1],
	Apply[iIsomorphicRDFStoreQ]
];
iIsomorphicRDFStoreQ[_, _] := False;

clear[isomorphicListQ];
isomorphicListQ[l1_List, l2_List] := findListIsomorphism[l1, l2] =!= {};

clear[findListIsomorphism];
findListIsomorphism[l1_List, l2_List] := With[
	{b1 = Cases[l1, _RDFBlankNode], b2 = Cases[l2, _RDFBlankNode]},
	If[Length[b1] =!= Length[b2],
		Return[{}]
	];
	SelectFirst[
		Permutations[b2],
		Function[b2p,
			ContainsExactly[
				l1 /. Thread[b1 -> b2p],
				l2
			]
		]
	] // Replace[{
		_Missing :> {},
		b2p_ :> {Thread[b1 -> b2p]}
	}]
];

(* end isomorphic store *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* isomorphic graph *)

clear[iIsomorphicRDFGraphQ];

iIsomorphicRDFGraphQ[g1_List, g2_List] := iFindRDFGraphIsomorphism[g1, g2, 1] =!= {};
iIsomorphicRDFGraphQ[_, _] := False;
iIsomorphicRDFGraphQ[x___] := AllTrue[
	Partition[{x}, 2, 1],
	Apply[iIsomorphicRDFGraphQ]
];

(* end isomorphic graph *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
