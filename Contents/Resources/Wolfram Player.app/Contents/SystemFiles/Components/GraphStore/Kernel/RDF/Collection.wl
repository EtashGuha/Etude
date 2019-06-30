BeginPackage["GraphStore`RDF`Collection`", {"GraphStore`", "GraphStore`RDF`"}];
Begin["`Private`"];

CompactRDFCollection[args___] := With[{res = Catch[iCompactRDFCollection[args], $failTag]}, res /; res =!= $failTag];
ExpandRDFCollection[args___] := With[{res = Catch[iExpandRDFCollection[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


rdf[s_] := IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s];


clear[iCompactRDFCollection];
iCompactRDFCollection[l_List] := Module[
	{res, restPos, firstPos, current, before, bn, coll, collections},
	res = l;
	res = replaceRepeated[
		res,
		{
			RDFTriple[s_, Except[rdf["rest"], p_], rdf["nil"]] :> RDFTriple[s, p, RDFCollection[{}]],
			RDFTriple[rdf["nil"], p_, o_] :> RDFTriple[RDFCollection[{}], p, o]
		},
		{1}
	];
	restPos = With[
		{restIndex = With[
			{rpos = listPosition[l, RDFTriple[_, rdf["rest"], Except[rdf["nil"]]]]},
			rpos[[#]] & /@ PositionIndex[l[[rpos, 3]]][[All, 1]]
		]},
		Function[end,
			current = {end};
			While[
				before = Lookup[restIndex, l[[First[current], 1]]];
				! MissingQ[before],
				PrependTo[current, before]
			];
			current
		] /@ listPosition[res, RDFTriple[_, rdf["rest"], rdf["nil"]]]
	];
	firstPos = With[
		{firstIndex = With[
			{fpos = listPosition[l, RDFTriple[_, rdf["first"], _]]},
			fpos[[#]] & /@ PositionIndex[l[[fpos, 1]]][[All, 1]]
		]},
		Lookup[firstIndex, l[[#, 1]]] & /@ restPos
	];
	collections = <||>;
	res = Fold[
		Function[{r, ind},
			bn = l[[First[First[ind]], 1]];
			coll = RDFCollection[l[[Last[ind], 3]]];
			AppendTo[collections, bn -> coll];
			replaceRepeated[
				r,
				{
					RDFTriple[s_, p_, bn] :> RDFTriple[s, p, coll],
					RDFTriple[bn, p_, o_] :> RDFTriple[coll, p, o]
				},
				{1}
			]
		],
		res,
		Thread[{restPos, firstPos}]
	];
	res = Delete[res, List /@ Flatten[Join[restPos, firstPos]]];
	res = res //. collections;
	res
];

clear[listPosition];
listPosition[expr_, patt_] := Position[expr, patt, {1}, Heads -> False][[All, 1]];

clear[replaceRepeated];
replaceRepeated[expr_, args__] := FixedPoint[
	Replace[#, args] &,
	expr
];


clear[iExpandRDFCollection];
iExpandRDFCollection[RDFTriple[col_RDFCollection, p_, o_]] := With[
	{tl = collectionTriples[col]},
	Prepend[
		Last[tl],
		RDFTriple[First[tl], p, o]
	]
];
iExpandRDFCollection[RDFTriple[s_, p_, col_RDFCollection]] := With[
	{tl = collectionTriples[col]},
	Prepend[
		Last[tl],
		RDFTriple[s, p, First[tl]]
	]
];
iExpandRDFCollection[l_List] := FixedPoint[
	Replace[
		#,
		t : RDFTriple[_, _, _RDFCollection] | RDFTriple[_RDFCollection, _, _] :> Sequence @@ iExpandRDFCollection[t],
		{1}
	] &,
	l
];

clear[collectionTriples];
collectionTriples[RDFCollection[{}]] := {
	rdf["nil"],
	{}
};
collectionTriples[RDFCollection[l : {__}]] := With[
	{tl = Join @@ BlockMap[
		Apply[Function[{current, next},
			{
				RDFTriple[First[current], rdf["first"], Last[current]],
				RDFTriple[First[current], rdf["rest"], First[next]]
			}
		]],
		Append[
			{RDFBlankNode[CreateUUID["b-"]], #} & /@ l,
			{rdf["nil"]}
		],
		2,
		1
	]},
	{
		tl[[1, 1]],
		tl
	}
];
collectionTriples[RDFCollection[x_]] := (Message[RDFCollection::invl, x]; fail[]);
collectionTriples[RDFCollection[x___]] := (Message[RDFCollection::argx, RDFCollection, Length[Hold[x]]]; fail[]);

End[];
EndPackage[];
