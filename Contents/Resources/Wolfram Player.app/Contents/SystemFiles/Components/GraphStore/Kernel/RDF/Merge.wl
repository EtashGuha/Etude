BeginPackage["GraphStore`RDF`Merge`", {"GraphStore`", "GraphStore`RDF`"}];
Begin["`Private`"];

RDFMerge[args___] := With[{res = Catch[iRDFMerge[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* merge *)

clear[iRDFMerge];
iRDFMerge[{}] := RDFStore[{}];
iRDFMerge[{store_RDFStore}] := store;
iRDFMerge[{RDFStore[default1_List, named1_?AssociationQ], RDFStore[default2_List, named2_?AssociationQ]}] := With[
	{mapping = AssociationMap[
		RDFBlankNode[CreateUUID["b-"]] &,
		Intersection[
			Join[
				Cases[Append[Values[named1], default1], RDFBlankNode[_String], {3}],
				Cases[Keys[named1], RDFBlankNode[_String]]
			],
			Join[
				Cases[Append[Values[named2], default2], RDFBlankNode[_String], {3}],
				Cases[Keys[named2], RDFBlankNode[_String]]
			]
		]
	]},
	RDFStore[
		Join[
			default1,
			Replace[default2, mapping, {2}]
		],
		Join[
			named1,
			named2 // KeyMap[Replace[mapping]] // Map[Replace[#, mapping, {2}] &]
		]
	]
];
iRDFMerge[{store1_, store2_, rest__}] := iRDFMerge[{iRDFMerge[{store1, store2}], rest}];

(* end merge *)
(* -------------------------------------------------- *)

End[];
EndPackage[];
