BeginPackage["GraphStore`SPARQL`ResultsEqual`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`RDF`"];

Begin["`Private`"];

SPARQLResultsEqual[a_?BooleanQ, b_?BooleanQ] := a === b;
SPARQLResultsEqual[a : {___?AssociationQ}, b : {___?AssociationQ}] := ContainsExactly[
	a,
	b,
	SameTest -> Function[{sol1, sol2},
		With[{
			bn1 = DeleteDuplicates[Cases[sol1, RDFBlankNode[_]]],
			bn2 = DeleteDuplicates[Cases[sol2, RDFBlankNode[_]]]
		},
			Which[
				{} === bn1 === bn2,
				Equal[normalizeSolution[sol1], normalizeSolution[sol2]],
				Length[bn1] === Length[bn2],
				With[{
					s1 = normalizeSolution[sol1],
					s2 = normalizeSolution[sol2]
				},
					AnyTrue[
						Range[0, Length[bn1] - 1],
						Function[r,
							Equal[
								s1 /. Thread[bn1 -> RotateRight[bn2, r]],
								s2
							]
						]
					]
				],
				True,
				False
			]
		]
	]
];
SPARQLResultsEqual[a_RDFStore, b_RDFStore] := IsomorphicRDFStoreQ[a, b];
SPARQLResultsEqual[_, _] := False;

normalizeSolution[sol_?AssociationQ] := KeySort[sol] /. {
	RDFString[s_, lang_?StringQ] :> With[{tmp = RDFString[s, ToLowerCase[lang]]}, tmp /; True],
	RDFLiteral[s_, IRI[dt_?StringQ]] :> RDFLiteral[s, dt]
};

End[];
EndPackage[];
