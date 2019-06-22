BeginPackage["GraphStore`IRIValue`", {"GraphStore`"}];

Needs["GraphStore`Formats`"];
Needs["GraphStore`RDF`"];
Needs["GraphStore`SPARQL`"];

IRIValue;
Options[IRIValue] = {
	SPARQLEndpoint -> None
};

SPARQLEndpoint;

Begin["`Private`"];

IRIValue[args___] := With[{res = Catch[iIRIValue[args], $failTag]}, res /; res =!= $failTag];


fail[___, f_Failure, ___] := Throw[f, $failTag];
fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[execute];
Options[execute] = Options[IRIValue];
execute[subj_, query_, opts : OptionsPattern[]] := SPARQLExecute[
	OptionValue[SPARQLEndpoint] // Replace[
		None :> iIRIValue[subj, "RDFStore"]
	],
	query
];


clear[iIRIValue];
Options[iIRIValue] = Options[IRIValue];


(* subject - predicate *)
iIRIValue[s_IRI, p_IRI, opts : OptionsPattern[]] := execute[
	s,
	SPARQLSelect[
		RDFTriple[s, p, SPARQLVariable["o"]]
	],
	opts
] // Query[All, "o"];

iIRIValue[_IRI, {}, OptionsPattern[]] := {};
iIRIValue[s_IRI, pList : {__IRI}, opts : OptionsPattern[]] := execute[
	s,
	SPARQLSelect[{
		SPARQLValues["p", pList],
		RDFTriple[s, SPARQLVariable["p"], SPARQLVariable["o"]]
	}],
	opts
] // GroupBy[Key["p"] -> Key["o"]] //
Lookup[#, pList, {}] &;

iIRIValue[{}, _IRI, OptionsPattern[]] := {};
iIRIValue[sList : {__IRI}, p_IRI, opts : OptionsPattern[]] := execute[
	sList,
	SPARQLSelect[{
		SPARQLValues["s", sList],
		RDFTriple[SPARQLVariable["s"], p, SPARQLVariable["o"]]
	}],
	opts
] // GroupBy[Key["s"] -> Key["o"]] //
Lookup[#, sList, {}] &;

iIRIValue[{}, {___IRI}, OptionsPattern[]] := {};
iIRIValue[sList : {___IRI}, {}, OptionsPattern[]] := ConstantArray[{}, Length[sList]];
iIRIValue[sList : {__IRI}, pList : {__IRI}, opts : OptionsPattern[]] := execute[
	sList,
	SPARQLSelect[{
		SPARQLValues["s", sList],
		SPARQLValues["p", pList],
		RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]
	}],
	opts
] // GroupBy[{Key["s"], Key["p"] -> Key["o"]}] //
Lookup[#, sList, <||>] & //
Lookup[#, pList, {}] &;


(* subject - property *)
iIRIValue[{}, "RDFStore", OptionsPattern[]] := RDFStore[{}];
iIRIValue[i : _IRI | {__IRI}, "RDFStore", opts : OptionsPattern[]] := OptionValue[SPARQLEndpoint] // Replace[{
	None :> If[ListQ[i],
		RDFMerge[iIRIValue[#, "RDFStore", opts] & /@ i],
		Module[
			{res},
			res = URLRead[
				HTTPRequest[
					URL @@ i,
					<|
						"Headers" -> {
							"Accept" -> $RDFMediaTypes
						}
					|>
				]
			];
			If[FailureQ[res],
				fail[res];
			];
			ImportByteArray[
				res["BodyByteArray"],
				MediaTypeToFormat[res["ContentType"]]
			]
		]
	],
	endpoint_ :> Module[
		{res},
		res = SPARQLExecute[
			endpoint,
			SPARQLConstruct[
				If[ListQ[i],
					{
						SPARQLValues["s", i],
						RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]
					} -> RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]],
					RDFTriple[i, SPARQLVariable["p"], SPARQLVariable["o"]]
				]
			]
		];
		If[FailureQ[res],
			fail[res];
		];
		res
	]
}];

iIRIValue[{}, "Association", OptionsPattern[]] := <||>;
iIRIValue[i : _IRI | {__IRI}, "Association", opts : OptionsPattern[]] := execute[
	i,
	SPARQLSelect[
		If[ListQ[i],
			{
				SPARQLValues["s", i],
				RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]
			},
			RDFTriple[i, SPARQLVariable["p"], SPARQLVariable["o"]]
		]
	],
	opts
] // If[ListQ[i],
	GroupBy[{Key["s"], Key["p"] -> Key["o"]}],
	GroupBy[Key["p"] -> Key["o"]]
];

iIRIValue[i_, "Dataset", opts : OptionsPattern[]] := Dataset[iIRIValue[i, "Association", opts]];

iIRIValue[{}, "Classes", OptionsPattern[]] := {};
iIRIValue[i : _IRI | {__IRI}, "Classes", opts : OptionsPattern[]] := execute[
	i,
	SPARQLSelect[
		If[ListQ[i],
			{
				SPARQLValues["s", i],
				RDFTriple[SPARQLVariable["s"], RDFPrefixData["rdf", "type"], SPARQLVariable["class"]]
			},
			RDFTriple[i, RDFPrefixData["rdf", "type"], SPARQLVariable["class"]]
		]
	] /*
	SPARQLProject["class"] /*
	SPARQLDistinct[],
	opts
] // Query[All, "class"];

iIRIValue[{}, "Properties", OptionsPattern[]] := {};
iIRIValue[i : _IRI | {__IRI}, "Properties", opts : OptionsPattern[]] := execute[
	i,
	SPARQLSelect[
		If[ListQ[i],
			{
				SPARQLValues["s", i],
				RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]
			},
			RDFTriple[i, SPARQLVariable["p"], SPARQLVariable["o"]]
		]
	] /*
	SPARQLProject["p"] /*
	SPARQLDistinct[],
	opts
] // Query[All, "p"];


(* one argument *)
iIRIValue[i_, opts : OptionsPattern[]] := iIRIValue[i, "RDFStore", opts];

End[];
EndPackage[];
