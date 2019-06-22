BeginPackage["GraphStore`RDF`RDFStore`", {"GraphStore`", "GraphStore`RDF`"}];
Begin["`Private`"];

Needs["GraphStore`SPARQL`"];

Options[RDFStore] = {
	SPARQLEntailmentRegime -> None
};

RDFStore[default_List, ___]["DefaultGraph"] := default;
RDFStore[_, named_Association, ___]["NamedGraphs"] := named;

RDFStore[default_List, opts : OptionsPattern[]] := RDFStore[default, <||>, opts];
RDFStore[default_List?(MemberQ[RDFTriple[_RDFCollection, _, _] | RDFTriple[_, _, _RDFCollection]]), named_, opts : OptionsPattern[]] := With[
	{ex = ExpandRDFCollection[default]},
	RDFStore[
		ex,
		named,
		opts
	] /; FreeQ[ex, ExpandRDFCollection]
];
RDFStore[default_, named_Association?(AnyTrue[MemberQ[RDFTriple[_RDFCollection, _, _] | RDFTriple[_, _, _RDFCollection]]]), opts : OptionsPattern[]] := With[
	{ex = named // Map[ExpandRDFCollection]},
	RDFStore[
		default,
		ex,
		opts
	] /; FreeQ[ex, ExpandRDFCollection]
];
RDFStore[default_List?(Not @* DuplicateFreeQ), named_, opts : OptionsPattern[]] := RDFStore[DeleteDuplicates[default], named, opts];
RDFStore[default_, named_Association?(AnyTrue[Not @* DuplicateFreeQ]), opts : OptionsPattern[]] := RDFStore[default, DeleteDuplicates /@ named, opts];

RDFStore /: MakeBoxes[g_RDFStore, fmt_] := With[{res = Catch[iRDFStoreMakeBoxes[g, fmt], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* formatting *)

clear[iRDFStoreMakeBoxes];
iRDFStoreMakeBoxes[store : RDFStore[_List, <||>, OptionsPattern[]], fmt_] := BoxForm`ArrangeSummaryBox[
	RDFStore,
	store,
	None,
	BoxForm`SummaryItem /@ {
		{
			"Triples: ",
			execute[store, triplesQuery[]] // getNo
		},
		{
			"Entities: ",
			execute[store, entitiesQuery[]] // getNo
		},
		{
			"Classes: ",
			execute[store, classesQuery[]] // getNo
		},
		{
			"Properties: ",
			execute[store, propertiesQuery[]] // getNo
		}
	},
	BoxForm`SummaryItem /@ {
		{
			"Distinct subjects: ",
			execute[store, distinctSubjectsQuery[]] // getNo
		},
		{
			"Distinct objects: ",
			execute[store, distinctObjectsQuery[]] // getNo
		}
	},
	fmt
];
iRDFStoreMakeBoxes[store : RDFStore[_List, named_Association?(Length /* GreaterThan[0]), OptionsPattern[]], fmt_] := BoxForm`ArrangeSummaryBox[
	RDFStore,
	store,
	None,
	BoxForm`SummaryItem /@ {
		{
			"Default graph: ",
			Row[{execute[store, triplesQuery[]] // getNo, " triples"}]
		},
		{
			"Named graphs: ",
			Length[named]
		}
	},
	Function[g,
		BoxForm`SummaryItem[{
			Row[{First[g], ": "}],
			Row[{execute[store, triplesQuery[g]] // getNo, " triples"}]
		}]
	] /@ Keys[named],
	fmt
];


clear[execute];
execute[store_, query_] := store // SPARQLQuery[
	query,
	SPARQLEntailmentRegime -> None
];

clear[getNo];
getNo[{<|"no" -> no_|>}] := no;


clear[triplesQuery];
triplesQuery[] := (
	SPARQLSelect[RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]] /*
	SPARQLAggregate["no" -> SPARQLEvaluation["COUNT"][]]
);
triplesQuery[g_] := (
	SPARQLSelect[SPARQLGraph[g, RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]]] /*
	SPARQLAggregate["no" -> SPARQLEvaluation["COUNT"][]]
);

clear[entitiesQuery];
entitiesQuery[] := (
	SPARQLSelect[RDFTriple[SPARQLVariable["s"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"], SPARQLVariable["o"]]] /*
	SPARQLAggregate["no" -> SPARQLEvaluation["COUNT"][SPARQLVariable["s"], "Distinct" -> True]]
);

clear[classesQuery];
classesQuery[] := (
	SPARQLSelect[RDFTriple[SPARQLVariable["s"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"], SPARQLVariable["o"]]] /*
	SPARQLAggregate["no" -> SPARQLEvaluation["COUNT"][SPARQLVariable["o"], "Distinct" -> True]]
);

clear[propertiesQuery];
propertiesQuery[] := (
	SPARQLSelect[RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]] /*
	SPARQLAggregate["no" -> SPARQLEvaluation["COUNT"][SPARQLVariable["p"], "Distinct" -> True]]
);

clear[distinctSubjectsQuery];
distinctSubjectsQuery[] := (
	SPARQLSelect[RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]] /*
	SPARQLAggregate["no" -> SPARQLEvaluation["COUNT"][SPARQLVariable["s"], "Distinct" -> True]]
);

clear[distinctObjectsQuery];
distinctObjectsQuery[] := (
	SPARQLSelect[RDFTriple[SPARQLVariable["s"], SPARQLVariable["p"], SPARQLVariable["o"]]] /*
	SPARQLAggregate["no" -> SPARQLEvaluation["COUNT"][SPARQLVariable["o"], "Distinct" -> True]]
);

(* end formatting *)
(* -------------------------------------------------- *)

End[];
EndPackage[];
