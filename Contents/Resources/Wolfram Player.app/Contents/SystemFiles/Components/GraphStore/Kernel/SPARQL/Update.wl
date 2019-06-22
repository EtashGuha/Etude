BeginPackage["GraphStore`SPARQL`Update`", {"GraphStore`", "GraphStore`SPARQL`"}];
Begin["`Private`"];

Options[SPARQLAdd] = {
	"Silent" -> False
};
SPARQLAdd[args___][gs_] := With[{res = Catch[iSPARQLAdd[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLClear] = {
	"Silent" -> False
};
SPARQLClear[args___][gs_] := With[{res = Catch[iSPARQLClear[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLCopy] = {
	"Silent" -> False
};
SPARQLCopy[args___][gs_] := With[{res = Catch[iSPARQLCopy[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLCreate] = {
	"Silent" -> False
};
SPARQLCreate[args___][gs_] := With[{res = Catch[iSPARQLCreate[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLDelete] = {
	"Using" -> Automatic,
	"UsingNamed" -> Automatic,
	"With" -> None
};
SPARQLDelete[args___][gs_] := With[{res = Catch[iSPARQLDelete[gs, args], $failTag]}, res /; res =!= $failTag];

SPARQLDeleteData[args___][gs_] := With[{res = Catch[iSPARQLDeleteData[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLDeleteInsert] = {
	"Using" -> Automatic,
	"UsingNamed" -> Automatic,
	"With" -> None
};
SPARQLDeleteInsert[args___][gs_] := With[{res = Catch[iSPARQLDeleteInsert[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLDrop] = {
	"Silent" -> False
};
SPARQLDrop[args___][gs_] := With[{res = Catch[iSPARQLDrop[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLInsert] = {
	"Using" -> Automatic,
	"UsingNamed" -> Automatic,
	"With" -> None
};
SPARQLInsert[args___][gs_] := With[{res = Catch[iSPARQLInsert[gs, args], $failTag]}, res /; res =!= $failTag];

SPARQLInsertData[args___][gs_] := With[{res = Catch[iSPARQLInsertData[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLLoad] = {
	"Silent" -> False
};
SPARQLLoad[args___][gs_] := With[{res = Catch[iSPARQLLoad[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLMove] = {
	"Silent" -> False
};
SPARQLMove[args___][gs_] := With[{res = Catch[iSPARQLMove[gs, args], $failTag]}, res /; res =!= $failTag];

Options[SPARQLUpdate] = {
	"Base" -> None
};
SPARQLUpdate[args___][gs_] := With[{res = Catch[iSPARQLUpdate[gs, args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


PossibleUpdateQ[_SPARQLAdd | _SPARQLClear | _SPARQLCopy | _SPARQLCreate | _SPARQLDelete | _SPARQLDeleteData | _SPARQLDeleteInsert | _SPARQLDrop | _SPARQLInsert | _SPARQLInsertData | _SPARQLLoad | _SPARQLMove | _SPARQLUpdate] := True;
PossibleUpdateQ[comp : _Composition | _RightComposition] := AllTrue[comp, PossibleUpdateQ];
PossibleUpdateQ[_] := False;


(* -------------------------------------------------- *)
(* update *)

clear[iSPARQLUpdate];
Options[iSPARQLUpdate] = Options[SPARQLUpdate];
iSPARQLUpdate[gs_, update_, OptionsPattern[]] := Block[
	{$Base},
	$Base = OptionValue["Base"];
	Module[
		{updateexp},
		updateexp = update;
		Switch[updateexp,
			_String,
			updateexp = ImportString[updateexp, "SPARQLUpdate"],
			_File | _IRI | _URL,
			updateexp = Import[updateexp, "SPARQLUpdate"]
		];
		gs // updateexp // Replace[_updateexp :> fail[]]
	]
];

(* end update *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* translation *)

clear[applyWith];
applyWith[patt_, None] := patt;
applyWith[patt_, g_] := SPARQLGraph[g, patt];

clear[iriQ];
iriQ[_File | _IRI | _URL] := True;
iriQ[IRI[{_?StringQ, _?StringQ}]] := True;
iriQ[_] := False;


clear[iSPARQLAdd];
iSPARQLAdd[store : RDFStore[default_, named_?AssociationQ], from_, to_, OptionsPattern[SPARQLAdd]] := If[from === to,
	store,
	datasetUnion[
		store,
		If[to === "Default",
			RDFStore[
				Lookup[named, from, If[TrueQ[OptionValue["Silent"]], Return[store, With], fail[]]],
				named
			],
			RDFStore[
				default,
				Join[
					named,
					<|to -> If[from === "Default",
						default,
						Lookup[named, from, If[TrueQ[OptionValue["Silent"]], Return[store, With], fail[]]]
					]|>
				]
			]
		]
	]
];

clear[iSPARQLClear];
iSPARQLClear[store_RDFStore, x_, opts : OptionsPattern[SPARQLClear]] := opClear[store, x, opts];

clear[iSPARQLCopy];
iSPARQLCopy[store : RDFStore[default_, named_?AssociationQ], from_, to_, OptionsPattern[SPARQLCopy]] := If[from === to,
	store,
	If[to === "Default",
		RDFStore[
			Lookup[named, from, If[TrueQ[OptionValue["Silent"]], Return[store, With], fail[]]],
			named
		],
		RDFStore[
			default,
			Join[
				named,
				<|to -> If[from === "Default",
					default,
					Lookup[named, from, If[TrueQ[OptionValue["Silent"]], Return[store, With], fail[]]]
				]|>
			]
		]
	]
];

clear[iSPARQLCreate];
iSPARQLCreate[store : RDFStore[_, named_?AssociationQ], i_?iriQ, OptionsPattern[SPARQLCreate]] := If[KeyExistsQ[named, i],
	If[TrueQ[OptionValue["Silent"]], store, fail[]],
	opCreate[store, i]
];

clear[iSPARQLDelete];
iSPARQLDelete[store_RDFStore, (Rule | RuleDelayed)[where_, template_], OptionsPattern[SPARQLDelete]] := opDeleteInsert[
	store,
	ChooseRDFStore[OptionValue[{"Using", "UsingNamed"}], store],
	applyWith[template, OptionValue["With"]],
	{},
	applyWith[where, OptionValue["With"]]
];
iSPARQLDelete[store_, Except[_Rule | _RuleDelayed, where_], rest___] := iSPARQLDelete[store, where -> where, rest];

clear[iSPARQLDeleteData];
iSPARQLDeleteData[store_RDFStore, qp_, OptionsPattern[SPARQLDeleteData]] := opDeleteData[store, qp];

clear[iSPARQLDeleteInsert];
iSPARQLDeleteInsert[store_RDFStore, del_, ins_, where_, OptionsPattern[SPARQLDeleteInsert]] := opDeleteInsert[
	store,
	ChooseRDFStore[OptionValue[{"Using", "UsingNamed"}], store],
	applyWith[del, OptionValue["With"]],
	applyWith[ins, OptionValue["With"]],
	applyWith[where, OptionValue["With"]]
];

clear[iSPARQLDrop];
iSPARQLDrop[store : RDFStore[_, named_?AssociationQ], i_?iriQ, OptionsPattern[SPARQLDrop]] := If[KeyExistsQ[named, i],
	opDrop[store, i],
	If[TrueQ[OptionValue["Silent"]], store, fail[]]
];
iSPARQLDrop[store_RDFStore, x_, OptionsPattern[]] := opDrop[store, x];

clear[iSPARQLInsert];
iSPARQLInsert[store_RDFStore, (Rule | RuleDelayed)[where_, template_], OptionsPattern[SPARQLInsert]] := opDeleteInsert[
	store,
	ChooseRDFStore[OptionValue[{"Using", "UsingNamed"}], store],
	{},
	applyWith[template, OptionValue["With"]],
	applyWith[where, OptionValue["With"]]
];

clear[iSPARQLInsertData];
iSPARQLInsertData[store_RDFStore, qp_, OptionsPattern[SPARQLInsertData]] := opInsertData[store, qp];

clear[iSPARQLLoad];
iSPARQLLoad[store_RDFStore, di_?iriQ, opts : OptionsPattern[SPARQLLoad]] := opLoad[store, di, opts];
iSPARQLLoad[store_RDFStore, (Rule | RuleDelayed)[di_?iriQ, i_?iriQ], opts : OptionsPattern[SPARQLLoad]] := opLoad[store, di, i, opts];

clear[iSPARQLMove];
iSPARQLMove[store : RDFStore[default_, named_?AssociationQ], from_, to_, OptionsPattern[SPARQLMove]] := If[from === to,
	store,
	If[to === "Default",
		RDFStore[
			Lookup[named, from, If[TrueQ[OptionValue["Silent"]], Return[store, With], fail[]]],
			KeyDrop[named, {from}]
		],
		RDFStore[
			If[from === "Default",
				{},
				default
			],
			Join[
				If[from === "Default",
					named,
					KeyDrop[named, {from}]
				],
				<|to -> If[from === "Default",
					default,
					Lookup[named, from, If[TrueQ[OptionValue["Silent"]], Return[store, With], fail[]]]
				]|>
			]
		]
	]
];

(* end translation *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* evaluation *)

clear[graphUnion];
graphUnion[g___] := Union @@ {g};

clear[graphDiff];
graphDiff[g___] := Complement @@ {g};


(* 4.2 Auxiliary Definitions *)

(* 4.2.1 Dataset-UNION *)
clear[datasetUnion];
datasetUnion[] := RDFStore[{}];
datasetUnion[store_] := store;
datasetUnion[RDFStore[default1_List, named1_?AssociationQ], RDFStore[default2_List, named2_?AssociationQ]] := RDFStore[
	graphUnion[default1, default2],
	Merge[{named1, named2}, Apply[graphUnion]]
];
datasetUnion[ds1_, ds2_, rest__] := datasetUnion[datasetUnion[ds1, ds2], rest];

(* 4.2.2 Dataset-DIFF *)
clear[datasetDiff];
datasetDiff[RDFStore[default1_List, named1_?AssociationQ], RDFStore[default2_List, named2_?AssociationQ]] := RDFStore[
	graphDiff[default1, default2],
	Join[
		KeyComplement[{named1, named2}],
		Merge[KeyIntersection[{named1, named2}], Apply[graphDiff]]
	]
];

(* 4.2.3 Dataset( QuadPattern, μ, DS, GS ) *)
clear[datasetMu];
datasetMu[qp : {__RDFTriple}, sol_?AssociationQ, ds_, gs_] := RDFStore[Union[sk[sol, qp] /. sol]];
datasetMu[t_RDFTriple, rest___] := datasetMu[{t}, rest];
datasetMu[SPARQLGraph[g_, qp : {__RDFTriple}], sol_?AssociationQ, ds_, gs_] := RDFStore[{}, <|Replace[g, sol] -> Union[sk[sol, qp] /. sol]|>];
datasetMu[SPARQLGraph[g_, t_RDFTriple], rest___] := datasetMu[SPARQLGraph[g, {t}], rest];
datasetMu[SPARQLGraph[_, g_SPARQLGraph], rest___] := datasetMu[g, rest];
datasetMu[SPARQLGraph[g_, l_List], rest___] := datasetMu[Replace[l, Except[_SPARQLGraph, x_] :> SPARQLGraph[g, x], {1}], rest];
datasetMu[qp_List, sol_, ds_, gs_] := datasetUnion @@ Function[datasetMu[#, sol, ds, gs]] /@ qp;

(* 4.2.4 Dataset( QuadPattern, P, DS, GS ) *)
clear[datasetP];
datasetP[qp_, patt_, ds_, gs_] := Block[
	{sk, skCache},
	sk[sol_, template_List] := Replace[
		template,
		b : RDFBlankNode[_String] :> skCache[sol, b],
		{2}
	];
	skCache[sol_, b_] := skCache[sol, b] = EvaluateSPARQLFunction["BNODE"];
	datasetUnion @@ Function[sol, datasetMu[qp, sol // KeyMap[SPARQLVariable], ds, gs]] /@ SPARQLSelect[patt][ds]
];


(* 4.3 Graph Update Operations *)

(* 4.3.1 Insert Data Operation *)
clear[opInsertData];
opInsertData[gs_, qp_] := datasetUnion[gs, datasetP[qp, {}, gs, gs]];

(* 4.3.2 Delete Data Operation *)
opDeleteData[gs_, qp_] := datasetDiff[gs, datasetP[qp, {}, gs, gs]];

(* 4.3.3 Delete Insert Operation *)
clear[opDeleteInsert];
opDeleteInsert[gs_, ds_, del_, ins_, patt_] := datasetUnion[
	datasetDiff[
		gs,
		datasetP[del, patt, ds, gs]
	],
	datasetP[ins, patt, ds, gs]
];

(* 4.3.4 Load Operation *)
clear[opLoad];
opLoad[gs_, di_, OptionsPattern[SPARQLLoad]] := With[
	{g = If[TrueQ[OptionValue["Silent"]],
		Quiet[Import[di]],
		Import[di]
	]},
	If[FailureQ[g],
		If[TrueQ[OptionValue["Silent"]], gs, fail[]],
		datasetUnion[gs, g]
	]
];
opLoad[gs_, di_, i_, OptionsPattern[SPARQLLoad]] := With[
	{g = If[TrueQ[OptionValue["Silent"]],
		Quiet[Import[di]],
		Import[di]
	]},
	If[FailureQ[g],
		If[TrueQ[OptionValue["Silent"]], gs, fail[]],
		datasetUnion[gs, RDFStore[{}, <|i -> First[g]|>]]
	]
];

(* 4.3.5 Clear Operation *)
clear[opClear];
opClear[store : RDFStore[default_, named_?AssociationQ], i : _File | _IRI | _URL, OptionsPattern[SPARQLClear]] := If[KeyExistsQ[named, i],
	RDFStore[default, Join[named, <|i -> {}|>]],
	If[TrueQ[OptionValue["Silent"]], store, fail[]]
];
opClear[RDFStore[_, named_], "Default", OptionsPattern[SPARQLClear]] := RDFStore[{}, named];
opClear[RDFStore[default_, named_?AssociationQ], "Named", OptionsPattern[SPARQLClear]] := RDFStore[default, {} & /@ named];
opClear[RDFStore[_, named_?AssociationQ], "All" | All, OptionsPattern[SPARQLClear]] := RDFStore[{}, {} & /@ named];


(* 4.4 Graph Management Operations *)

(* 4.4.1 Create Operation *)
clear[opCreate];
opCreate[store : RDFStore[default_, named_?AssociationQ], i_] := If[KeyExistsQ[named, i],
	store,
	RDFStore[default, Append[named, i -> {}]]
];

(* 4.4.2 Drop Operation *)
clear[opDrop];
opDrop[store : RDFStore[default_, named_?AssociationQ], i : _File | _IRI | _URL] := RDFStore[default, KeyDrop[named, {i}]];
opDrop[store_, default : "Default"] := opClear[store, default];
opDrop[RDFStore[default_, _], "Named"] := RDFStore[default, <||>];
opDrop[_, "All" | All] := RDFStore[{}, <||>];

(* end evaluation *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
