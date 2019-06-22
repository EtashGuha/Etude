Package["NeuralNetworks`"]

PackageExport["NetDelete"]

(* TODO: Shift message handlign to upvalues *)

DoWith[
$head /: Delete[net_$head ? ValidNetQ, spec_] := NetDelete[net, ToList @ spec],
$head -> {NetChain, NetGraph}
]

NetGraph /: VertexDelete[ng_NetGraph ? ValidNetQ, spec_] := NetDelete[ng, ToList @ spec];


(***************************)
(* NetChain implementation *)
(***************************)

NetDelete[nc_NetChain ? ValidNetQ, delete_] := CatchFailureAsMessage @ Scope[
	data = NData[nc];
	layers = data["Nodes"];
	layerNames = AssociationThread[#, #]& @ Keys[layers];
	deleteList = ToList[delete];
	deleteList = PartElse[layerNames, #, badLayer[#]]& /@ deleteList;
	KeyDropFrom[layers, deleteList];
	makeSubNetChain[data, layers]
]

NetDelete::invchdspec = "The layer `` does not exist.";
badLayer[spec_] := ThrowFailure["invchdspec", spec];


(***************************)
(* NetGraph implementation *)
(***************************)

NetDelete::invelide = "Cannot remove interior layer ``: its input and output are not compatible, and so removing it would disconnect the graph."

NetDelete[ng_NetGraph ? ValidNetQ, delete_] := CatchFailureAsMessage @ Scope[
	data = NData[ng];
	UnpackAssociation[data, $nodes:"Nodes", $edges:"Edges"];
	deleteList = ToList[delete];
	deleteList = toVertex /@ deleteList;
	elisions = KeyValueMap[toElisionRules[$edges, #1, #2]&, $nodes[[deleteList]]];
	KeyDropFrom[$nodes, deleteList];
	$edges = Select[$edges /. elisions, FreeQ[NetPath["Nodes", Alternatives @@ deleteList, ___]]];
	res = CatchFailure[NetGraph, makeSubNetGraph[data, $nodes, $edges]];
	If[FailureQ[res] && ValueQ[$dup] && MatchQ[res[["MessageName"]], "dupedges" | "invportshape"],
		ThrowFailure["invelide", $dup]];
	If[FailureQ[res], ThrowRawFailure[res]];
	res
];

(* this is copied in TakeDrop.m, which was itself copied from NetGraph.m.
TODO: figure out how to unify all this, but it wants to share $nodes *)
toVertex[n_ ? PositiveMachineIntegerQ] := toVertex[IntegerString[n]];
toVertex[name_String /; KeyExistsQ[$nodes, name]] := name;
toVertex[spec_] := ThrowFailure[NetGraph::invnetvert2, spec];

NetDelete::nomix = "Deletion specifications that are a mixture of NetPort and layer specs are not currently supported.";
toVertex[NetPort[_String]] := ThrowFailure["nomix"];

PackageScope["toElisionRules"]

(* also used by NetMap to implement replacement by Nothing, aka Delete *)
toElisionRules[edges_, name_, node_] :=
	Match[
		{node["Inputs"], node["Outputs"]}, 
		{<|inname_ -> type1_|>, <|outname_ -> type2_|>} /; UnifiableQ[type1, type2] :> Rule[
			NetPath["Nodes", name, "Outputs", outname],
			Replace[
				NetPath["Nodes", name, "Inputs", inname], 
				Append[edges, _ :> Return[$dup = name; Nothing]]
			]
		],
		$dup = name; Nothing
	];


NetDelete[ng_NetGraph ? ValidNetQ, ports:(NetPort[_String] | {Repeated[NetPort[_String]]})] := CatchFailureAsMessage @ Scope[
	data = NData[ng];
	UnpackAssociation[data, nodes, edges];
	portNames = DeepCases[ports, _String];
	extra = Complement[portNames, InputNames[data]];
	If[extra =!= {}, ThrowFailure["nodelport", First @ extra]];
	patt = NetPath["Inputs", Alternatives @@ portNames];
	dests = Cases[edges, (np_NetPath -> patt) :> np];
	If[dests === {} || !MatchQ[dests, List @ Repeated @ NetPath["Nodes", _, "States", _]],
		ThrowFailure["invdelport", name];
	];
	edges = DeleteCases[edges, _NetPath -> patt];
	makeSubNetGraph[data, nodes, edges]
];

NetDelete::nodelport = "`` is not an input port of graph.";
NetDelete::invdelport = "Cannot delete specified port(s) as they are connected to at least one non-state layer input.";

NetDelete::arg1 = "First argument to NetDelete should be a NetChain or NetGraph."; 
NetDelete[_, _] := (Message[NetDelete::arg1]; $Failed);

DeclareArgumentCount[NetDelete, 2];
