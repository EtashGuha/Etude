Package["NeuralNetworks`"]

PackageExport["NetInsert"]


DoWith[
$head /: Insert[net_$head ? ValidNetQ, spec_, pos_] := NetInsert[net, spec, pos],
$head -> {NetChain, NetGraph}
]


(***************************)
(* NetChain implementation *)
(***************************)

General::invnetins1 = "Use NetInsert[chain, \"name\" -> layer, pos] to insert into a chain composed of named layers."
General::invnetins2 = "Use NetInsert[chain, layer, pos] to insert into a chain composed of unnamed layers."
General::invnetins3 = "Key `` already exists in chain."
General::invnetins4 = "`` is not a valid position to insert the given layer."

NetInsert[nc_NetChain ? ValidNetQ, layer_, pos_] := CatchFailureAsMessage @ Scope[
	newkey = Null;
	data = NData[nc];
	If[!StringQ[pos] && !IntegerQ[pos], ThrowFailure["invnetins4", pos]];
	If[RuleQ[layer], newkey = First[layer]; layer = Last[layer]];
	layer = ToLayer[layer];
	layers = data["Nodes"]; lay1 = First[layers]; layn = Last[layers];
	data = NData[nc];
	If[DigitStringKeysQ[layers],
		If[!IntegerQ[pos] || newkey =!= Null, ThrowFailure["invnetins2"]];
		If[Not[1 <= Abs[pos] <= Length[layers]+1], ThrowFailure["invnetins4", pos, Length[layers]]];
		layers = NumberedAssociation @ Insert[Values[layers], layer, pos];
	,
		If[newkey === Null, ThrowFailure["invnetins1"]];
		If[KeyExistsQ[layers, newkey], ThrowFailure["invnetins3", newkey]];
		If[!Or[
			IntegerQ[pos] && 1 <= Abs[pos] <= Length[layers]+1, 
			StringQ[pos] && KeyExistsQ[layers, pos]],
			ThrowFailure["invnetins4", pos]
		];
		layers = Insert[layers, newkey -> layer, pos]
	];
	ioSpecs = chainIOSpecs[data, lay1 === First[layers], layn === Last[layers]];
	(* ^ expensive to do the ===, but i'm lazy *)
	sharedArrays = GetSharedArrays[data];
	toNetChain[layers, ioSpecs, sharedArrays]
];

(***************************)
(* NetGraph implementation *)
(***************************)

NetInsert::arg1 = "First argument to NetInsert should be a NetChain."; 
NetInsert[_, _, _] := (Message[NetInsert::arg1]; $Failed);

DeclareArgumentCount[NetInsert, 3];
