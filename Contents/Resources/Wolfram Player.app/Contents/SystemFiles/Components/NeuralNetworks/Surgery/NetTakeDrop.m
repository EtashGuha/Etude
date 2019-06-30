Package["NeuralNetworks`"]

PackageExport["NetTake"]
PackageExport["NetDrop"]

DoWith[
$head /: $func[net_$head ? ValidNetQ, spec_] := CatchFailureAsMessage[$func, NetTakeDrop[$func, net, spec]],
$func -> {Take, Drop},
$head -> {NetChain, NetGraph}
]

NetTake[net_, startend_, p:$PortHintP] := CatchFailureAsMessage[
	TestValidNet[net];
	AttachPortHints[p] @ NetTakeDrop[Take, net, startend]
];

NetDrop[net_, startend_, p:$PortHintP] := CatchFailureAsMessage[
	TestValidNet[net];
	AttachPortHints[p] @ NetTakeDrop[Drop, net, startend]
];

DeclareArgumentCount[NetTake, 2];
DeclareArgumentCount[NetDrop, 2];

(***************************)
(* NetChain implementation *)
(***************************)

$LayerNameP = _String | _Integer | _NetPort;

$tof = <|Take -> NetTake, Drop -> NetDrop|>;

General::invchlayername = "`` is not the name of layer."
General::invportname = "`` is not the name of an input or output port of layer ``."
NetTakeDrop[f_, nc_NetChain, spec:$LayerNameP | {$LayerNameP, $LayerNameP}] := Scope[
	data = NData[nc];
	layers = data["Nodes"];
	first = First[layers];
	last = Last[layers];
	len = Length[layers];
	keys = Keys[layers];

	spec = ReplaceAll[spec, NetPort[{a_, b_}] :> NetPort[a, b]];
	spec = ReplaceAll[spec, p:NetPort[a_] :> With[{path = ToNetPath[nc, a]},
		If[MatchQ[path, NetPath["Inputs", _]],
			0.5,
			If[MatchQ[path,	NetPath["Outputs", _]],
				len + 0.5,
				ThrowFailure["invportspec", p]
			]
		]
	]];
	spec = ReplaceAll[spec, NetPort[layer_, port_String] :> With[
		{
			index = If[StringQ[layer], IndexOf[keys, layer, ThrowFailure["invchlayername", layer]], layer]
		},
		Which[
			KeyExistsQ[layers[[index]]["Inputs"], port],
				index - 0.5,
			KeyExistsQ[layers[[index]]["Outputs"], port],
				index + 0.5,
			True,
				ThrowFailure["invportname", port, layer]
		]		
	]];
	spec = ReplaceAll[spec, str_String :> IndexOf[keys, str, ThrowFailure["invchlayername", str]]];
	(* The replacement rules above do the following:
	1. standardise the form of the two argument NetPort
	2. replace *net* input ports with an index of 0.5 & *net* output ports with an index of (len + 0.5)
	3. replace *layer* input ports with an index of (layer index - 0.5) and *layer* output ports with and index of (layer index + 0.5)
	4. replace any named layers with the index of that layer in the list of layers
	*)
	If[ListQ[spec],
		spec = ReplaceAll[spec, n_Integer/;Negative[n] :> len + 1 + n];
		(* replace any negative indices with the corresponding index counting from the back *)
		spec = Function[{Ceiling @ First @ #, Floor @ Last @ #}] @ spec;
		(* ^ the start index of the take gets rounded up and the end index gets rounded down.
		In conjunction with the above replacement rules this has the following implications:
		1. if the start or end for the take is specified as a (positive) index or as a named layer, the index will be unchanged
		2. if the start is specified as the input port of the net, the index will be that of the first layer
		3. if the end is specified as the output port of the net, the index will be that of the last layer
		4. if the start is specified as the input port of a layer, the index will the that of the same layer
		5. if the end is specified as the input port of a layer, the index will be that of the next layer
		6. if the start is specified as the output port of a layer, the index will be that of the next layer
		7. if the end is specified as the output port of a layer the index will be that of the same layer
		*)
		If[First @ spec > Last @ spec && f === Take, ThrowFailure["netempty", NetChain]];
	,
		spec = Floor @ spec;
		If[spec == 0 && f === Take, ThrowFailure["netempty", NetChain]];
	];

	newNodes = UnsafeQuietCheck[f[layers, spec], $Failed]; 
	If[FailureQ[newNodes] || Length[newNodes] == 0, ThrowFailure["invchtakespec", $tof @ f]];
	newKeys = Keys[newNodes];
	ioSpecs = chainIOSpecs[data, First[newKeys] == First[keys], Last[newKeys] == Last[keys]];
	If[DigitStringVectorQ[keys], newNodes = NumberedAssociation[Values[newNodes]]];
	sharedArrays = GetSharedArrays[data];
	toNetChain[newNodes, ioSpecs, sharedArrays]
];

General::invchtakespec = "Invalid `` specification."
NetTakeDrop[f_, _NetChain, _] := ThrowFailure["invchtakespec", $tof @ f];


(***************************)
(* NetGraph implementation *)
(***************************)

NetGraph::notsupp = "`` is not supported by NetGraph."
NetTakeDrop[Drop, _NetGraph, _] := ThrowFailure[NetGraph::notsupp, Drop];

NetTakeDrop[Take, ng_NetGraph, a_]:= NetTakeDrop[Take, ng, {All, a}];

NetTakeDrop[Take, ng_NetGraph, {a_, b_}]:= Scope[
	net = NData[ng];

	$nodes = net["Nodes"];
	edges = net["Edges"];
	$inputs = InputNames[net];
	$outputs = OutputNames[net];
	
	$starting = procTakeStartElem[a];
	$ending = procTakeEndElem[b];

	{$starting, $ending} = Replace[{$starting, $ending},
		{{NetPath["Nodes", name_, "Inputs", _]}, {NetPath["Nodes", name_, "Outputs", _]}} ->
			 {{NetPath["Nodes", name]}, {NetPath["Nodes", name]}}];
	(* ^ handle special case of selecting a node using the input and output ports of that node as start and end *)

	graph = FullNetPathGraph[edges, $nodes];
	sourceNodes = Cases[GraphSinks[graph], NetPath["Nodes", _]];

	(* get component *)
	in = VertexOutComponent[graph, $ending];
	out = VertexInComponent[graph, $starting];

	newGraph = Intersection[in, Union[out, sourceNodes]];
	$newNodes = Cases[newGraph, NetPath["Nodes", v_] :> v];
	
	(* these are for starts of the form NetPort["layer3",out], where we want
	to map them into a new uniquely named inputs (see 346450) *)
	danglingInputRules = Cases[$starting, np:NetPath["Nodes", node_, "Outputs", output_] :> 
		Rule[np, toDanglingInputPort[node, output]]];
	$starting = $starting /. danglingInputRules;

	(* these are for ends of the form NetPort["layer3",in], where we want
	to map them into a new uniquely named outputs *)
	danglingOutputRules = Cases[$ending, np:NetPath["Nodes", node_, "Inputs", input_] :> 
		Rule[np, toDanglingOutputPort[node, input]]];
	$ending = $ending /. danglingOutputRules;

	If[danglingInputRules =!= {}, edges = edges /. danglingInputRules];
	If[danglingOutputRules =!= {}, edges = edges /. danglingOutputRules];
	$nodes = KeySelect[$nodes, MemberQ[$newNodes, #]&];
	edges = Select[edges, keepEdgeQ];

	$nodes = KeySelect[$nodes, keepNodeQ[#, edges]&];
	(* ^ keepNodeQ checks that each node has at least one edge to something else.
	This is needed because sometimes we have a source node that isn't connected
	to anything in the new graph, for example a ConstantArrayLayer. See 349847. *)

	makeSubNetGraph[net, $nodes, edges]
]

toDanglingInputPort[node_, "Output"] := NetPath["Inputs", node];
toDanglingInputPort[node_, out_] := NetPath["Inputs", node <> "/" <> out];
toDanglingOutputPort[node_, "Input"] := NetPath["Outputs", node];
toDanglingOutputPort[node_, in_] := NetPath["Outputs", node <> "/" <> in];

procTakeStartElem[elem_] := 
	Block[{$io = $inputs, $type = "Inputs"}, Flatten @ List @ procTakeElem[elem]];

procTakeEndElem[elem_] := 
	Block[{$io = $outputs, $type = "Outputs"}, Flatten @ List @ procTakeElem[elem]];

procTakeElem[e_List] := procTakeElem /@ e;
procTakeElem[All] := NetPath[$type, #]& /@ $io;

General::invportspec = "The port specification `` is invalid."
procTakeElem[p:NetPort[name_String]] :=
	If[MemberQ[$io, name], 
		NetPath[$type, name],
		ThrowFailure["invportspec", p]
	];

procTakeElem[NetPort[a_, b_]] := procTakeElem[NetPort[{a, b}]];

procTakeElem[p:NetPort[{part_Integer | part_String, name_String}]] := Scope[
	part = toVertex[part];
	inputs = $nodes[[part, "Inputs"]];
	outputs = $nodes[[part, "Outputs"]];
	NetPath["Nodes", part, Which[
		KeyExistsQ[inputs, name], "Inputs",
		KeyExistsQ[outputs, name], "Outputs",
		True, ThrowFailure["invportspec", p]
	], name]
];

toVertex[n_ ? PositiveMachineIntegerQ] := toVertex[IntegerString[n]];
toVertex[name_String /; KeyExistsQ[$nodes, name]] := name;
General::invnetvertex = "The specified vertex `` does not exist.";
toVertex[spec_] := ThrowFailure["invnetvertex", spec];

procTakeElem[part_] := NetPath["Nodes", toVertex[part]];

keepEdgeQ[NetPath["Nodes", name1_, __] -> NetPath["Nodes", name2_, __]] := 
	MemberQ[$newNodes, name1] && MemberQ[$newNodes, name2]

keepEdgeQ[end:NetPath["Outputs", _] -> NetPath["Nodes", name_, __]] :=
	MemberQ[$newNodes, name] && MemberQ[$ending, end | NetPath["Nodes", name]];

keepEdgeQ[NetPath["Nodes", name_, __] -> start:NetPath["Inputs", _]] :=
	MemberQ[$newNodes, name] && MemberQ[$starting, start | NetPath["Nodes", name]];

keepSourceQ[NetPath["Nodes", name_, ___]] := MemberQ[$newNodes, name];
keepSourceQ[np:NetPath["Inputs", _]] := MemberQ[$starting, np];
keepSourceQ[_] := False;

keepEdgeQ[pat_] := False

keepNodeQ[node_, edges_] := MatchQ[edges, KeyValuePattern[NetPath["Nodes", node, __] -> _] | KeyValuePattern[_ -> NetPath["Nodes", node, __]]];
