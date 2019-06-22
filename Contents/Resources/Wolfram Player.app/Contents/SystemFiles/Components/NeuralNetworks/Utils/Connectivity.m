Package["NeuralNetworks`"]



PackageScope["InitialVertices"]
PackageScope["FinalVertices"]

InitialVertices[graph_] := Pick[VertexList[graph], VertexInDegree[graph], 0];
FinalVertices[graph_] := Pick[VertexList[graph], VertexOutDegree[graph], 0];


PackageExport["UnfoldStarGraph"]

SetUsage @ "
UnfoldStarGraph[net$, {name$1, name$2, $$}] takes a graph with multiple inputs and one output and \
returns the 'legs' as separate chains, along with the 'star' layer at the center that connects the \
various legs. 
* The name$i are the names of the input ports of the net, which will usually be a NetGraph.
* The return value is {arms$, starInputs$, star$, epilog$}.
* arms$ and epilog$ are lists of nodes, connected output-to-input.
* starInputs$ is {n$1, n$2}, where n$i is the name of the input of the star that is computed by arm number i$.
* star$ is a single layer association."

UnfoldStarGraph::nostar = "No star found."
UnfoldStarGraph::badstar = "No simple star epilog, another star after `` occurs at ``."
UnfoldStarGraph::multistar = "Multiple stars found (``)."

UnfoldStarGraph[net_NetP, iports_List] := CatchFailure @ Scope[
	$path = NetPath[]; $net = net;
	CollectTo[{$GraphNodes, $GraphEdges}, BuildPortGraph[net]];
	$predFunc = Merge[$GraphEdges, Identity];
	$postFunc = Merge[Reverse[Flatten @ $GraphEdges, 2], Identity];
	$layerNodeQ = ConstantAssociation[$GraphNodes, True];
	ipaths = Thread @ NetPath["Inputs", iports];
	{arms, starPaths, starDests} = Transpose[collectBranch /@ ipaths];
	If[Not[Equal @@ starPaths], ThrowFailure["multistar", starPaths]];
	starPath = First[starPaths];
	If[starPath === Null, ThrowFailure["nostar"]];
	star = $net @@ First[starPaths];
	{epilog, epath, tmp} = collectBranch[starPath];
	If[epath =!= Null, ThrowFailure["badstar", starPath, epath]];
	{arms, starDests, star, epilog}
]

UnfoldStarGraph::div = "Path diverges at ``, with next paths being ``."

collectBranch[path_] := Scope[
	nodes = edges = {};
	starPath = starDest = Null;
	opath = path;
	Do[
		DoWhile[
			opath = path; next = $postFunc[path];
			If[MissingQ[next], Goto[Done]];
			If[Length[next] > 1, ThrowFailure["div", path, next]];
			path = First[next]
			,
			!KeyExistsQ[$layerNodeQ, path]
		];
		prev = $predFunc[path];
		node = $net @@ path;
		If[Length[prev] =!= 1, 
			starPath = path;
			starDest = Last[opath];
			Goto[Done];
		];
		AppendTo[nodes, node];
	,
		Infinity
	]; 
	Label[Done];
	{nodes, starPath, starDest}
];

	



PackageExport["LayerDependencyGraph"]

SetUsage @ "
LayerDependencyGraph[net$] gives a Graph, where vertices represent layers and edges represent connections between layers.
* The vertices are NetPath[$$] specifications that correspond to positions in the original net association.
* Edges p$1->p$2 exist when vertex p$2 takes as input one of the outputs of p$1.
* Containers do not appear in the dependency graph.
* The dependency graph does not represent which inputs and outputs when a layer has multiple inputs or outputs, merely \
whether there is any connection at all."

LayerDependencyGraph[net_NetP] := Scope[
	$path = NetPath[];
	CollectTo[{$GraphNodes, $GraphEdges}, BuildPortGraph[net]];
	$predFunc = Merge[$GraphEdges, Identity];
	virtNodes = Complement[DeepCases[$GraphEdges, _NetPath], $GraphNodes];
	$virtQ = ConstantAssociation[virtNodes, True];
	edges = Flatten @ Map[layerPredecessorEdges, $GraphNodes];
	Graph[
		$GraphNodes, Reverse[edges, 2],
		VertexLabels -> Placed["Name", Tooltip],
		GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Left},
		ImageSize -> {800}
	]
];

layerPredecessorEdges[layer_] := 
	Thread @ Rule[layer, 
		Flatten @ ReplaceRepeated[
			$predFunc[layer],
			p_NetPath ? $virtQ :> Lookup[$predFunc, p, {}]
		]
	];


PackageExport["PortConnectivityGraph"]

SetUsage @ "
PortConnectivityGraph[net$] gives a bipartite-ish Graph, where vertices represent arrays or layers, and edges represent flow.
* The vertices are NetPath[$$] specifications that correspond to positions in the original net association.
* Vertices can represent an input or output of a layer, e.g. NetPath[1, 'Inputs', 'Input'], or a layer itself, e.g. NetPath[1].
* Operators and containers are also present in the graph, but 'in parallel' to their contents.
* Edges flow typically from inputs to layers to outputs to inputs to layers to outputs etc.
* The graph is not strictly bipartite because outputs are themselves connected to inputs."

$includeContainers = False;

PortConnectivityGraph[net_NetP] := Scope[
	$path = NetPath[];
	$includeContainers = True;
	CollectTo[{$GraphNodes, $GraphEdges}, BuildPortGraph[net]];
	Graph[$GraphNodes, Reverse[Flatten @ $GraphEdges, 2], 
		VertexLabels -> Placed["Name", Tooltip],
		GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Left},
		ImageSize -> {800}
	]
];

PackageScope["PortConnectivity"]

$causal = True;
PortConnectivity[net_NetP, causal_:True] := Scope[
	$path = NetPath[]; $causal = causal;
	CollectTo[{$GraphNodes, $GraphEdges}, BuildPortGraph[net]];
	{$GraphNodes, Flatten @ $GraphEdges}
];	


PackageScope["BuildPortGraph"]

DeclareMethod[BuildPortGraph, BuildLayerPortGraph, BuildContainerPortGraph, BuildOperatorPortGraph];

BuildLayerPortGraph[node_] := (
	SowGraphNode[node];
	SowStateEdges[node];
	SowSelfEdges[node];
);

SowStateEdges[node_] := 
	If[$causal && KeyExistsQ[node, "States"],
		BagPush[$GraphEdges, List[
			Outer[Rule, OutputPaths[node], StatePaths[node]],
			Outer[Rule, StatePaths[node], InputPaths[node]]
		]]
	];

SowSelfEdges[assoc_] :=
	BagPush[$GraphEdges, List[
		Thread[OutputPaths[assoc] -> $path],
		Thread[$path -> InputPaths[assoc]]
	]];

BuildContainerPortGraph[assoc_] := (
	If[$includeContainers, SowSelfEdges[assoc]];
	MapAtFields["Nodes", BuildPortGraph, assoc];
	Scan[SowPortGraphEdge, assoc["Edges"]];
)

BuildOperatorPortGraph[node_] := Scope[
	If[SubNetPaths[node] === {}, BuildLayerPortGraph[node]; Return[]];
	(* ^ some layers have dynamic operator-ness, e.g. MXLayer *)
	subouts = subins = subpaths = {};
	If[$includeContainers, SowSelfEdges[node]];
	If[$causal, 
		SowStateEdges[node];
		ScanSubNets[
			Function[subnode,
				BuildPortGraph[subnode];
				AppendTo[subpaths, $path];
				AppendTo[subouts, OutputPaths[subnode]];
				AppendTo[subins, InputPaths[subnode]];
			],
			node
		];
		BagPush[$GraphEdges, List[
			Outer[Rule, OutputPaths[node], Flatten[subouts]],
			Outer[Rule, Flatten[subins], InputPaths[node]]
		]];	(* 
			our outputs depend on the subnet's outputs
			the subnets inputs depend on our inputs
		*)
	,
		BuildLayerPortGraph[node]
	];
];


PackageScope["SowGraphNode"]

SowGraphNode[node_] := BagPush[$GraphNodes, $path];


PackageScope["SowPortGraphEdge"]

SowPortGraphEdge[edge_] := BagPush[$GraphEdges, PrefixPorts[edge]];
