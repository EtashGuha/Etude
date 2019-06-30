Package["NeuralNetworks`"]


PackageExport["NetGraph"]

SetupGenericDispatch[NetGraph, False];

(ng_NetGraph ? System`Private`HoldEntryQ) := 
	UseMacros @ RuleCondition @ CatchFailureAsMessage[NetGraph, make[ng]];

SetHoldAll[make];

make[NetGraph[chain_NetChain]] := 
	ConstructNet @ NetChainToNetGraph @ chain;

make[NetGraph[layers_Association | layers_List, conns_List, rules___Rule]] :=
	toNetGraph[ToContainerLayers[layers], Flatten @ conns, {rules}];

make[ng:NetGraph[<|"Type" -> "Graph", ___|>]] :=
	UpgradeAndSeal11V0Net[ng];

make[ng:NetGraph[<|"Type" -> "Graph", ___|>, _Association]] :=
	SealNet[ng];

make[ng_NetGraph] /; PatternFreeQ[Unevaluated[ng]] :=
	(CheckArgumentCount[ng, 2, 2]; $Failed);

make[_] := Fail;

PackageExport["NetParallel"]

NetParallel[layers___] := Scope[
	layers = Map[ToLayer, {layers}]; 
	len = Length[layers];
	NetGraph[Append[$Raw /@ layers, CatenateLayer[]], Thread[Range[len]->(len+1)]]
];


PackageScope["toNetGraph"]

NetGraph::acyclic = "The connectivity of layers within a NetGraph must be acyclic."
General::netempty = "`` objects must contain at least one layer."

toNetGraph[nodes_Association, edges_List, rules_List, sharedArrays_:<||>] := Scope[
	$currentNodes = $nodes = nodes; $keys = Keys[$nodes];
	$currentNodeName = "node"; (* <- err reporting shared with NetChain *)

	(* Argument checking *)
	If[Length[nodes] === 0, ThrowFailure["netempty", NetGraph]];
	If[!ListQ[edges] || !AllTrue[edges, RuleQ[#] || MatchQ[#, $Raw[_]]&],
		ThrowFailure[NetGraph::nginvedgespec, edges]
	];

	(* mapping from layer input and output ports to their types *)
	$itypes = $otypes = $stypes = $istates = $inputs = $outputs = $alltypes = Association[];
	KeyValueScan[collectPortTypes, $nodes];

	(* edge processing *)
	edges = canonicalizeEdges[edges];
	edges = edges /. {NetPort[a_, b_] :> NetPort[ToList[a,b]], NetPort[{a_}] :> NetPort[a]};
	If[!FreeQ[edges, NetPort[{_, i_Integer}]], primeMultiports[edges]];
	(* ^ scan for numbered inputs that correspond to unexpanded multiports *)
	edges = edges /. NetPort[{a_, b_Integer ? Positive}] :> NetPort[{a, IntegerString[b]}];
	edges = procEdge /@ edges;

	(* find in and out ports that aren't bound *)
	ofree = Complement[Keys[$otypes], Flatten @ Values @ edges];
	ifree = Complement[Keys[$itypes], Flatten @ Keys @ edges];

	If[!DuplicateFreeQ[edges[[All, 1]]],
		edges = Flatten @ Map[handleDuplicateEdge, GatherBy[edges, First]];
	];

	(* if these types clash that'll get picked up in the inference step *)
	AssociateTo[$inputs,  KeyMap[Last] @ KeyTake[$itypes, ifree]];
	(* detect collision on the Output port that must be unique *)
	implicitoutputs = KeyMap[Last] @ KeyTake[$otypes, ofree];
	conflicts = Intersection[Keys[implicitoutputs], Keys[$outputs]];
	If[Length[conflicts] > 0,
		General::outportconfl = "Multiple vertices are connected to `1`.";
		ThrowFailure["outportconfl", NetPort[First @ conflicts]];
	];
	AssociateTo[$outputs, implicitoutputs];

	(* add on connections to virtual ports for the unbound inputs and outputs *)
	edges = ToList[
		edges, 
		virtIn /@ ifree, 
		Flatten @ KeyValueMap[virtOut, GroupBy[ofree, Last]]
	];

	(* sort the nodes and edges into topological order, so that when we scan
	them to feed to MXNet a node can assume its inputs have already been scanned. *)
	{$nodes, edges} = sortNodesAndEdges[$nodes, edges];

	(* remove from interiorStates all states that have been fed via connections *)
	If[$istates =!= {},
		(* TODO: update ToPlan to depend on the fact that interiorstates is free of fed states *)
		fedStates = Cases[Keys @ edges, NetPath["Nodes", _, "States", _]];
		If[fedStates =!= {},
			$istates = DeleteCases[$istates, Alternatives @@ fedStates];
		];
	];

	(* encoders and decoders are only on the outside.. *)
	$nodes = StripCoders[$nodes];

	(* shared arrays are removed from nodes and put into our own SharedArrays param *)
	{$nodes, sarrays} = HoistSharedArrays[$nodes];

	(* handle custom type specifications *)
	scanTypeHints[rules, $inputs, $outputs];

	(* make sure that inputs that are used first come earlier. should not matter but
	a temporary workaround for a problem tracking the input vs label in NetTrain. *)
	$inputs = KeySortBy[$inputs, FirstPosition[edges, #]&];
	
	(* build the association*)
	assoc = Association[{
		"Type" -> "Graph",
		"Inputs" -> $inputs, 
		"Outputs" -> $outputs,
		"Nodes" -> $nodes,
		"Edges" -> edges,
		If[$istates === <||>, Nothing, "InteriorStates" -> $istates],
		If[sarrays === <||>, Nothing, "SharedArrays" -> sarrays]
	}];

	assoc = AttachSharedArrays[assoc, sharedArrays]; (* <- existing shared arrays etc *)

	(* make sure none of the nodes have $Multiport keys, still *)
	CheckForMultiports[assoc];

	ConstructWithInference[NetGraph, assoc]
];

(* Flatten and then factorize edges *)
canonicalizeEdges[edges_] := Module[{ports, flatedges},
	If[MemberQ[edges, $Raw[_]], Return[edges]];
	ports = Union @ Flatten[edges /. Rule -> List];
	flatedges = Cases[edges, _Rule, Infinity];
	(* Remove the previous/next rules that were catched by the Cases[..., Infinity] anyway *)
	flatedges = Replace[flatedges,
		Rule[in_, out_] :> Rule[lastInRule[in], firstInRule[out]]
	, {1}];
	(* Flatten when several input ports / output ports*)
	flatedges = ReplaceRepeated[flatedges,{
		Rule[ins_List, out_] :> Sequence @@ Map[lastInRule[#] -> out&, ins],
		Rule[in_, outs_List] :> Sequence @@ Map[in -> firstInRule[#]&, outs]
	}];
	(* Factor by output port *)
	Map[
		Function[With[
		{inports = Cases[flatedges, Rule[in_,#] :> in]},
		Switch[Length[inports],
			0, Nothing,
			1, First[inports] -> #,
			_, inports -> #
			]
		]],
		ports
	] // SortBy[Last]
];
(* Remove rules in input ports, to handle {1 -> {2 -> {3, 4 -> 5} -> 6, 7} -> {8, 9} -> 10} *)
lastInRule[rule_Rule] := ReplaceRepeated[rule, Rule[_,out_] :> out];
lastInRule[x_] := x;
firstInRule[rule_Rule] := ReplaceRepeated[rule, Rule[in_,_] :> in];
firstInRule[x_] := x;

primeMultiports[edges_] :=
	KeyValueMap[
		checkMultiportIn,
		Merge[
			Cases[edges, e:(_ -> NetPort[{dst_, i_Integer}]) :> 
				($lastEdge = e; toVertex[dst] -> i)
			],
			Identity
		]
	]

checkMultiportIn[dst_, ints_] := Scope[
	itypes = $nodes[dst, "Inputs"];
	len = Length[ints];
	Which[
		Sort[ints] =!= Range[len],
			NetGraph::noncontigmi = "Non-contiguous port numbers provided to ``.";
			ThrowFailure["noncontigmi", NodeForm[dst]],
		MatchQ[itypes, <|$Multiport -> _|>], 
			SetMultiport[$nodes[dst, "Inputs"], len];
			path = NetPath["Nodes", dst, "Inputs", $Multiport];
			$itypes[path] =.; $alltypes[path] =.;
			Do[
				path[[-1]] = IntegerString[i];
				$alltypes[path] = $itypes[path] = Last[itypes],
				{i, len}
			];
		,
		!DigitStringKeysQ[itypes],
			NetGraph::nonmi = "Numbered input ports cannot be used with `` (``).";
			ThrowFailure["nonmi", NodeForm[dst], MsgForm[$nodes[dst]]],
		Length[ints] =!= Length[itypes],
			NetGraph::mismatchmi = "The `` (``) expected `` inputs but received `` inputs.";
			ThrowFailure["mismatchmi", NodeForm[dst], MsgForm[$nodes[dst]], len, Length[itypes]]
	]
];


SetHoldRest[scanTypeHints];

scanTypeHints[{}, _, _] := Null;

scanTypeHints[rules_, inputs_, outputs_] := Scope[
	$CurrentArgumentHead = NetGraph;
	Scan[
		Match[#,
			(name_String -> spec_) :> Which[
				KeyExistsQ[inputs, name], 
					setType[inputs[name], ParseInputSpec, spec],
				KeyExistsQ[outputs, name],
					setType[outputs[name], ParseOutputSpec, spec],
				True,
					ThrowFailure["netinvgport", name]
			],
			(NetPath["Inputs", name_] -> spec_) :> 
				setType[inputs[name], ParseInputSpec, spec],
			(NetPath["Outputs", name_] -> spec_) :> 
				setType[outputs[name], ParseOutputSpec, spec],
			ThrowFailure["invtypehint", #]
		]&,
		rules
	];
];

NetGraph::invtypehint = "`` should be a rule of the form \"port\" -> type."

SetHoldFirst[setType];

setType[slot:(_[name_]), func_, spec_] := If[
	MissingQ[slot], ThrowFailure["netinvgport", name], 
	Set[slot, func[name, slot, spec]]
];

NetGraph::netinvgport = "`` is neither a valid input or output port for the given NetGraph.";


PackageScope["CheckForMultiports"]

General::indmultiport = "The `` has an indeterminate number of input ports. Please ensure it is connected to at least one other layer or manually specify its input ports using \"Inputs\" -> {...}."

CheckForMultiports[assoc_] := 
	ScanNodes[
		If[MatchQ[Inputs[#], <|$Multiport -> _|>], ThrowFailure["indmultiport", MsgForm[$path]]]&,
		assoc
	];

Clear[handleDuplicateEdge];

handleDuplicateEdge[{rule_}] := rule;

(* this is for non-expanded multiports *)
handleDuplicateEdge[list_List] /; list[[1,1,-1]] === $Multiport := Scope[
	dst = list[[1,1]]; to = dst[[2]];
	SetMultiport[$nodes[to, "Inputs"], Length[list]];
	IMap[
		ReplacePart[dst, -1 -> IntegerString[#1]] -> #2&,
		list[[All,2]]
	]
];

(* this is for already-expanded multiports *)
(* jeromel: This code should not been hit since canonicalizeEdges was introduced *)
handleDuplicateEdge[list_List] /; list[[1,1,-1]] === "1" := Scope @ RuleCondition[
	dst = list[[1,1]]; to = dst[[2]];
	ins = $nodes[to, "Inputs"];
	If[!DigitStringKeys[ins], Return[Fail]];
	If[Length[ins] =!= Length[list], 
		NetGraph::invincnt = "The `` expected `` inputs but `` edges were connected to it.";
		ThrowFailure["invincnt", NetPathString @ Take[dst, 2], Length[ins], Length[list]]
	];
	newEdges = IMap[
		ReplacePart[dst, -1 -> IntegerString[#1]] -> #2&,
		list[[All,2]]
	];
	ifree ^= Complement[ifree, Keys @ newEdges]; 
	(* ^ make sure the implied multiports aren't considered to be danglers *)
	newEdges
];

NetGraph::dupedges =  "Multiple vertices have been connected to the ``.";

handleDuplicateEdge[list_List] := 
	ThrowFailure["dupedges", NetPathString @ list[[1,1]]]

handleDuplicateEdge[other_] := Panic[];

collectPortTypes[i_, assoc_] := Scope[
	AssociateTo[$itypes, it = iosTypes[i, "Inputs", assoc]];
	AssociateTo[$otypes, ot = iosTypes[i, "Outputs", assoc]];
	AssociateTo[$stypes, st = iosTypes[i, "States", assoc]];
	AssociateTo[$istates, interiorStateRules["Nodes"][i, assoc]];
	AssociateTo[$alltypes, Join[it, ot, st]];
];

iosTypes[i_, field_, layer_] := KeyValueMap[
	NetPath["Nodes", i, field, #1] -> #2&, 
	Lookup[layer, field, <||>]
]

PackageScope["interiorStateRules"]

interiorStateRules[prefix_][key_, assoc_] := Scope[
	states = GetInteriorStates[assoc];
	If[states === <||>, Return[{}]];
	name = If[IntStringQ[key], FromDigits[key], key];
	path = NetPath[prefix, key];
	KeyValueMap[Prepend[#1, name] -> Join[path, #2]&, states]
];



PackageScope["makeSubNetGraph"]

(* this takes care of remapping pure int keys (i.e. keys from a NetGraph
that used a list spec), as well as preserving port hints (i.e. Encoders and Decoders *)
makeSubNetGraph[oldnet_, nodes_, edges_, shared_:<||>] := Scope[
	{ledges, redges} = KeysValues[edges];
	{ipaths, opaths} = PortTypes[oldnet];
	hints = Join[
		Cases[ipaths, Rule[port_, type_] /; !FreeQ[redges, port] :> 
			Rule[port, $Raw[type]]],
		Cases[opaths, Rule[port_, type_] /; !FreeQ[ledges, port] :> 
			Rule[port, $Raw[type]]]
	];
	If[DigitStringKeysQ[oldnet["Nodes"]],
		{nodes, edges} = sortNodesAndEdges[nodes, edges];
		(* ^ even though toNetGraph will do this, we do it now so that the remapping is in nice
		topo order *)
		{nodes, mapping} = RemapKeys[nodes];
		rule = NetPath["Nodes", node_, rest__] :> RuleCondition @ NetPath["Nodes", mapping @ node, rest];
		edges = edges /. rule;
		hints = hints /. rule;
	];
	toNetGraph[nodes, $Raw /@ edges, hints, JoinSharedArrays[GetSharedArrays[oldnet], shared]]
];

sortNodesAndEdges[nodes_, edges_] := Scope[
	graph = NetPathGraph[edges];
	If[!AcyclicGraphQ[graph] || !LoopFreeGraphQ[graph], ThrowFailure["acyclic"]];
	nodeOrder = TopologicalSort @ graph;
	nodes = KeySortBy[nodes, -First[FirstPosition[nodeOrder, #]]&];
	edges = SortBy[edges, -First[FirstPosition[nodeOrder, Take[Last[#], 2]]]&];
	{nodes, edges}
];


PackageScope["NetPathGraph"]

NetPathGraph[edges_] := Graph[
	edges[[All, All, 1;;2]], 
	VertexLabels -> "Name"
];


PackageScope["FullNetPathGraph"]

FullNetPathGraph[edges_, nodes_] := Scope[
	rules = Flatten[{
		edges,
		KeyValueMap[Function[
			node = NetPath["Nodes", #1];
			{Thread[node -> Thread[NetPath["Nodes", #1, "Inputs", InputNames[#2]]]], 
			 Thread[Thread[NetPath["Nodes", #1, "Outputs", OutputNames[#2]]] -> node]}], 
			nodes
		]
	}];
	Graph[rules, VertexLabels -> Placed["Name",Tooltip]]
];

(* TODO: support e.g.
 NetGraph[<|"cat" -> CatenateLayer[]|>, {NetPort["A"] -> NetPort["cat", 1], NetPort["B"] -> NetPort["cat", 2]}]
 *)

(* when someone specifies 1 -> 2, we must lookup the first input of 1 and 
	the first output of 2 *)
toPortAuto[node_, type_] :=
	toPort[node, type, First @ Keys @ PartElse[$nodes, toVertex[node], type, vertexPanic[node]]];

(* check the port exists and canonicalize it *)
General::invport = "The `` does not exist.";
toPort[node_, type_, name_] := Scope[
	vert = toVertex[node];
	path = NetPath["Nodes", vert, type, name];
	(* check the provided path actually exists *)
	If[MissingQ[$nodes @@ Rest[path]],
		path2 = NetPath["Nodes", vert, "States", name];
		If[MissingQ[$nodes @@ Rest[path2]],
			ThrowFailure["invport", NetPathString @ path],
			path = path2;
		];
	];
	path
];
	
Clear[toVertex];
toVertex[n_ ? PositiveMachineIntegerQ] := Block[{ns = IntegerString[n]}, 
	If[KeyExistsQ[$nodes, ns], ns, UnsafeQuietCheck[$keys[[n]], vertexPanic[n]]]
];
toVertex[name_String /; KeyExistsQ[$nodes, name]] := name;
toVertex[spec_] := vertexPanic[spec];

General::invnetvert = "The vertex ``, specified in the edge ``, does not exist.";
General::invnetvert2 = "The specified vertex `` does not exist.";
vertexPanic[spec_] := ThrowFailure[NetGraph::invnetvert, spec, $lastEdge];

toLPort[NetPort[{p:NetPathElemP, name_String}]] := toPort[p, "Outputs", name];
toLPort[p:NetPathElemP] := toPortAuto[p, "Outputs"];
toLPort[NetPort[name_String]] := toGraphPort[$otypes, NetPath["Inputs", name]];

NetGraph::invedgesrc = "`` is not a valid source for ``."
toLPort[e_] := ThrowFailure[NetGraph::invedgesrc, First[$lastEdge], Last[$lastEdge]];

toRPort[NetPort[{p:NetPathElemP, name_String}]] := toPort[p, "Inputs", name];
toRPort[p:NetPathElemP] := toPortAuto[p, "Inputs"];
toRPort[NetPort[name_String]] := toGraphPort[$itypes, NetPath["Outputs", name]];

NetGraph::invedgedst = "`` is not a valid destination for ``."
toRPort[e_] := ThrowFailure[NetGraph::invedgedst, Last[$lastEdge], First[$lastEdge]];

SetHoldAll[toGraphPort];
toGraphPort[typesym_, p_] := (
	(* TODO: do we even need typesym any more? 
	is itypes only distinguished from otypes for the purposes of taking Keys
	to know the dangling ports? we should then unify the types and separately
	collect the keys. *)
	If[!KeyExistsQ[typesym, p], typesym[p] = TypeT]; 
	If[!KeyExistsQ[$alltypes, p], $alltypes[p] = TypeT]; 
	p
);

Clear[procEdge];

procEdge[$Raw[edge_]] := (
	Cases[edge, Alternatives[
		NetPath["Outputs", name_] /; AppendTo[$outputs, name -> TypeT],
 		NetPath["Inputs", name_] /; AppendTo[$inputs, name -> TypeT]],
		Infinity];
	edge
);

multiQ[inputs_] := MatchQ[inputs, <|$Multiport -> _|>];
procEdge[from_List -> to:NetPathElemP] /; multiQ[$nodes[[to, "Inputs"]]] := Scope[
	$lastEdge = from -> to;
	to = toVertex[to];
	(* ^ so ints become strings and [...] setting works *)
	SetMultiport[$nodes[to, "Inputs"], Length[from]];
	(* ^ create new inputs for this layer *)
	toRemove = NetPath["Nodes", to, "Inputs", $Multiport];
	KeyDropFrom[$itypes, toRemove]; 
	KeyDropFrom[$alltypes, toRemove];
	(* ^ make sure that multiport placeholder is removed and doesn't dangle *)
	collectPortTypes[to, $nodes[[to]]];
	(* ^ harvest the types of all its new input ports *)
	Sequence @@ IMap[procEdge[#2 -> NetPort[{to, IntegerString[#1]}]]&, from]
	(* ^ do edges one-by-one *)
];

(* {1, 2} -> LossLayer, where LossLayer has inputs "Input", "Target" *)
threadableQ[sources_, inputs_] := AssociationQ[inputs] && Length[inputs] == Length[sources]
procEdge[from_List -> to:NetPathElemP] /; threadableQ[from, $nodes[[to, "Inputs"]]] := 
	Sequence @@ MapThread[procEdge[#1 -> NetPort[{to, #2}]]&, {from, InputNames[$nodes[[to]]]}];

procEdge[from_ -> to_List] := Sequence @@ Map[procEdge[from -> #]&, to];

(* a -> b -> c chain *)
procEdge[from_ -> (mid_ -> to_)] := Sequence[
	procEdge[from -> mid],
	procEdge[mid -> to]
];

(* plain case *)
procEdge[from_ -> to_] := Scope[
	$lastEdge = from -> to;
	fpath = toLPort[from];  ftype = $alltypes[fpath]; 
	tpath = toRPort[to];    ttype = $alltypes[tpath];
	If[!UnifiableQ[ftype, ttype],
		edgeTypeError[fpath, tpath, ftype, ttype]
	];
	If[MatchQ[fpath, NetPath["Inputs", _]],   $inputs[Last[fpath]] ^= ttype]; 
	If[MatchQ[tpath, NetPath["Outputs", _]], $outputs[Last[tpath]] ^= ftype]; 
	tpath -> fpath
];

General::nginvedgespec = "Second argument should be a list of rules.";
General::invedge = "`` is not a valid edge specification.";
procEdge[e_] := ThrowFailure[NetGraph::invedge, e];

PackageScope["edgeTypeError"]
PackageScope["$currentNodes"]
PackageScope["$currentNodeName"]

$currentNodeName = "node";
(* also used by NetChain *)

General::ninctyp = "Incompatible ``s for `` (``) and `` (``).";
General::ninctyp2 = "Incompatible ``s for output of ``, ``, and input to ``, ``; `` is not compatible with ``, respectively.";
edgeTypeError[NetPath["Nodes", snode_, "Outputs", "Output"], NetPath["Nodes", dnode_, "Inputs", "Input"], stype_, dtype_] := Scope[
	{str1, str2, kind} = toTypeMismatchData[stype, dtype];
	ThrowFailure["ninctyp2", kind, pnameform @ snode, MsgForm @ $currentNodes @ snode, pnameform @ dnode, MsgForm @ $currentNodes @ dnode, str1, str2]
];

pnameform[s_String] := $currentNodeName <> " " <> If[IntStringQ[s], s, MsgForm[s]];

edgeTypeError[spath_, dpath_, stype_, dtype_] := Scope[
	{str1, str2, kind} = toTypeMismatchData[stype, dtype];
	ThrowFailure["ninctyp", kind, MsgForm[spath], str1, MsgForm[dpath], str2]
];

virtOut[name_, {p_NetPath}] := NetPath["Outputs", name] -> p;

(* make a unique output port for each clashing dangling output *)
virtOut[name_, list_] := Scope[
	(* this removes the original clashing port from the outputs var in toNetGraph *)
	KeyDropFrom[$outputs, name]; 
	Table[
		newName = name <> IntegerString[i];
		If[KeyExistsQ[$outputs, newName], ThrowFailure["multoutclash", name, newName]];
		$outputs[newName] = $otypes[list[[i]]];
		toGraphPort[$itypes, NetPath["Outputs", newName] -> list[[i]]]
	, 
		{i, Length[list]}
	]
];

General::multoutclash = "Cannot use integer suffix to automatically disambiguate multiple dangling output ports with name `` from each other because of existing port with name ``. Please specify names for these output ports manually.";

virtIn[ns:NetPath[__, name_]] := 
	ns -> NetPath["Inputs", name];

uniqueLabelling[list_] :=
	Values @ Sort @ Flatten @ KeyValueMap[labelGroup, PositionIndex[list]];

labelGroup[name_, {ind_}] := ind -> name;
labelGroup[name_, inds_List] := 
	Table[inds[[i]] -> name <> IntegerString[i], {i, Length[inds]}];


PackageExport["ReplaceLayer"]

ReplaceLayer[HoldPattern[NetGraph[assoc_Association, meta_]], pos_ -> newlayer_] := CatchFailureAsMessage @ Scope[
	If[!ValidNetQ[newlayer], ReturnFailed[]];
	newlayer = NData[newlayer];
	nodes = assoc["Nodes"];
	Which[
		StringQ[pos],		
			If[KeyExistsQ[nodes, pos],
				nodes[[pos]] = newlayer,
				AppendTo[nodes, pos -> newlayer];
			],
		IntegerQ[pos],
			If[1 <= Abs[pos] <= Length[nodes],
				nodes[[pos]] = newlayer,
				ThrowFailure[NetGraph::oorpart, pos];
			],
		True,
			ThrowFailure[NetGraph::badpart, pos];
	];
	assoc["Nodes"] = nodes;
	ConstructWithInference[NetGraph, assoc, meta]
];


DefineCustomBoxes[NetGraph, 
	NetGraph[assoc_Association, _Association] ? System`Private`HoldNoEntryQ :> formatNetGraph[assoc]
];

PackageScope["$NetGraphIconBoxes"]

$NetGraphIconBoxes := $NetGraphIconBoxes = Uncompress @ "
1:eJxTTMoPSmNhYGAo5gYS7kWJBRmZycVO+RVpXCBBkIxPZnFJGiOIxwkkQoDy2XmpxcVFRs/WqT5p
7rNPY0dRyQzicQCJIHcn5/yc/KKidrGb574HP7fHZGAzdodc6+vAHfOgciCz/QsSkzNLKosYwOADVA
ZkhVd+Zp5bflFuGhOyE4JZgQzfzJLUIqgWFQeIq/igWlJTnEuLylJBvmTE9CWqT2C8TJBcJoiXCRJH
U8qCwkNxDdQNRg4QRoOTAznyDXB5a+zyjlB5BmT5oNKcVHDEgv3rnJNfnJqC6vJMkJaBDx1PAqFDQP
6AB4HQcRjSoQOLewYD/GkHhzwi7VgSSDtWZIQOioGIrOmWmJwKzpqY5cG/KxUv1Qyf2Be9UjPkWCPz
yr5ojUxUivX9+/YQw0BB7ZaZk4MzqJlx88BuygQZgxR3qJKMxEmSGkn60EB0cMEeyGZEyh9wwB7J+r
jkiQpzSFDfsi9a9fGSb5IAMPC9eJi028VO4g5zVEvoEuZ0sQTVX4Qj1hyWe7BHHIMesfI4ItYcSZ6w
a2DJhMERu2kGxMrjcK0ZLnmiklkQuPa+YV9Un7WnZLLEC/uiapF17g+r3gyLrN1gjD/rEpJn0MSftT
HkiQrzrSfK9s2XOm9fdPtnHTDUH9tDm1DPRrM2sVmbwQR/1iUor4E/a4PkScjaWgSyLiF57MkQYT66
PFHJ7PHS2UcUNjyyh+bxB/ZFvKAK5OaroZy1G+wc4AysDUsnYuXNsWd9O1zyRIX5rJkgcNO+6BOo0o
64Cau1n45mbUKZyRmW2O2xZwY7YuWxRyzCfHMisnYDLJkwWGA3zZ5YeezJEGE+ujxRyUwwwnLLiTJg
O3zS2xp70zhgK3GBlP5dFbaDQzlrH3DD3+B28CJSHkeDHG4+eQ1yP2B+ttwCbIf/AlXaky/YF2WDWk
ws20azNqHMBIsYHA3uBjdi5bFHLMJ8ohrkMNW4GtzuxMrjaJB7YZMHd855gAzP3MT0VN/EovTMvGK0
0ASr4YSpCc6sSkU1P1MFxAMpcCwtyc9NLMlMRtMsAGQE5OSXBCXmpacGJKakZOalgy0BAKLXrX0=";

formatNetGraph[assoc_Association] := LengthVarScope[assoc,
	OptimizedArrangeSummaryBox[NetGraph, 
		If[InitializedNetQ[assoc], Identity, fmtUninitializedIcon] @ $NetGraphIconBoxes, 
		fmtCompactContainerInfo[assoc],
		{{netGraphPlot[assoc]}},
		True
	]
];
	

PackageScope["fmtCompactContainerInfo"]

fmtCompactContainerInfo[assoc_] := Scope[
	UnpackAssociation[assoc, inputs, outputs, nodes];
	fmtEntries @ Association[
		oneOrManyAssoc[inputs, "Number of inputs"],
		oneOrManyAssoc[outputs, "Number of outputs"],
		"Number of layers" -> Length[nodes]
	]
];

oneOrManyAssoc[<|name_ -> val_|>, type_] := Row[{Style[name, Black], " ", "port"}] -> val;
oneOrManyAssoc[assoc_, type_] := type -> Length[assoc];

RunInitializationCode[
	Format[HoldPattern[NetGraph[assoc_Association, _Association]] ? System`Private`HoldNoEntryQ, OutputForm] := 
		StringJoin[
			"NetGraph[<", 
			IntegerString @ Length @ assoc["Nodes"], ">, <", 
			IntegerString @ Length @ assoc["Edges"], ">]"
		];
]

PackageExport["$NetGraphInteractivity"]

$NetGraphInteractivity := $NetInteractivity;

PackageScope["netGraphPlot"]

netGraphPlot[assoc_Association, isSummary_:False] := Scope[
	UnpackAssociation[assoc, nodes, edges, inputs, outputs];
	edges = List @@@ Reverse[edges, 2];
	edges = DeleteDuplicatesBy[edges, ReplaceAll[p_NetPath :> Take[p, 2]]]; 
	(* ^ temp workaround for 316828 *)
	edgeTypes = StripCoders @ Extract[assoc, List @@@ edges[[All, 1]]];
	edges = edges /. p_NetPath :> Take[p, 2];
	{edges, vpaths} = Labelling[edges, 2];
	vertexInfo = Replace[vpaths, {
		NetPath["Nodes", id_] :> Append[nodes[id], "ID" -> id], 
		NetPath[io_, id_] :> <|"Type" -> "port", "ID" -> id|>
	}, {1}];
	bigmode = Length[nodes] > 16;
	{names, sizes} = KeysValues[vertexNameAndSize /@ vertexInfo];
	styles = vertexColor /@ vertexInfo;
	tooltips = vertexTooltip /@ vertexInfo;
	info = Association[
		"Edges" -> edges,
		"Labels" -> vertexInfo[[All, "ID"]],
		"EdgeLabels" -> Map[fmtDims, edgeTypes],
		"SelectionVariable" -> None, "Names" -> names, "Sizes" -> sizes,
		"Tooltips" -> If[isSummary, None, tooltips],
		"MaxImageSize" -> If[isSummary, None, {800, 500}],
		"Styles" -> styles, "BigMode" -> bigmode
	];
	If[isSummary, Return @ $netLayerPlotFunction[info, None]];
	portInfo = gridBox @ glueSideBySide[{
		fmtSection[KeyMap[AvoidDecamel] @ inputs, "Inputs", False],
		{{$framed, $framed}},
		fmtSection[KeyMap[AvoidDecamel] @ outputs, "Outputs", False]
	}];
	If[TrueQ[$NetGraphInteractivity],
		makeDynamicNetGraphPlot[info, portInfo, vpaths, 
			ReplaceArraysWithDummies @ nodes,
			ReplaceArraysWithDummies @ inputs, 
			ReplaceArraysWithDummies @ outputs,
			ReplaceArraysWithDummies @ GetSharedArrays @ assoc],
		columnBox[{toPlotBoxes @ $netLayerPlotFunction[info, None], portInfo}]
	]
];

$spf = "\[SpanFromLeft]";
$framed = ItemBox["", Frame -> {{False, False}, {False, LightGray}}];
glueSideBySide[tables_] := Scope[
	len = Max[Length /@ tables];
	list = PadRight[#, len, {{"", ""}}]& /@ tables;
	Join[Sequence @@ list, 2]
];

$netLayerPlotFunction = Function @  
	LayerPlot[
		#Edges, 
		"BaseEdgeStyle" -> GrayLevel[0.7],
		"VertexSelectionVariable" -> #2, 
		"VertexLabels" -> If[#BigMode, None, {Placed[#Labels, Below]}],
		"VertexShapes" -> $vertexShapeFunction,
		"VertexNames" -> #Names, "VertexTooltips" -> #Tooltips, "VertexStyles" -> #Styles,
		"VertexSizes" -> #Sizes,
		"EdgeLabels" -> If[#BigMode, None, Placed[#EdgeLabels, Above]],
		"EdgeTooltips" -> If[#BigMode, #EdgeLabels, None],
		"BaseVertexStyle" -> {EdgeForm[AbsoluteThickness[1]], FontSize -> 10},
		"BaseVertexLabelStyle" -> {FontColor -> Gray},
		"BaseEdgeLabelStyle" -> {FontColor -> GrayLevel[0.7], FontSize -> 8, FontWeight -> "Thin"},
		"RotateEdgeLabels" -> True,
		"Rotated" -> True,
		"ImageScale" -> If[#BigMode, 35, 55],
		"ArrowShape" -> "NarrowKite",
		"DuplicateInputVertices" -> False,
		"ImagePadding" -> {{4, 4}, {4, -6}},
		"ImageMargins" -> {{0, 0}, {0, 0}},
		"MaximumImageSize" -> #MaxImageSize,
		"MaxLabelLength" -> 10
	]

toPlotBoxes[g_] := ReplaceAll[(InterpretationFunction :> _) -> (InterpretationFunction :> None)] @ ToBoxes @ g;

$ChainInGraphIcon := $ChainInGraphIcon = Uncompress @ "
1:eJxTTMoPSmNnYGAo5gAS7kWJBRmZycVpnCARFiDhk1lcksYI4oGEQoCS2XmpxcVF6ds/rDac2Gefxo
pdJdCkSp/UstScIgYweGCPzZSNbxO8JylNhsqBnOFfkJicWVIJ1fUBKgNym1d+Zp5bflFuGhOyjcEg+3
0zS1KLoFpUHNKYQQq4oVpSU5xLi8pSIeaguhPOY0bmeYLkDMCkEYSdCpIHsZigGALMHSA+C4PQDHZwPs
KJQaU5qWCngB3hnJNfnJqCajnEipEbjA1OaMHoRkkwongK4Wm3xORUTE9DHAySD3J3cs7PyS8q+nel4q
Wa4RP7oldqhhxrZF7ZF62RiUqxvn8fX9gywQLKLTMnB3tAMePmYQYUTnlD4uThAc2CHNAM+tCAjYTQDe
Y4+MEOqOqDHSgOVkho3rIvWvXxkm+SADB8vXiYtNvFTpIcrFhsolewDkAkMmFEogU0cqKgtB52/gFYJF
qg8aGRzBACpQ2w8w9EoqoH8ilOBEE75FpfB96wL6rP2lMyWeKFfVG1yDr3h1Vvhn7earCFBBMsLx1wws
GHRgNcPRXy1tYTZfvmS523L7r9sw4YsI/ti8DhvOPZaN4iMW85QyIFnpdssfPheckZlQ+ryeB5yQ47H5
a3YOqpkbceL519RGHDI3toJntgX8QLKl9vvhr6eeuAKySY4HnJEwcfGg1w9VTIW7NmgsBN+6JPoGor4i
as3no6mrdIzFtekEiB5aUGV+x8eN7yQuU3QCMZnpfcsPPhecsTzkdrRYJazgE5+SVBiXnpqcUgxzrm5G
BpaToWF6QmA1WVZOYXpYHBM3s0ZQLIRgUkpqRk5qVnggIBTR0PkOGZm4hQw4CphhOmJjizKjVTAI8hvo
lF6Zl5xUWwkAUAKe5ufw==";

$GraphInGraphIcon := $GraphInGraphIcon = Uncompress @ "
1:eJxTTMoPSmNnYGAo5gAS7kWJBRmZycVprCARFiDhk1lcksYI4nECiRCgZHZeanFxkdGzdapPmvvs09
iwqwSaVOmTWpaaU8QABg/ssZmyQ671deCOeVA5kDP8CxKTM0sqobo+QGVAbvPKz8xzyy/KTWNCtjEY5F
LfzJLUIqgWFYc0ZpACbqiW1BTn0qKyVIg5qO6E85iReZ4gOUMwaQImjVNB8iAWSAnIdp5pjunTZzg7XF
vSavqu2c4BnZ8QwHrtUYSzw6vGtapfpzk6pLx+bfHHzsbhUFzG1YzZjg4ILwSV5qSCnQp2pHNOfnFqCq
rjwE4woL2f9K5Ou/J1lbODg+2ZOWdYzRzQ+QFz8jxXNzs7+F56zeMSYOQwt69uk9wzGwfVWwZhXnJG5P
gJrEMclFhS83NTS4oyk0OKEvOK04CRnFiSmZ8H0cSLlC7di/JLC1ATAN7ESZQBWIwDpTe3xORUcHpjho
kEuTs55+fkFxX9u1LxUs3wiX3RKzVDjjUyr+yL1shEpVjfv28PMQzkebfMnBzs8cSMmwcJGDBphBzxWO
QNiZPHiGcI0HcAUw4uELrBDDv/gIMDinogn4zggoTSLfuiVR8v+SYJAMPNi4dJu13sJI7gQrWBrsE1AJ
HDhBE55tBIgEYGgx4OPixyzNH40MhjcITSBjj4LmjqXRzQUipKRKQyQR0Jwnt9oyNa3/+0Z0ADnDnND0
KufbFPZYQq/DHnxecL0xUOFHjpaE2Yontg6Gb6IHBldcO+qD5rT8lkiRf2RdUi69wfVr0Zapm+wRhKm2
HnM2g6oKgH8skIrq0nyvbNlzpvX3T7Zx0wwB7bQyv7Z6OZngFPpmcwgdJ6OPgaDqjqYXxY5GlBaQMcfG
M09cbEZ/qLO/W39gb8wsj0L98pXNvShpTpF/pyfdmQrnigK7zm9ykJhiGc6R8vnX1EYcMje2juf2BfxA
uquW6+GiqZvsHOAYU+4ISDb44qDuSTEVyzZoLATfuiT6CKPuImrKZ/OprpGbBlemdIYDPYQ2k7HHxYZn
dG5TdAI4/BAk0fOt8OTb0d8Zm+99Xv58+8fmBk+p0zeR7o+H1GZPrf72WLXc6pHXjzXtiwVUbqAFqHAJ
RvA3LyS4IS89JTi0HB4ZiTg6XT4FhckJoMVAUsFjJBRqMpEUA2JiAxJSUzLz2TAVMdD5DhmZuIVw0nTE
1wZlVqJj8eQ3wTi9Iz84qLYJ4HAAYj7FU=";

$LinearInGraphIcon := $LinearInGraphIcon = Uncompress @ "
1:eJxTTMoPSmNiYGAo5gAS7kWJBRmZycUQEX4kEef83IKc1Io0VpAEC5DwySwugSiD8YpmzQSBnfaZQA
EGNDljMLi8P5MRU44BDB7YY5M7ewYEvuCRe4PNPjQeSHMxO5DwL0hMziyphDr1pD1EBqIuLzWNDbcZYA
dkMmHYBBFnxiHOginOisMcVhzmsMLNYcb0kyCQcEwqzs8pLUkNyM/MKwnOrEqFBiiPA0QHKGKD3J2c83
Pyi4rkW18H7pC7ao/JgBgIsg1sEGpMQ3wDdzrYnWieCyrNSS3mBDI8cxPTU0HuyBQC8gAN3oRf";


$vertexShapeFunction = Function @ Switch[#Name,
		_Image,
			Inset[#Name, #Position],
		_Framed,
			{makeRect[#Style, #Position, #Size/2, #Size/2], Inset[First[#Name], #Position]},
		_String | _Style,
			w = Max[5*StringLength[#Name /. Style[s_, ___] :> s], 10];
			{makeRect[#Style, #Position, w, #Size/2], Text[#Name, Offset[{0, -1}, #Position]]},
		None,
			{LightGray, EdgeForm[Gray], Disk[#Position, Offset[{2,2}]]}
	];

makeRect[style_, pos_, w_, h_] := {FaceForm[Lighter @ style], EdgeForm[style], absoluteRectangle[pos, {w, h}]};

absoluteRectangle[pos_, size_] := Rectangle[Offset[-size, pos], Offset[size, pos]];

vertexNameAndSize[assoc_] := Scope[
	type = assoc["Type"];
	Switch[type,
		"port", 
			None -> 8,
		"Chain", 
			Framed[grayIfDeinit[assoc, $ChainInGraphIcon]] -> 24,
		"Graph", 
			Framed[grayIfDeinit[assoc, $GraphInGraphIcon]] -> 24,
		"Total",
			Framed[Style["+", 14]] -> 20,
		"Dot",
			Framed[Style["\[CenterDot]", 14]] -> 20,
		"Linear",
			Framed[$LinearInGraphIcon] -> 20,
		"Replicate",
			Style["\[VerticalEllipsis]", 10] -> 20,
		"Convolution",
			Framed[Pane[Style["\:2217", 20], {11, 28}]] -> 20,
		"Threading",
			str = Lookup[$ThreadingIcons, First @ assoc["Parameters", "Function"], None];
			If[str === None, $funcSymbol, Framed @ Style[str, 14]] -> 20,
		"Elementwise", 
			icon = ActivationIcon[First @ assoc["Parameters", "Function"]];
			If[FailureQ[icon], $funcSymbol, Framed @ icon] -> 20,
		_, 
			toInitial[type] -> 20
	]
];

$funcSymbol = 
	Framed @ Style["\[NegativeVeryThinSpace]\[NegativeVeryThinSpace]#", 
		14, Bold, Italic, 
		FontColor -> RGBColor["#438958"], FontFamily -> "Source Code Pro"]

grayIfDeinit[assoc_, icon_] := If[!NetHasArraysQ[assoc] || !InitializedNetQ[assoc], toGrayIcon[icon, 0.4, 0.7], icon];

$ThreadingIcons = <|
	Plus -> "+", Times -> "\[Times]", Divide -> "\[Divide]", "Subtract" -> "\[Minus]"
|>;

toInitial[s_] := toInitial[s] = 
	StringJoin @ Select[Characters @ StringTrim[s, "Net" | "Loss" | "Operator"], UpperCaseQ];

vertexColor[assoc_] := Scope[
	type = assoc["Type"];
	arraysQ = NetHasArraysQ[assoc];
	Which[
		arraysQ && !InitializedNetQ[assoc], 		Lighter[$uninitializedColor, 0.65],
		type === "Chain" || type === "Graph",		LightGray,
		type === "port",							Gray,
		$LayerData[type, "IsLoss"],					Hue[0, 0.7, 0.9],
		$LayerData[type, "IsOperator"],				Hue[0.13, 0.8, 0.85],
		type === "MX",								Lighter[$uninitializedColor, 0.65],
		arraysQ,									GrayLevel[0.7],
		True,										LightGray
	]
];

vertexTooltip[assoc_] := Column[{
	Style[assoc["ID"], Bold], 
	If[assoc["Type"] === "port", Nothing,
		Block[{iform},
			iform = ToNetInputForm[assoc, HoldForm, 1];
			iform = iform /. f_Function :> Shallow[f, {4, 3}];
			Style[iform, FontFamily -> "Source Code Pro", ShowStringCharacters -> True]
		]
	]
}];

makeDynamicNetGraphPlot[info_, extra_, vpaths_, nodes_, inputs_, outputs_, shared_] := With[
	{interior = columnBox[{
		toPlotBoxes @ $netLayerPlotFunction[info, Dynamic[selection]],
		DynamicBox[
			If[IntegerQ[selection], 
				vpathInfo[vpaths[[selection]], inputs, outputs, nodes, shared], 
				extra
			],
			TrackedSymbols :> {selection}
		]
	}]},
	DynamicModuleBox[
		{selection = None},
		interior
		,
		Initialization :> {NetGraph}
	]
];

columnBox[list_] := gridBox[List /@ list];
gridBox[list_] := GridBox[list, GridBoxAlignment -> {"Columns" -> {{Left}}}];

vpathInfo[NetPath["Inputs", port_], inputs_, outputs_, nodes_, shared_] := 
	typeInfo[port -> inputs[[port]]];

vpathInfo[NetPath["Outputs", port_], inputs_, outputs_, nodes_, shared_] := 
	typeInfo[port -> outputs[[port]]];

vpathInfo[NetPath["Nodes", name_], inputs_, outputs_, nodes_, shared_] := Scope[
	$AmbientSharedArrays ^= Join[$AmbientSharedArrays, shared];
	itemInfo[name -> Lookup[nodes, name]]
]


PackageExport["ExtendNetGraph"]

ExtendNetGraph[NetGraph[oldAssoc_], newNodes_Association, newEdges_List] := CatchFailureAsMessage @ Scope[
	assoc = oldAssoc;
	newNames = Keys[newNodes];
	oldNames = Keys @ assoc["Nodes"];

	(* we don't allow overwriting of old layers... (shoudl we?) *)
	If[IntersectingQ[oldNames, newNames], Panic[]];

	(* ensure new layers are actually layers *)
	newNodes = ToLayer /@ newNodes;
	
	(* add new nodes *)
	$nodes = assoc["Nodes"];
	AssociateTo[$nodes, newNodes];

	(* gather up types in preparation for checking etc *)
	$itypes = $otypes = $stypes = Association[];
	KeyValueScan[collectPortTypes, $nodes];

	(* canonicalize new edges *)
	newEdges = canonicalizeEdges[newEdges];
	newEdges = procEdge /@ newEdges;

	(* update the assoc *)
	assoc["Nodes"] = $nodes;
	assoc["Edges"] = Join[assoc["Edges"], newEdges];

	(* construct a new graph *)
	ConstructWithInference[NetGraph, assoc]
];

NetGraph /: Length[ng_NetGraph ? ValidNetQ] := 
	Length[NData[ng]["Nodes"]];

NetGraph /: Normal[ng_NetGraph ? ValidNetQ] := 
	GetNodes[ng, True];

NetGraph /: VertexList[ng_NetGraph ? ValidNetQ] := NetGraphNodes[ng];

NetGraph /: EdgeList[ng_NetGraph ? ValidNetQ] := NetGraphEdges[ng];


PackageExport["GetNodes"]

GetNodes[HoldPattern @ NetGraph[assoc_Association, _]] := WithAmbientSharedArrays[assoc,
	Map[ConstructNet, assoc["Nodes"]]
];


PackageScope["NetGraphNodes"]

NetGraphNodes[ng_NetGraphP] := Map[ConstructNet, Values @ ng["Nodes"]];


PackageExport["NetGraphEdges"]

NetGraphEdges[ng_NetGraphP] := Scope[
	UnpackAssociation[ng, nodes, edges];
	$dedig = If[DigitStringKeysQ[nodes], FromDigits, Identity];
	$simpleq = Map[And[Length[#Inputs] == 1, Length[#Outputs] == 1]&, nodes];
	$multiportq = Map[DigitStringKeysQ[#Inputs]&, nodes];
	edges = tlNode[#2] -> tlNode[#1]& @@@ edges;
	edges = Catenate @ Lookup[GroupBy[edges, edgeType], {0, 2, 1}, {}];
	edges
];

edgeType[NetPort[_String] -> _] := 0;
edgeType[_ -> NetPort[_String]] := 1;
edgeType[_] := 2;

SetListable[tlNode];

tlNode[NetPath["Nodes", name_, t:"Inputs"|"Outputs"|"States", in_]] := 
	If[$simpleq[name] && t =!= "States", $dedig @ name, 
		NetPort[{$dedig @ name, If[$multiportq[name] && t === "Inputs", FromDigits[in], in]}]];

tlNode[NetPath["Inputs"|"Outputs", port_]] := NetPort[port];


PackageExport["NetGraphPortContributors"]

NetGraphPortContributors[ng_NetGraph, port_] := Scope[
	outNames = OutputNames[ng];
	If[!MemberQ[outNames, port], Return[{}]];
	UnpackAssociation[NData[ng], nodes, edges];
	graph = NetPathGraph[edges];
	delPorts = Cases[outNames, port];
	keepPorts = DeleteCases[outNames, port];
	contribEdges = Complement[
		VertexOutComponent[graph, NetPath["Outputs", #]& /@ delPorts],
		VertexOutComponent[graph, NetPath["Outputs", #]& /@ keepPorts]
	];
	Cases[contribEdges, NetPath["Nodes", name_] :> name]
];


PackageExport["NetGraphRandomInput"]

NetGraphRandomInput::spec = "NetGraph is not fully specified."

NetGraphRandomInput[ng_NetGraph] := Scope[
	If[!FullySpecifiedNetQ[ng], ThrowFailure["spec"]];
	TypeRandomInstance /@ Inputs[ng]
]


PackageExport["LinearNetGraph"]
PackageExport["ABCLinearNetGraph"]
PackageExport["ABCNetGraph"]

(* these functions makes tests easier to write *)

LinearNetGraph[layers_, rules___Rule] := Scope[
	NetGraph[layers, Table[i -> (i+1), {i, Length[layers]-1}], rules]
];

ABCLinearNetGraph[layers_, rules___Rule] := Scope[
	keys = FromLetterNumber @ Range @ Length @ layers;
	NetGraph[Thread[keys -> layers], Thread[Most[keys] -> Rest[keys]], rules]
];

ABCNetGraph[layers_, edges_, rules___Rule] := Scope[
	keys = FromLetterNumber @ Range @ Length @ layers;
	edges = edges /. i_Integer :> keys[[i]];
	NetGraph[Thread[keys -> layers], edges, rules]
]
	


PackageScope["NetChainToNetGraph"]

NetChainToNetGraph[net_NetP] := KeySortBy[
	ReplacePart[net, "Type" -> "Graph"],
	(* jeromel: NetChain and NetGraph do not have the same order of argument when created (which can be addressed later)
		This reordering is done to simplify tests
	*)
	FirstPosition[{"Type", "Inputs", "Outputs", "Nodes", "Edges"}, #]&
];
