Package["NeuralNetworks`"]

PackageScope["NetMap"]
PackageScope["RawNetMap"]
PackageScope["$NetMapFlatten"]
PackageScope["$lastNetMapCount"]

DoWith[
$head /: Map[f_, net_$head ? ValidNetQ] := CatchFailureAsMessage[$head, NetMap[f, net]],
$head -> {NetChain, NetGraph}
]

NetMap[f_, net_NetChain] := iNetMap[FreshPath[f[#]]&, net, False, True];
NetMap[f_, net_NetGraph] := iNetMap[FreshPath[f[#]]&, net, False, False];

RawNetMap[f_, net_Association] := 
	iNetMap[f, net, True,
		Switch[net["Type"], "Chain", True, "Graph", False, _, "Layer"]
	];

iNetMap[f_, net_, rawQ_, chainQ_] := Scope[
	data = If[rawQ, net, NData[net]];
	UnpackAssociation[data, $nodes:"Nodes", $edges:"Edges"];
	$mapf = If[rawQ, f /* checkRaw, unrawWrapper[f]];
	$newid = Length[$nodes]; (* <- 
	for renaming in case of digit keys, these will START too high
	and then we'll rely on makeSubNetGraph / makeSubNetChain to remap them later  *)
	$postf = If[!$disableFlatten, If[chainQ, expandChainInplace, expandGraphInplace], Function[Null]];
	$digitsQ = DigitStringKeysQ[$nodes];
	$lastNetMapCount = 0; 
	Block[{$path = Join[$path, NetPath["Nodes", Null]]},
		KeyValueScan[mapLayer, $nodes];
	];
	If[$lastNetMapCount === 0, net, 
		res = If[chainQ, 
			makeSubNetChain[data, $nodes, $AmbientSharedArrays], 
			makeSubNetGraph[data, $nodes, $edges, $AmbientSharedArrays]
		];
		(* ^ these should deal with new shared arrays introduced by f because
		they will eventually call HoistSharedArrays *)
		If[rawQ, NData[res], res]
	]
];


PackageScope["unrawWrapper"]
PackageScope["$forceFlatten"]
PackageScope["$disableFlatten"]

$forceFlatten = False;
$disableFlatten = False;

checkRaw[e:Null|Nothing|$Failed] := e;
checkRaw[assoc_Association /; KeyExistsQ[assoc, "Type"]] := assoc;
checkRaw[e_] := repFail["replacement `` was not a valid net", MsgForm[e]];

unrawWrapper[f_][assoc_] := deconstruct @ f @ ConstructNet @ assoc;

deconstruct[e:Null|Nothing|$Failed] := e;
deconstruct[net_ ? ValidNetQ] := NData[net];
deconstruct[e_] := repFail["replacement `` was not a valid net", MsgForm[e]];

expandChainInplace[name_, assoc_] := Scope[
	If[assoc["Type"] =!= "Chain", Return[]];
	div = If[$digitsQ, "999", "/"];
	(* dummy numbering now, will pass DigitStringKeys test and get renumbered *)
	rules = KeyValueMap[StringJoin[$name, div, #1] -> #2&, assoc["Nodes"]];
	$nodes ^= Insert[$nodes, rules, IndexOf[Keys[$nodes], name]];
	$nodes[name] =.;
];

expandGraphInplace[name_, assoc_] := Scope[
	If[assoc["Type"] =!= "Graph", Return[]];
	innerNodes = assoc["Nodes"];
	innerEdges = assoc["Edges"];
	renaming = AssociationMap[
		If[DigitStringKeysQ[$nodes], 
			Function[IntegerString[++$newid]], 
			Function[$name <> "/" <> #]
		], 
		Keys @ innerNodes
	];
	innerNodes = KeyMap[renaming, assoc["Nodes"]];
	innerEdges = innerEdges /. NetPath["Nodes", n_, rest___] :> NetPath["Nodes", renaming[n], rest];
	(* ^ offset the nodes and edges of the inner graph to be safe to embed into the ambient graph *)	
	(* we need to do 4 kinds of rewrite here:
	1. rewrite inner edges whose source (RHS) is NetPort["Inputs", ..] to have source be the transitive source in the outer graph (via 4)
	2. drop inner edges whose dest (LHS) is NetPort["Outputs, ..]. instead we will use these edges for rename purposes in 3.
	3. rewrite outer edges whose source (RHS) is NetPort["Nodes", name, "Outputs", ..] to have source be the transitive source in the inner graph (via 3).
	4. drop outer edges whose dest (LHS) is NetPort["Nodes", name, "Inputs", ...], instead use these for rename purposes in 1.
	*)

	{outputAliases, innerEdges} = SelectDiscard[innerEdges, MatchQ[NetPath["Outputs", _] -> _]];
	(* ^ seperate the inner edges into those that get renamed and those that cross the output edge of the inner 
	graph and will be used to rename outer edges whose source are outputs of the inner graph *)
	{inputAliases, $edges} ^= SelectDiscard[$edges, MatchQ[NetPath["Nodes", name, "Inputs", _] -> _]];
	(* ^ seperate the outer edges into those that feed the inputs of the inner graph, which will be completely
	dropped and used instead to rename the corresponding inner edges, and unrelated outer edges *)
	nodePath = NetPath["Nodes", $name];
	outerRewrites = MapColumn[Join[nodePath, #]&, 1, outputAliases];
	innerRewrites = MapColumn[Drop[#, 2]&, 1, inputAliases];
	$nodes[name] =.;
	$edges ^= Join[$edges /. outerRewrites, innerEdges /. innerRewrites];
	$nodes ^= Join[$nodes, innerNodes];
];

mapLayer[name_, assoc_] := Scope[
	$path[[-1]] = $name = name;
	res = $mapf[assoc];
	If[res === Null || FailureQ[res] || (res === assoc && !$forceFlatten), Return[]];
	$lastNetMapCount++;
	If[res === Nothing,
		$nodes[name] =.;
		elisions = List @ toElisionRules[$edges, name, assoc];
		If[elisions === {}, repFail["cannot remove `` as its input and output are not compatible", MsgForm[$path]]];
		$edges ^= Select[$edges /. elisions, FreeQ[NetPath["Nodes", $name, ___]]];
		Return[];
	];
	res = TestLayersCompatible[assoc, res];
	$nodes[name] ^= res;
	$postf[$name, res];
];

iNetMap[f_, net_, True, "Layer"] := Scope[
	If[NProperty[net, "SubNets"] === {}, Return[net]];
	$mapf = f;
	MapAtSubNets[mapSubnet, net]
];

mapSubnet[subnet_] := Scope[
	$name = Last[$path];
	res = $mapf[subnet]; 
	If[res === subnet, Return[subnet]];
	If[res === Nothing, repFail["cannot replace operator subnet with Nothing"]];
	TestLayersCompatible[subnet, res];
	res
];

PackageScope["TestLayersCompatible"]
(* also used in NetReplacePart to test layer replacements *)
TestLayersCompatible[old_, new_] := Scope[
	testCompatIO[Outputs[old], Outputs[new], "output"];
	ins1 = Inputs[old]; ins2 = Inputs[new];
	If[MatchQ[ins2, <|$Multiport -> _|>],
		If[!DigitStringKeysQ[ins1], 
			repFail["cannot replace fixed-input layer with ``, a layer that takes a variable number of inputs",
				MsgForm[new]
			]
		];
		new["Inputs"] = ins1;
	,
		testCompatIO[ins1, ins2, "input"];
	];
	new
];

testCompatIO[old_, new_, type_] := 
	MapAssocAssoc[
		If[!UnifiableQ[#2, #3],
			repFail["`` of replacement net (``) was incompatible with original `` (``)", 
				nameIOPort[type, #1], MsgForm[#3],
				nameIOPort[type, #1], MsgForm[#2]
			]
		]&, 	
		old, new,
		repFail["replacement net was missing ``", nameIOPort[type, #]]&,
		repFail["replacement net had extra ``",  nameIOPort[type, #]]&
	]

nameIOPort[type_, name_] := StringForm["`` port named \"``\"", type, name];

General::invnetrep = "Replacement of `` failed: ``."
repFail[args___] := 
	ThrowFailure["invnetrep", 
		MsgForm @ $path, 
		StringForm[args] /. np_NetPath :> MsgForm[np]
	];

