Package["NeuralNetworks`"]

PackageExport["NetFlatten"]

Options[NetFlatten] = {
	AllowedHeads -> Automatic
};
(* Option AllowedHeads can be:
	"Same": NetChain -> NetChain | NetGraph -> NetGraph
	NetGraph: NetChain -> NetGraph | NetGraph -> NetGraph
	Automatic: NetChain -> NetGraph or NetChain | NetGraph -> NetGraph
*)

Clear[NetFlatten];

(* jeromel: Backward compatibility for internal option *)
NetFlatten[args__, "Complete" -> True] := NetFlatten[args, AllowedHeads -> NetGraph];
NetFlatten[args__, "Complete" -> False] := NetFlatten[args, AllowedHeads -> "Same"];

NetFlatten[net_, opts:OptionsPattern[]] := NetFlatten[net, Infinity, opts];
NetFlatten[net_, maxLevel_, OptionsPattern[]] := CatchFailureAsMessage[
	TestValidNet[net];
	If[Or[!IntegerQ[maxLevel] && maxLevel =!= Infinity, maxLevel <= 0],
		ThrowFailure[NetFlatten::flev, maxLevel]
	];
	If[!ValidPropertyQ[NetFlatten, OptionValue[AllowedHeads],
			{Automatic, Inherited, "Same", NetGraph, NetChain},
			{"NetGraph", "NetChain"}
		],
		ThrowFailure[]
	];
	iNetFlatten[NData[net], maxLevel, OptionValue[AllowedHeads]]
];

DeclareArgumentCount[NetFlatten, {1, 2}];

Clear[iNetFlatten];

NetFlatten::err = "Cannot flatten net.";
NetFlatten::notchained = "The network cannot be a flattened to NetChain."; (* Failure happens because of option AllowedHeads -> NetChain *)
iNetFlatten[net_, maxLevel_, headoutput_] := Scope @ WithAmbientSharedArrays[net,

	chainCanBeGraph = Or[
		MatchQ[headoutput, NetGraph|"NetGraph"|Automatic],
		And[headoutput === Inherited, net["Type"] === "Graph"]
	];
	graphCanBeChain = Or[
		MatchQ[headoutput, NetChain|"NetChain"|Automatic],
		And[headoutput === Inherited, net["Type"] === "Chain"]
	];

	$forceFlatten = True;
	$maxLevelFlatten = maxLevel;
	net = ReplaceRepeated[net, {
		Which[
			chainCanBeGraph,
				assoc:<|"Type" -> "Chain", ___, "Nodes" -> _, ___|>
				:> RuleCondition[
					res = NetChainToNetGraph[assoc];
					If[FailureQ[res], ThrowFailure["err"]];
					res
				],
			graphCanBeChain,
				assoc:<|"Type" -> "Graph", ___, "Nodes" -> _, "Edges" -> edges_, ___|> /; chainedEdgesQ[edges]
				:> RuleCondition[
					res = netGraphToNetChain[assoc];
					If[FailureQ[res], ThrowFailure["err"]];
					res
				],
			True,
				Nothing
		]
	}];
	net = rawFlatten @ net;
	If[graphCanBeChain,
		net = ReplaceRepeated[net,
			assoc:<|"Type" -> "Graph", ___, "Nodes" -> _, "Edges" -> edges_, ___|> /; chainedEdgesQ[edges]
			:> RuleCondition[
				res = netGraphToNetChain[assoc];
				If[FailureQ[res], ThrowFailure["err"]];
				res
			]
		];
	];
	If[MatchQ[headoutput, NetChain|"NetChain"] && net["Type"] =!= "Chain",
		ThrowFailure["notchained"];
	];
	ConstructNet @ net
];

chainedEdgesQ[edges_List] := Scope[
	inputs = Join[
		Cases[edges, Rule[NetPath["Nodes", layer_, ___], _] :> layer],
		Cases[edges, Rule[_, NetPath["Inputs", ___]] :> NetPath["Inputs"]]
	];
	outputs = Join[
		Cases[edges, Rule[_, NetPath["Nodes", layer_, ___]] :> layer],
		Cases[edges, Rule[NetPath["Outputs", ___], _] :> NetPath["Outputs"]]
	];
	Max[Counts[inputs], Counts[outputs]] <= 1
];
netGraphToNetChain[net_] := KeySortBy[
	ReplacePart[net, "Type" -> "Chain"],
	(* jeromel: NetChain and NetGraph do not have the same order of argument when created (which can be addressed later)
		This reordering is done to simplify tests
	*)
	FirstPosition[{"Type", "Nodes", "Edges", "Inputs", "Outputs"}, #]&
];

NetFlatten::flev = "The level argument `1` in position 2 should be a non-negative integer or Infinity giving the levels to flatten through.";

DeclareMethod[rawFlatten, rawFlattenLayer, rawFlattenContainer, rawFlattenOperator]

rawFlattenLayer[net_] := net;

rawFlattenContainer[net_] := Scope @ If[$maxLevelFlatten > 0,
	--$maxLevelFlatten;
	res = RawNetMap[rawFlatten, net];
	++$maxLevelFlatten;
	res
,
	net
];
rawFlattenOperator[net_] := Scope @ If[$maxLevelFlatten > 0,
	--$maxLevelFlatten;
	res = MapAtSubNets[rawFlatten, net];
	++$maxLevelFlatten;
	res
,
	net
];