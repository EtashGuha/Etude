Package["NeuralNetworks`"]

PackageExport["ToNetPathPattern"]

ToNetPathPattern[net_NetP, part_] := Scope[
	toPath = toPathP;
	ToNetPath[net, part]
];


PackageExport["ToNetPath"]

ToNetPath[net_NetP, part_] := Scope[
	bag = Bag[];
	Catch[
		toPath[net, Sequence @@ ToList[part]];
		NetPath @@ BagContents[bag],
		ToNetPath
	]
];

Clear[toPath, toPathNP, toPathP, findPath, getSubPath];
toPath = toPathNP;

stuff[e_] := BagPush[bag, e];

(* this has to check for resolution consistency: do all the alternatives resolve
'then' the same way? if so, we can represent the result with one pattern, otherwise
its an error. used for now by evaluation port patterns, net[data, NetPort[{All, "Output"}]].
*)
toPathP[net_, patt:(All | _Alternatives | _Span | Verbatim[_]), then__] := Scope[
	toPathP[net, patt]; 
	base = BagPart[bag, -1];
	If[base === _, base = Keys @ net["Nodes"]];
	subs = Table[
		ToNetPath[net["Nodes", base[[i]]], then],
		{i, Length[base]}
	];
	first = First[subs, $Failed];
	If[!FailureQ[first] && (SameQ @@ subs), Scan[stuff, first], pathFail[]];
];

toPathP[net_, Verbatim[___]] := stuff[___];

toPathP[net_, All | Verbatim[_]] := Scope[
	If[!MatchQ[net["Type"], "Chain"|"Graph"], pathFail[]];
	stuff["Nodes"]; stuff[_];
];

(* TODO: instead of sometimes pathFailing when a part is missing (e.g. layer 9 of 5-layer chain)
and othertimes not (layer key "foo" that doesn't exist), we should always fail with a nice message,
to make it easier for users to know when they've made a mistake *)

(* determines how a container should resolve integer key specs.
in the case of a graph, we can't resolve integer keys when the
assoc form was used to construct the graph. 
*)
int2keyFunc[net_] := Scope[
	nodes = net["Nodes"];
	Switch[net["Type"], 
		"Chain",
			If[!DigitStringKeysQ[nodes],
				With[{keys = Keys[nodes]}, UnsafeQuietCheck[keys[[#]], pathFail[]]&],
				intKeyStr[Length[nodes]]
			],
		"Graph",
			If[!DigitStringKeysQ[nodes],
				NetGraph::intpartassoc = "Sorry, you cannot use integer layer specification `` when the original graph was constructed with string keys. Please use an explicit string specification instead.";
				Function[Message[NetGraph::intpartassoc, #]; pathFail[]],
				intKeyStr[Length[nodes]]
			],
		_,
			pathFail[]
	]
];

intKeyStr[n_][x_] := IntegerString @ Which[
	1 <= x <= n, x, 
	-n <= x <= -1, n + 1 + x,
	True, pathFail[]
];

toPathP[net_, alt_Alternatives] := Scope[
	keyf = int2keyFunc[net];
	stuff["Nodes"]; stuff[alt /. i_Integer :> keyf[i]];
];

toPathP[net_, s_Span] := Scope[
	If[net["Type"] =!= "Chain", pathFail[]];
	$keys = Keys @ net["Nodes"];
	span = Map[toKey] @ MapAt[Replace[All -> 1], 1] @ MapAt[Replace[All -> -1], 2] @ s;
	stuff["Nodes"]; stuff[Alternatives @@ UnsafeQuietCheck[Part[$keys, span], pathFail[]]];
];

toPathP[args___] := toPathNP[args];

toKey[key_String] := First @ FirstPosition[$keys, key, pathFail[]];
toKey[pos_Integer] := pos;
toKey[_] := pathFail[];

toPathNP[net_, NetSharedArray[name_String]] := Scope[
	res = net["SharedArrays", name];
	If[MissingQ[res], pathFail[]];
	stuff["SharedArrays"]; stuff[name];
];

toPathNP[net_, part_String] := Block[{},
	Do[
		If[KeyExistsQ[Lookup[net, field, <||>], part],
			stuff[field]; stuff[part]; 
			Return[net[field, part], Block];
		],
		{field, {"Inputs", "Outputs", "Parameters", "Arrays", "Nodes"}}
	];
	pathFail[]
];

toPathNP[net_] := Null;

toPathNP[net_, part_, rest___] := 
	toPath[findPath[net["Type"], net, part], rest];

Clear[findPath];

findPath[type:"Graph"|"Chain", net_, part_] := (
	stuff["Nodes"]; 
	$kf = int2keyFunc[net];
	getSubPath[net["Nodes"], part]
);

findPath[type_, net_, part_] := Scope[
	subnets = SubNetPaths[net];
	If[!StringQ[part] || !MemberQ[subnets[[All, 2]], part], pathFail[]];
	stuff["Parameters"]; stuff[part];
	net["Parameters", part]
]

getSubPath[subs_, n_Integer] := Scope[
	If[1 <= Abs[n] <= Length[subs],
		key = $kf[n];
		stuff[key];
		subs[key]
	,
		pathFail[]
	]
];

getSubPath[subs_, part_String] := 
	If[KeyExistsQ[subs, part],
		stuff[part];
		subs[part]
	,
		pathFail[]
	];

getSubPath[___] := pathFail[];

pathFail[] := Throw[$Failed, ToNetPath];



PackageExport["FromNetPath"]

FromNetPath[(List|NetPath)[pos___]] := drop[pos];
FromNetPath[list:{__List|__NetPath}] := drop @@@ list;

drop["Inputs"|"Outputs"|"Arrays"|"Parameters"|"States", in_] := {in};
drop["Nodes", layer_, rest___] := Prepend[drop[rest], fromLayerSpec[layer]];
drop["Parameters", name_, rest__] := Prepend[drop[rest], name];
drop["SharedArrays", sa_] := NetSharedArray[sa];
drop[] := {};
drop[p___] := Panic["InvalidNetPathSpec", "Don't know how to convert ``.", {p}];

fromLayerSpec[s_String] := If[IntStringQ[s], FromDigits[s], s];


PackageScope["NetPathString"]

NetPathString[p_NetPath] := TextString[NetPathForm[p]];


PackageScope["NetPathForm"]

NetPathForm[p_NetPath] := pathForm @@ p;

PackageScope["IntStringQ"]

IntStringQ[str_String] := StringMatchQ[str, DigitCharacter..];
IntStringQ[_] := False;

Clear[pathForm];
pathForm[name_String] := ToLowerCase@name; (* for {in,out}put of {en,de}coder *)
pathForm[] := "net";
pathForm["Parameters", "Net"] := "net within operator";
pathForm["Parameters", "Net", rest__] := StringForm["`` of net within operator", pathForm[rest]];
pathForm["Nodes", vert_] := PartForm[vert, "layer"];
pathForm["Nodes", vert_, rest__] := StringForm["`` of ``", pathForm[rest], PartForm[vert, "layer"]];
pathForm[___, "SharedArrays", param_] := StringForm["shared array \"``\"", param];
pathForm[___, "Arrays", param_] := StringForm["array \"``\"", param];
pathForm[___, "Parameters", param_] := paramForm[param];
pathForm[___, "Inputs", $Multiport] := "input";
pathForm[___, "Inputs", port_] := PartForm[port, "input"];
pathForm[___, "Outputs", port_] := PartForm[port, "output"];
pathForm[___, "States", state_] := StringForm["recurrent state \"``\"", state];
pathForm[___] := "unknown part"; (* <- copout, but avoid showing code in message *)

(* Internal parameters like $InputDimensions are rewritten as 'input dimensions' *)
paramForm[param_] := If[StringStartsQ[param, "$"],
	ToLowerCase @ deCamelCase @ StringDrop[param, 1],
	StringForm["parameter \"``\"", param]
];


PackageScope["FindSubNets"]

SetUsage @ "
FindSubNets[net$, layerType$] finds all layers of type layerType$ and returns their NetPath positions.
FindSubNets[net$, layerType$, n$] returns only the first n$ matches.
FindSubNets[net$, layerType$, n$, True] finds layers that are within operators."

FindSubNets[net_NetP, type_, n_:Infinity, withinOp_:False] := Scope[
	pos = Position[net, <|"Type" -> type, ___|>, {0, Infinity}, n, Heads -> False];
	paths = (NetPath @@@ pos) /. Key[k_] :> k;
	If[withinOp === False, paths = Discard[paths, SubnetPathQ]];
	paths
];


PackageScope["PartForm"]

PartForm[part_Integer, type_] := 
	PartForm[IntegerString[part], type];

PartForm[part_, type_] := Switch[
	ToLowerCase[part],
	"1", StringForm["first ``", type],
	"2", StringForm["second ``", type],
	"3", StringForm["third ``", type],
	type | (First @ StringSplit @ type), type, (* <- e.g. "Input" for an input *)
	_ ? IntStringQ,  StringForm["`` ``", type, part],
	_, StringForm["`` \"``\"", type, part]
];


PackageScope["FmtPortName"]

FmtPortName[str_] := PartForm[str, "port"];
FmtPortName[str_, "Inputs"] := PartForm[str, "input port"];
FmtPortName[str_, "Outputs"] := PartForm[str, "output port"];


PackageScope["SubnetPathQ"]

SubnetPathQ[path_] := ContainsQ[path[[ ;; ;; 2]], "Parameters"];


PackageScope["ArrayPathQ"]

ArrayPathQ[path_] := MatchQ[path, NetPath[___, "Arrays", __]|NetPath["SharedArrays", __]];
(* ^ the reason that there can be multiple suffixes in the paths above is that they will have $Agg appended to them if they are to be aggregated *)