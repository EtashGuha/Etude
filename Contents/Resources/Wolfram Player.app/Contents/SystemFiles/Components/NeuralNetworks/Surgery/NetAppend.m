Package["NeuralNetworks`"]


PackageScope["$PortHintP"]

$PortHintP = RepeatedNull[_String -> _];

PackageScope["AttachPortHints"]

General::noioport = "No input or output port named ``.";
AttachPortHints[] := Identity;
AttachPortHints[hints__][net_] := Scope[
	hints = {hints};
	unknown = Complement[Keys[hints], InputNames[net], OutputNames[net]];
	If[unknown =!= {}, ThrowFailure["noioport", First @ unknown]];
	iNetReplacePart[net, hints]
];

PackageScope["NetPrependAppend"]
PackageExport["NetAppend"]
PackageExport["NetPrepend"]

Options[NetAppend] = Options[NetPrepend] = {"Input" -> Automatic, "Output" -> Automatic};

NetAppend[net_ ? ValidNetQ, spec_, p:$PortHintP] :=  CatchFailureAsMessage @ AttachPortHints[p] @ NetPrependAppend[Append, net, spec];
NetPrepend[net_ ? ValidNetQ, spec_, p:$PortHintP] := CatchFailureAsMessage @ AttachPortHints[p] @ NetPrependAppend[Prepend, net, spec]

DeclareArgumentCount[NetAppend, 2];
DeclareArgumentCount[NetPrepend, 2];

DoWith[
$head /: $func[net_$head ? ValidNetQ, spec_] := CatchFailureAsMessage[$func, NetPrependAppend[$func, net, spec]],
$func -> {Prepend, Append},
$head -> {NetChain, NetGraph}
]


(***************************)
(* NetChain implementation *)
(***************************)

General::invnetcharg1 = "First argument must be a NetChain."
General::invnetchpend1 = "Use ``[chain, \"name\" -> layer] to modify a chain composed of named layers."
General::invnetchpend2 = "Use ``[chain, layer] to modify a chain composed of unnamed layers."

NetPrependAppend[f_, nc_NetChain, layer_List] := 
	Fold[NetPrependAppend[f, #1, #2]&, nc, If[f === Prepend, Reverse, Identity] @ layer];

NetPrependAppend[f_, nc_NetChain, layer_] := Scope[
	pos = Null;
	data = NData[nc];
	If[RuleQ[layer], pos = First[layer]; layer = Last[layer]];
	layer = ToLayer[layer];
	layers = data["Nodes"];
	If[DigitStringKeysQ[layers],
		If[pos =!= Null, ThrowFailure["invnetchpend2", f]];
		layers = NumberedAssociation @ f[Values[layers], layer];
	,
		If[pos === Null, ThrowFailure["invnetchpend1", f]];
		layers = f[layers, pos -> layer]
	];
	ioSpecs = chainIOSpecs[data, f === Append, f === Prepend];
	sharedArrays = GetSharedArrays[data];
	toNetChain[layers, ioSpecs, sharedArrays]
]

(***************************)
(* NetGraph implementation *)
(***************************)

NetPrependAppend[f_, _NetGraph, _] := ThrowFailure[NetGraph::notsupp, f];

NetPrependAppend[f_, _, _] := ThrowFailure["invnetcharg1"];
