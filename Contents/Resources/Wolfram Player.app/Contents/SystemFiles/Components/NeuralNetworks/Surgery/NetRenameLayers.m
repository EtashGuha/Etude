Package["NeuralNetworks`"]


PackageExport["NetRename"]

NetRename::invarg1 = "First argument should be a NetChain or NetGraoph."

NetRename[net_, spec_] := CatchFailureAsMessage[
	If[!ValidNetQ[net] || !MatchQ[net, _NetChain | _NetGraph], ThrowFailure["invarg1"]];
	If[MatchQ[ToList[spec], List @ Repeated[NetPort[_String] -> NetPort[_String]]],
		Return @ NetRenamePorts[net, ToList @ spec]];
	If[RuleLikeQ[spec] || RuleLikeVectorQ[spec],
		f = Replace[spec], f = spec];
	NetKeyMap[f, net]
];

DeclareArgumentCount[NetRename, 2];

NetRenamePorts[net_NetP, spec_] := Scope[
	$inputs = Inputs[net];
	$outputs = Outputs[net];
	rules = toIORule @@@ spec;
	net = net /. rules;
	net["Inputs"] = $inputs;
	net["Outputs"] = $outputs;
	ConstructNet @ net
];

NetRename::inviopath = "`` is not a name of an input or output port of the net.";

toIORule[NetPort[old_], NetPort[new_]] := Which[
	KeyExistsQ[$inputs, old], 
		$inputs = KeyMap[Replace[old -> new], $inputs];
		NetPath["Inputs", old] -> NetPath["Inputs", new],
	KeyExistsQ[$outputs, old], 
		$outputs = KeyMap[Replace[old -> new], $outputs];
		NetPath["Outputs", old] -> NetPath["Outputs", new],
	True, 
		ThrowFailure["inviopath", old]
];