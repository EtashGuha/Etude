Package["NeuralNetworks`"]

PackageScope["NetKeyMap"]


DoWith[
$head /: KeyMap[func_, net_$head ? ValidNetQ] := CatchFailureAsMessage[KeyMap, NetKeyMap[func, net]],
$head -> {NetChain, NetGraph}
]


General::invnetkeymapres = "Function returned a non-string result `` for the key ``."
General::invnetkeymapinp = "Cannot change keys when layers were originally provided as a list."
General::invnetkeymapdup = "Function mapped the keys `` and `` to the same value ``."

NetKeyMap[func_, (head:NetChain|NetGraph)[assoc_, meta_]] := Scope[
	nodes = assoc["Nodes"];
	If[DigitStringKeysQ[nodes], ThrowFailure["invnetkeymapinp"]];
	$remaprules = {}; $backward = <||>;
	assoc["Nodes"] = KeyMap[keyFunc[func], nodes];
	assoc = assoc /. $remaprules;
	head[assoc, meta]
];

keyFunc[func_][key_] := Scope[
	newKey = func[key];
	If[!StringQ[newKey], ThrowFailure["invnetkeymapres", newKey, key]];
	If[!MissingQ[last = $backward[newKey]], ThrowFailure["invnetkeymapdup", key, last, newKey]];
	$backward[newKey] = key;
	If[newKey =!= key,
		AppendTo[$remaprules, With[{newKey = newKey},
			NetPath["Nodes", key, mid___] :> NetPath["Nodes", newKey, mid]
		]];
	];
	newKey
];
