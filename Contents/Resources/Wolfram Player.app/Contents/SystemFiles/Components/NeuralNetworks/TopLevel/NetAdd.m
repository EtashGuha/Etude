Package["NeuralNetworks`"]


PackageExport["NetAdd"]
PackageExport["NetMultiply"]
PackageExport["NetSubstract"]

NetAdd[net_, other_] := CatchFailureAsMessage @ iNetCombine[net, other, Plus];
NetMultiply[net_, other_] := CatchFailureAsMessage @ iNetCombine[net, other, Times];
NetSubstract[net_, other_] := CatchFailureAsMessage @ iNetCombine[net, other, Subtract];

iNetCombine[net_, other_, f_] := Scope[

	If[!ValidNetQ[net], ReturnFailed[]];

	$data = NData[net];
	$func = f;

	Which[
		ValidNetQ[other], 
			other = NetArrays[other],
		AssociationQ[other],
			other = KeyMap[
				Replace[
					ToNetPath[$data, #], 
					$Failed :> ThrowFailure["netnomatcharr", #]
				]&, 
				other
			],
		NumberQ[other],
			other = AssociationThread[Keys @ NetArrays[$data], N @ other],
		True,
			ReturnFailed[]
	];	

	rules = KeyValueMap[makeReplacement, other];

	ConstructNet @ ReplacePart[$data, rules]
];

General::netupdnotnum = "Update value `` for  `` should be a number or array."
General::netupdfail = "Update of array at `` failed.";

makeReplacement[key_NetPath, val_] := Scope[
	val = ToPackedArray[val];
	If[!MachineArrayQ[val] && !NumberQ[val], 
		ThrowFailure["netupdnotnum", val, FromNetPath @ key]];
	old = checkRA[key, $data @@ key];
	new = Quiet @ NumericArray[$func[fromRA @ old, fromRA @ val], "Real32"];
	If[!NumericArrayQ[new], ThrowFailure["netupdfail", FromNetPath @ key]];
	Apply[List, key] -> new
];

makeReplacement[key_, _] := ThrowFailure["netnomatcharr", key];

checkRA[_, ra_NumericArray] := ra;

fromRA[ra_NumericArray] := Normal[ra];
fromRA[e_] := e;

General::netnomatcharr = "There is no parameter array in net that matches ``."
checkRA[key_, _Missing] := ThrowFailure["netnomatcharr", key];

General::netnotarray = "Cannot modify uninitialized parameter array."
checkRA[key_, _Nullable | _TensorT] := ThrowFailure["netnotarray"];


PackageExport["NetInterpolate"]

NetInterpolate[net1_, net2_, frac_] := CatchFailureAsMessage @ Scope[
	If[!NumberQ[frac], Return[$Failed]];
	n2 = N[frac]; n1 = 1 - n2;
	iNetCombine[net1, net2, n1 * #1 + n2 * #2&]
];