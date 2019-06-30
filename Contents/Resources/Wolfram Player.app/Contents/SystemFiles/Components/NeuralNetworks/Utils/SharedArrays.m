Package["NeuralNetworks`"]


PackageScope["GetSharedArrays"]
PackageScope["AttachSharedArrays"]

GetSharedArrays[net_NetP] := Lookup[net, "SharedArrays", <||>];

AttachSharedArrays[net_, <||>] := net;
AttachSharedArrays[net_, arrays_Association] := Scope[
	keys = DeleteDuplicates @ DeepCases[net, _NetSharedArray];
	If[keys === {}, Return[net]];
	new = KeyTake[arrays, keys[[All, 1]]];
	old = GetSharedArrays[net];
	Append[net, "SharedArrays" -> KeySort[Join[old, new]]]
];


PackageScope["JoinSharedArrays"]

Clear[JoinSharedArrays];
JoinSharedArrays[shared__Association] := JoinSharedArrays[{shared}];
JoinSharedArrays[shared:{___Association}] := KeySort @ Association @ Merge[
	shared,
	Function[Last @ SortBy[#,
		Function[arrayOrSpec, {NumericArrayQ[arrayOrSpec], arrayOrSpec /. {ListT[__] :> 0}}]]
	]
];
(* ^ jeromel:
	The Merge[#, Last @ Sort...] is to fix 370173 (already initialized arrays are prioritary, when setting a shared array).
	The result of the merge is the last element in an ordered list, with NumericArray at last (if any).
	If there is no NumericArray, and there are different tensor specifications, we try to find the most specific tensor dimension (without ListT).
*)


PackageScope["$AmbientSharedArrays"]
PackageScope["InheritAmbientSharedArrays"]

$AmbientSharedArrays = <||>;

InheritAmbientSharedArrays[net_] :=
	If[$AmbientSharedArrays === <||> || FreeQ[net, _NetSharedArray], net,
		AttachSharedArrays[net, $AmbientSharedArrays]];


PackageScope["WithAmbientSharedArrays"]

SetHoldRest[WithAmbientSharedArrays];
WithAmbientSharedArrays[net_, body_] := Block[
	{$AmbientSharedArrays = Join[$AmbientSharedArrays, GetSharedArrays[net]]},
	body
];