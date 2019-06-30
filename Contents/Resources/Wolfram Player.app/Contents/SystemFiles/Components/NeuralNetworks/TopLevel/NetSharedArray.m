Package["NeuralNetworks`"]


(* 

Design of shared arrays:

All nets acquire an optional SharedArray field. When nets are embedded in one another, shared arrays are 
hoisted up one level. When subnets are extracted, shared arrays are copied into the sub-arrays.

As a courtesy we can complain when a net is put into a container and wants to add its shared arrays but they already exist
as a different size or as literal rawarrays with different contents.

When doing inference, SharedArray references are replaced with simple NetPath["SharedArrays", name], thanks to the fact that
shared arrays are always hoisted to the root of a net.

Applying a NetInsertSharedArrays to a net will replace all its interior arrays with shared versions named after the original
paths where they were found.
*)

PackageExport["NetSharedArray"]


PackageScope["HoistSharedArrays"]

(* HoistSharedArrays is used by containers, applies to an assoc of nodes, returns the nodes now stripped of their
shared arrays along with the grouped shared arrays *)

HoistSharedArrays[nodes_Association] := 
	If[!MemberQ[nodes, node_ /; KeyExistsQ[node, "SharedArrays"]],
		{nodes, <||>}
	,
		{
			KeyDrop["SharedArrays"] /@ nodes, 
			JoinSharedArrays[Lookup[Values @ nodes, "SharedArrays", Nothing]]
		}
	];

HoistSharedArrays[params_Association, elems_] := Scope @ 
	If[NoneTrue[Lookup[params, elems], KeyExistsQ["SharedArrays"]], 
		{params, <||>}
	,
		sarrays = <||>; 
		params = MapAt[
			Function[
				sarrays = JoinSharedArrays[sarrays, Lookup[#, "SharedArrays", <||>]];
				KeyDrop[#, "SharedArrays"]
			], 
			params, List /@ elems
		];
		{params, sarrays}
	];

PackageExport["NetInsertSharedArrays"]

NetInsertSharedArrays[expr:(head_Symbol[assoc_Association, meta_]), prefix_:None] := CatchFailure @ Scope[
	$sharedArrays = Association[];
	If[prefix =!= None && !StringQ[prefix], 
		NetInsertSharedArrays::badprefix = "Prefix `` should be either None or a string.";
		ReturnFailed["badprefix", prefix];
	];
	$sharingPrefix = Replace[prefix, None -> ""];
	assoc = shareAllArrays[assoc];
	If[$sharedArrays === <||>,
		NetInsertSharedArrays::noarr = "Net contained no arrays that could be shared, returning net unchanged.";
		Message[NetInsertSharedArrays::noarr];
		Return[expr];
	];
	assoc["SharedArrays"] = $sharedArrays;
	System`Private`ConstructNoEntry[head, assoc, meta]
];

DeclareArgumentCount[NetInsertSharedArrays, {1, 2}]

DeclareMethod[shareAllArrays, shareLayerArrays, shareContainerArrays, shareOperatorArrays];

$auxArrays = {};
shareLayerArrays[layer_] := Scope[
	$auxArrays = $LayerData[layer["Type"], "AuxArrays"];
	MapAtFields["Arrays", shareArray, layer]
];

shareContainerArrays[cont_] := 
	MapAtFields["Nodes", shareAllArrays, cont];

shareOperatorArrays[op_] := 
	MapAtSubNets[shareAllArrays, shareLayerArrays @ op];

shareArray[sa_NetSharedArray] := sa;

shareArray[None] := None;

shareArray[arr_] /; !MemberQ[$auxArrays, Last[$path]] := Scope[
	path = FromNetPath[$path];
	name = $sharingPrefix <> StringRiffle[path, "/"];
	If[KeyExistsQ[$sharedArrays, name], 
		NetInsertSharedArrays::nameclash = "Cannot share array at `` because the target name `` already exists.";
		ThrowFailure["nameclash", path, name]
	];
	$sharedArrays[name] = Replace[arr, Nullable[a_] :> a];
	NetSharedArray[name]
];

shareArray[arr_] := arr;