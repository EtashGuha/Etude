Package["NeuralNetworks`"]


PackageExport["$Multiport"]

$Multiport = "XXXxXXX";
(* This key, when used as the name of a single input port, represents an unbounded
set of inputs that have just yet to be created. Creation can happen when the layer is constructed 
if a special "Inputs" option is given, otherwise it happens during constructin time or by 
JIT running (not implemented yet) *)


PackageScope["SetMultiport"]

SetHoldFirst[SetMultiport];
SetMultiport[paramsVar_, n_] :=
	paramsVar = ConstantAssociation[
		IntegerString @ Range @ n, 
		paramsVar[$Multiport]
	];
(* ^ TODO jeromel: to properly hangle integers with multi-port layers,
	we need to enforce some constraints between all the types of the different tensors.
	Otherwise, some inferences that used to work thru these layers in the past wouldn't work anymore
*)

PackageScope["UpgradeToMultiport"]

UpgradeToMultiport[assoc_] := Scope[
	inputs = assoc["Inputs", "Input"] /. ListT[n_Integer, z_] :> Table[z, n];
	assoc = assoc;
	If[ListQ[inputs],
		assoc["Inputs"] = AssociationThread[
			IntegerString @ Range @ Length @ inputs,
			inputs
		],
		assoc["Inputs"] = NProperty[assoc, "Inputs"];
	];
	assoc
];


PackageScope["UpgradeGraphTupleEdges"]

UpgradeGraphTupleEdges[assoc_] := MapAt[
	ReplaceAll[rule:(_NetPath -> {___NetPath}) :> RuleCondition[expandTupleEdge[rule]]],
	assoc, "Edges"
];

expandTupleEdge[dst_ -> srcs_List] :=
	Sequence @@ IMap[ReplacePart[dst, -1 -> IntegerString[#1]] -> #2&, srcs];