Package["NeuralNetworks`"]

PackageExport["NetJoin"]
PackageScope["NetChainJoin"]

validNetOrListQ[e_] := (ValidNetQ[e] && MatchQ[e, _NetChain]) || (ListQ[e] && VectorQ[e, ValidNetQ]);

NetJoin[nc__ ? validNetOrListQ, p:$PortHintP] := 
	CatchFailureAsMessage[NetJoin, AttachPortHints[p] @ NetChainJoin[{nc}]];

NetJoin::invargs = "Arguments to NetJoin should be one or more NetChain objects."
NetJoin[__] := RuleCondition[Message[NetJoin::invargs]; Fail];

(* this technique avoids infinite recursion with Join[NetChain[{3, 4, Ramp}], {1, 2, 3}], 
which happens because of Flat attribute on Join *)

NetChain /: Join[c1_NetChain ? ValidNetQ, c2_NetChain ? ValidNetQ] :=
	CatchFailureAsMessage[Join, NetChainJoin[{c1, c2}]];

NetChain /: Join[c1_NetChain ? ValidNetQ, c2_NetChain ? ValidNetQ, c3_NetChain ? ValidNetQ] :=
	CatchFailureAsMessage[Join, NetChainJoin[{c1, c2, c3}]];

NetChain /: Join[c1_NetChain ? ValidNetQ, c2_NetChain ? ValidNetQ, c3_NetChain ? ValidNetQ, c4_NetChain ? ValidNetQ] :=
	CatchFailureAsMessage[Join, NetChainJoin[{c1, c2, c3, c4}]];


General::mixedchjoin = "Provided NetChains were composed of a mixture of numbered and named layers. Either all layers must be numbered, or all layers must be named."
General::dupchjoin = "Provided NetChains contain at least one duplicated layer name (``)."
General::incchjoin = "Output of NetChain `` and input of NetChain `` have incompatible shapes, being `` and `` respectively."

NetChainJoin[{chain_}] := chain;

NetChainJoin[chains_] := Scope[
	chains = Replace[chains, e_List :> NetChain[e], {1}];
	If[!VectorQ[chains, ValidNetQ], ReturnFailed[]];
	chains = NData /@ chains;
	ioSpecs = Join[
		chainIOSpecs[First[chains], True, False],
		chainIOSpecs[Last[chains], False, True]
	];
	nodes = chains[[All, "Nodes"]];
	Do[
		u = UnifyTypes[
			t1 = First @ nodes[[i,-1,"Outputs"]], 
			t2 = First @ nodes[[i+1,1,"Inputs"]]
		];
		If[u === $Failed, ThrowFailure["incchjoin", i, i+1, MsgForm[t1], MsgForm[t2]]];
	,
		{i, Length[chains]-1}
	];
	digitQ = Map[DigitStringKeysQ, nodes];
	If[Not[SameQ @@ digitQ], ThrowFailure["mixedchjoin"]];
	If[First[digitQ],
		joined = NumberedAssociation @ Catenate @ nodes;
	,
		allKeys = Counts @ Flatten[Keys /@ nodes];
		dups = Select[allKeys, GreaterThan[1]];
		If[dups =!= <||>, ThrowFailure["dupchjoin", MsgForm @ First @ Keys @ dups]];
		joined = Join @@ nodes;
	];
	sharedArrays = JoinSharedArrays @@ Map[GetSharedArrays, chains];
	(* ^ todo: detect conflicts among shared arrays *)
	toNetChain[joined, ioSpecs, sharedArrays]
];
