
BeginPackage["Compile`Core`Analysis`Utilities`IntervalPartition`"]



Begin["`Private`"]

Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`FunctionModule`"]


(** Section 3.2 (Algorithms for constructing intervals)
 *  in http://www.cs.umb.edu/~offner/files/flow_graph.pdf
 *)
IntervalPartition[fm_?FunctionModuleQ] :=
	Module[{start, h, j, bbs, curr, pi},
		bbs = fm["getBasicBlocks"];
		start = fm["firstBasicBlock"];
		h = {start};
		j = {};
		fm["clearVisited"];
		While[h =!= {},
			{curr, h} = {First[h], Rest[h]};
			pi = makePreInterval[start, bbs, curr];
			Scan[#["setVisited", True]&, pi];
			AppendTo[j, pi];
			Do[
				Do[
					If[!child["visitedQ"],
						AppendTo[h, child]
					],
					{child, bb["getChildren"]}
				],
				{bb, pi}
			]
		];
		fm["clearVisited"];
		j
	]

makePreInterval[startBB_, bbs0_, header_?BasicBlockQ] :=
	Module[{a, bbs, changed},
		bbs = bbs0;
		a = {header["id"]};
		changed = True;
		While[changed,
			changed = False;
			Do[
				If[bb["id"] =!= startBB["id"] &&
				   AllTrue[MemberQ[a, #["id"]&], bb["getParents"]],
					a = Union[a, bb["id"]];
					bbs = Select[bbs, #["id"] =!= bb["id"]&];
					changed = True
				],
				{bb, bbs}
			]
		];
		a
	]

End[] 

EndPackage[]