
(*
Make the graph acyclic by reversing as few edges as possible.
*)


BeginPackage["Compile`Core`Utilities`Visualization`Detail`RemoveCycles`"]

RemoveCycles;

Begin["`Private`"] (* Begin Private Context *) 

Needs["Compile`Utilities`DataStructure`PointerGraph`"]

	
dfsFAS[grph_] :=
	Module[{dfs, visited, fas, stack},
		fas = {};
		stack = {};
		visited = <||>;
		dfs[vert_] :=
			If[!KeyExistsQ[visited, vert],
				AssociateTo[visited, vert -> True];
				PrependTo[stack, vert];
				Do[
					If[MemberQ[stack, Last[e]],
						PrependTo[fas, e],
						dfs[Last[e]]
					],
					{e, OutEdges[grph, vert]}
				];
				stack = DeleteCases[stack, vert];
			];
		dfs /@ VertexList[grph];
		fas
	]
	
reverseEdge[DirectedEdge[a_, b_]] := DirectedEdge[b, a]
reverseEdge[a_] := a

RemoveCycles[grph_] := 
	With[{edgesToReverse = dfsFAS[grph]},
	    DeleteEdge[grph, #]& /@ edgesToReverse;
	    AddEdge[grph, reverseEdge[#]]& /@ edgesToReverse;
	    grph["properties"]["associateTo", "reversedEdges" -> edgesToReverse];
	    grph
	]

(*
An enhanced greedy heuristic by Eades et al 
P. Eades, X. Lin, and W. Smyth, "A fast and effective heuristic for the feedback arc
set problem," Information Processing Letters, vol. 47, no. 6, pp. 319 - 323, 1993.
*)
greedyFAS[grph_] :=
	Module[{sl, sr, sink, source, min},
		sl = {};
		sr = {};
		While[VertexList[grph] =!= {},
			If[HasSink[grph],
				sink = GetSink[grph];
				DeleteVertex[grph, sink];
				PrependTo[sr, sink]
			];
			If[HasSource[grph],
				source = GetSource[grph];
				DeleteVertex[grph, source];
				PrependTo[sl, source]
			];
			If[VertexList[grph] =!= {},
				min = First[
					SortBy[VertexList[grph], (InDegree[#] - OutDegree[#])&]
				];
				DeleteVertex[grph, min];
				AppendTo[sl, min]
			]
		];
		Join[sl, sr]
	]

	
End[] (* End Private Context *)

EndPackage[]