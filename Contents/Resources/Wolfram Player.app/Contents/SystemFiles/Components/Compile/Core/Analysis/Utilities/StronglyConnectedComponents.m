
BeginPackage["Compile`Core`Analysis`Utilities`StronglyConnectedComponents`"]



Begin["`Private`"]

(* Textbook implementation of tarjan's strongly connected components *)
strongConnectedComponents[verts_, edges_] :=
    Module[ {idx, Scc = {}, comp, S, w, idxs = <||>, lowlink = <||>, strongConnect},
	    idx = 0;
	    S = {};
	    strongConnect[v_] :=
	        Module[{},
	            idxs[v] = idx;
	            lowlink[v] = idx;
	            idx++;
	            AppendTo[S, v];
	            Do[
	             Which[
	             	!KeyExistsQ[idxs, w],
	              		strongConnect[w];
	              		lowlink[v] = Min[lowlink[v], lowlink[w]],
	             	MemberQ[S, w],
	             		lowlink[v] = Min[lowlink[v], idxs[w]]
	             ],
	             {w, succ[v, edges]}
	            ];
	            If[idxs[v] === lowlink[v],
	                comp = {};
	                w = None;
	                While[w =!= v && S =!= {},
	                 {w, S} = {Last[S], Most[S]};
	                 AppendTo[comp, w]
	                ];
	                If[comp =!= {},
	                	AppendTo[Scc, comp]
	                ]
	            ]
	        ];
	    Do[
	     If[!KeyExistsQ[idxs, v],
	     	strongConnect[v]
	     ],
	     {v, verts}
	    ];
	    Scc
    ]
    
    
End[] 

EndPackage[]
