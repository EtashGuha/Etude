
BeginPackage["Compile`Core`Analysis`Dominator`Utilities`"]

CreateDominatorGraph;

Begin["`Private`"]


Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Reference`ListReference`"]


getProperty[bb_, "dominator"] :=
	bb["dominator"]
getProperty[bb_, "immediateDominator"] :=
	bb["immediateDominator"]
getProperty[bb_, prop_] :=
	bb["getProperty", prop]

CreateDominatorGraph[fm_, prop_, reverseQ_:False] :=
	Module[{verts = CreateReference[{}], edges = CreateReference[{}], go, graph},
	    go[bb_] := Module[{makeEdge},
		    	makeEdge[v_] := If[reverseQ,
		    	    DirectedEdge[bb["id"], v["id"]],
		    	    DirectedEdge[v["id"], bb["id"]]
		    	];
	   		verts["pushBack", bb["id"]];
	   		Do[
	   			edges["pushBack", makeEdge[v]]
	   			,
	   			{v, Select[Flatten[{getProperty[bb, prop]}], # =!= None&]}
	   		]
	    ];
	    fm["topologicalOrderScan", go];
	    
	    graph = Graph[
	        	verts["toList"],
	        	edges["toList"],
				VertexLabels -> "Name",
				GraphLayout ->  {"LayeredDigraphEmbedding", "RootVertex" -> fm["firstBasicBlock"]["id"]},
				ImagePadding -> 15
	    	];
	    	
	    graph
	]
	

End[] 

EndPackage[]