
BeginPackage["Compile`Core`Analysis`Loop`LoopNestingForest`"]

LoopNestingForestPass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Analysis`Dominator`DominatorPass`"]
Needs["Compile`Core`Analysis`Dominator`StrictDominator`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`LoopInformation`"]
Needs["Compile`Core`Analysis`Dominator`Utilities`"]

ClearAll[run]
ClearAll[process]

initialize[ fm_, opts_] :=
    (
        fm["removeProperty", "loopinformation"];
	)
finalize[ fm_, opts_] :=
    0;  


scc[fm_?FunctionModuleQ] :=
    scc[fm["controlFlowGraph"]]
scc[g_?GraphQ] :=
    ConnectedComponents[g]

process[fm_, loopId_, graph_] :=
    With[{
    	   components = scc[graph]
    },
       Table[
       	   Which[
       	   	    component === {},
       	   	        Nothing,
       	   	    isTrivialSCC[component],
       	            fm["getBasicBlock", First[component]],
       	        True,
       	            process[
       	            	   fm,
       	            	   loopId,
       	            	   Subgraph[graph, component, VertexLabels -> Automatic],
       	            	   component
       	            	]
       	   ],
       	   {component, components}
       ]
    ];
outVerts[graph_ , vert_] :=
    Sort[Last /@ EdgeList[graph, DirectedEdge[vert, _]]];
process[fm_, loopId_, graph0_, components_] :=
    Module[{
    	   header,
    	   graph = graph0,
    	   loopEdges,
    	   info,
    	   sdom,
         verts,
         dominatorGraph
    },
      
      dominatorGraph = CreateDominatorGraph[fm, "dominator"];

      sdom = Subgraph[dominatorGraph, components, VertexLabels -> Automatic];
      verts = Sort[VertexList[sdom]];
       (* nodes that are not strictly dominated by any other node of the same SCC. *)
       header = Select[verts, 
            Function[{vert},
            	   outVerts[sdom, vert] === verts
            ]
       ];
       AssertThat["A header must exist", header
       	    ]["named", "header"
            ]["isNotEqualTo", {}
       ];
       AssertThat["There should only be one header", Length[header]
       	    ]["named", "header"
            ]["isEqualTo", 1 
       ];
       header = First[header];

       graph = hackyVertexDelete[graph, header];

       info = CreateLoopInformation[
          loopId["increment"],
          fm["getBasicBlock", header],
          process[fm, loopId, graph]
       ];
       loopEdges = EdgeList[graph0, DirectedEdge[_, header]] /. DirectedEdge -> Rule;
       info["setEdges", loopEdges];
       info
    ];

(*

hackyVertexDelete exists simply to work around bug 352826

https://bugs.wolfram.com/show?number=352826

In a debug kernel, you get this assert:

Graph::debug : Edgeless graphs should have directed and undirected flags set.

/Users/brenton/kernel-development/kernel_proto/Kernel/Source/Graphs/graph.mc: (173) assertion FALSE failed
(eval.mc, line 3505)


When 352826 is fixed, then replace hackyVertexDelete with VertexDelete

*)
hackyVertexDelete[graph_, vertex_] :=
Module[{vertices, edges},
  vertices = DeleteCases[VertexList[graph], vertex];
  edges = DeleteCases[EdgeList[graph], _[vertex, _] | _[_, vertex] ];
  Graph[vertices, edges]
]



run[fm_?FunctionModuleQ, opts_] :=
    With[{
    	   id = CreateReference[1],
    	   graph = fm["controlFlowGraph"]
    },
    With[{
    	   info = CreateLoopInformation[
       	    id["increment"],
       	    None,
            process[fm, id, graph]
       ]
    },
       fm["setProperty", "loopinformation" -> info];
    ]];

isTrivialSCC[e_?ListQ] := Length[e] === 1

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
        "LoopNestingForest",
        "This pass computes the loop nesting forest for a function module.",
        "References" -> {
        	   "Ramalingam, Ganesan. \"On loops, dominators, and dominance frontiers.\" ACM transactions on Programming Languages and Systems 24, no. 5 (2002): 455-490.",
        	   "Boissinot, Benoit, Philip Brisk, Alain Darte, and Fabrice Rastello. \"SSI properties revisited.\" ACM Transactions on Embedded Computing Systems (TECS) 11, no. 1 (2012): 21."
        }
];

LoopNestingForestPass = CreateFunctionModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "initializePass" -> initialize,
    "finalizePass" -> finalize,
    "preserves" -> {
    	   DominatorPass,
    	   StrictDominatorPass
    	},
    "passClass" -> "Analysis"
|>];

RegisterPass[LoopNestingForestPass]
]]

End[]

EndPackage[]
