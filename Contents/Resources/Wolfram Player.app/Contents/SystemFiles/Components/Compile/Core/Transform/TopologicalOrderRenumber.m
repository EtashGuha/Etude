BeginPackage["Compile`Core`Transform`TopologicalOrderRenumber`"]

TopologicalOrderRenumberPass;
PostTopologicalOrderRenumberPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]



initialize[fm_, opts_] := 
	fm["invalidateOrder"]


topologicalOrderRenumber[fm_, opts_] :=
	Module[{bbIdx = 1, instrIdx = 1, basicBlockMap = <||>},
	    fm["topologicalOrderScan", 
		    	Function[{bb},
	    	    	AssociateTo[basicBlockMap, bbIdx -> bb];
		    		bb["setId", bbIdx++];
		    		bb["scanInstructions",
		    			Function[{inst},
		    				inst["setId", instrIdx++]
		    			]
		    		]
		    	]
	    ];
		fm["setBasicBlockMap", CreateReference[basicBlockMap]];
	    fm["setProperty", "order" -> "topologicalOrder"];
	    fm
	]

RegisterCallback["RegisterPass", Function[{st},
topologicalOrderRenumberInfo = CreatePassInformation[
	"TopologicalOrderRenumber",
	"The pass reindexes all the basic block ids so they are in topological order (pre-order).",
	"Topological ordering of a directed graph is a linear ordering of its vertices "<>
	"such that for every directed edge uv from vertex u to vertex v, u comes before v " <>
	"in the ordering. i.e. the index of the parent is always less than that of its children."
];

TopologicalOrderRenumberPass = CreateFunctionModulePass[<|
	"information" -> topologicalOrderRenumberInfo,
	"runPass" -> topologicalOrderRenumber,
	"initializePass" -> initialize
|>];

RegisterPass[TopologicalOrderRenumberPass]
]]


(****************************************************************)
(****************************************************************)

postTopologicalOrderRenumber[fm_, opts_] :=
	Module[{bbIdx = 1, instrIdx = 1, basicBlockMap = <||>},
	    fm["postTopologicalOrderScan", 
		    	Function[{bb},
		    	    AssociateTo[basicBlockMap, bbIdx -> bb];
		    		bb["setId", bbIdx++];
		    		bb["reverseScanInstructions",
		    			Function[{inst},
		    				inst["setId", instrIdx++]
		    			]
		    		]
		    	]
	    ];
		fm["setBasicBlockMap", CreateReference[basicBlockMap]];
	    fm["setProperty", "order" -> "postTopologicalOrder"];
	    fm
	]


RegisterCallback["RegisterPass", Function[{st},
postTopologicalOrderRenumberInfo = CreatePassInformation[
	"PostTopologicalOrderRenumber",
	"The pass reindexes all the basic block ids so they are in reverse post topological order (post-order).",
	"Topological ordering of a directed graph is a linear ordering of its vertices "<>
	"such that for every directed edge uv from vertex u to vertex v, v comes before u " <>
	"in the ordering. i.e. the index of the child is always less than that of its parents."
];

PostTopologicalOrderRenumberPass = CreateFunctionModulePass[<|
	"information" -> postTopologicalOrderRenumberInfo,
	"runPass" -> postTopologicalOrderRenumber,
	"initializePass" -> initialize
|>];

RegisterPass[PostTopologicalOrderRenumberPass]
]]


End[]
	
EndPackage[]
