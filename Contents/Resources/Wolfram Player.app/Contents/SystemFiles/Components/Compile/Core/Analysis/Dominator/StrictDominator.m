
BeginPackage["Compile`Core`Analysis`Dominator`StrictDominator`"]

StrictDominatorPass;

Begin["`Private`"]

Needs["Compile`Core`Analysis`Dominator`ImmediateDominator`"]
Needs["Compile`Core`Analysis`Dominator`Utilities`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_, opts_] :=
	fm["scanBasicBlocks", #["removeProperty", "strictDominator"]&]

run[fm_, opts_] := (
	fm["topologicalOrderScan", collectStrictDominator];
	fm
)
	
collectStrictDominator[bb_] :=
	Module[{doms = {}, idom},
	    Which[
	        bb["getParents"] === {},
			   bb["setProperty", "strictDominator" -> {}],
	    	!bb["hasProperty", "strictDominator"],
		        idom = bb["immediateDominator"];
		     	doms = Join[
		     	    {idom},
		     	    idom["getProperty", "strictDominator"]
		     	];   
			    bb["setProperty", "strictDominator" -> doms]
	    ]
	]




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"StrictDominator",
	"The pass insert the strictDominators property for each basic block.",
	"A node d dominates n (written d DOM n or d\[GreaterGreater]n) if every path from " <>
	"the start node to n contains d. d strictly dominates n if d DOM n and d != n. " <>
	"Notice that any node d dominates itself: a node d strictly dominates n iff d DOM n and d != n"
];

StrictDominatorPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"preserves" -> {ImmediateDominatorPass},
	"traversalOrder" -> "reversePostOrder",
	"passClass" -> "Analysis"
|>];

RegisterPass[StrictDominatorPass]
]]


End[] 

EndPackage[]
