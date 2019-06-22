(**
 * http://www.cs.rice.edu/~keith/EMBED/dom.pdf
 * http://www.cs.nyu.edu/leunga/MLRISC/Doc/html/compiler-graphs.html
 *)

BeginPackage["Compile`Core`Analysis`Dominator`DominatorPass`"]

DominatorPass;

Begin["`Private`"]

Needs["Compile`Core`Analysis`Dominator`StrictDominator`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`Analysis`Dominator`Utilities`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_, opts_] :=
	fm["scanBasicBlocks", #["setDominator", None]&]

run[fm_, opts_] := (
	fm["scanBasicBlocks",
		Function[{bb},
			bb["setDominator", Join[
		    		bb["getProperty", "strictDominator"],
		    		{bb}
			]]
		]
	];
	fm
)



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"Dominator",
		"The pass updates the dominator field in the basic blocks.",
		"A node d dominates n (written d DOM n or d\[GreaterGreater]n) if every path from " <>
		"the start node to n contains d. d strictly dominates n if d DOM n and d != n. " <>
		"Notice that any node d dominates itself: a node d strictly dominates n iff d DOM n and d != n"
];

DominatorPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"preserves" -> {StrictDominatorPass},
	"traversalOrder" -> "reversePostOrder",
	"passClass" -> "Analysis"
|>];

RegisterPass[DominatorPass]
]]


End[] 

EndPackage[]
