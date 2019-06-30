
BeginPackage["Compile`Core`Analysis`Dominator`PostDominator`"]

PostDominatorPass;

Begin["`Private`"]

Needs["Compile`Core`Analysis`Dominator`StrictPostDominator`"]
Needs["Compile`Core`Analysis`Dominator`Utilities`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_] :=
	fm["scanBasicBlocks", #["setProperty", "postDominator" -> {}]&]

run[fm_, opts_] := (
	fm["topologicalOrderScan",
		Function[{bb},
			bb["setProperty", "postDominator" -> Append[
			    bb["getProperty", "strictPostDominator", {}],
			    bb
			]]
		]
	];
	fm
)



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"PostDominator",
	"The pass updates the post dominator property in each basic blocks.",
	"A node z is said to post-dominate a node n if all paths to the exit node of the graph starting " <>
	"at n must go through z. Similarly, the immediate post-dominator of a node n is the postdominator " <>
	"of n that doesn't strictly postdominate any other strict postdominators of n."
];

PostDominatorPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"preserves" -> {StrictPostDominatorPass},
	"traversalOrder" -> "reversePostOrder"
|>];

RegisterPass[PostDominatorPass]
]]

End[] 

EndPackage[]
