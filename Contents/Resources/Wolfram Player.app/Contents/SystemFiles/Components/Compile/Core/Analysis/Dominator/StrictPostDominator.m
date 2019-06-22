
BeginPackage["Compile`Core`Analysis`Dominator`StrictPostDominator`"]

StrictPostDominatorPass;

Begin["`Private`"]

Needs["Compile`Core`Analysis`Dominator`ImmediatePostDominator`"]
Needs["Compile`Core`Analysis`Dominator`Utilities`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_, opts_] :=
	fm["scanBasicBlocks", #["setProperty", "strictPostDominator" -> {}]&]

run[fm_, opts_] := (
	fm["postTopologicalOrderScan", collectStrictPostDominator];
	fm
)

collectStrictPostDominator[bb_] :=
	Module[{doms = {}, idom},
	    If[bb["getChildren"] === {},
			bb["setProperty", "strictPostDominator" -> {}],
			idom = bb["getProperty", "immediatePostDominator"];
			doms = Join[
				{idom},
				idom["getProperty", "strictPostDominator", {}]
			];
			bb["setProperty", "strictPostDominator" -> doms]
	    ]
	]




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"StrictPostDominator",
	"The pass computes the strict post dominator for each basic block in the function module.",
	"A node z is said to post-dominate a node n if all paths to the exit node of the graph starting " <>
	"at n must go through z. Similarly, the immediate post-dominator of a node n is the postdominator " <>
	"of n that doesn't strictly postdominate any other strict postdominators of n."
];

StrictPostDominatorPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"preserves" -> {ImmediatePostDominatorPass}
|>];

RegisterPass[StrictPostDominatorPass]
]];

End[] 

EndPackage[]
