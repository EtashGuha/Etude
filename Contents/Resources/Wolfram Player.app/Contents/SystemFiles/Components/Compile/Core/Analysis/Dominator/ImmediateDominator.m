
BeginPackage["Compile`Core`Analysis`Dominator`ImmediateDominator`"]

ImmediateDominatorPass;

Begin["`Private`"] 

Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["Compile`Core`Analysis`Dominator`Utilities`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_, opts_] :=
	fm["scanBasicBlocks", #["setImmediateDominator", None]&]
	
run[fm_, opts_] :=
	Module[{computeDominator, idom = <||>},
		computeDominator[bb_] /; bb["getParents"] === {} :=
			AssociateTo[idom, bb -> None];
		computeDominator[bb_] := Module[{candidate = None, pred, alt},
		    Do[
		        If[pred["id"] < bb["id"],
		            If[candidate === None,
		                candidate = pred,
			            alt = pred;
			            While[alt =!= candidate,
			                If[candidate["id"] >= alt["id"],
			                    candidate = idom[candidate],
			                    alt = idom[alt] 
			                ]
			            ]
		            ]
		        ],
		        {pred, bb["getParents"]}
		    ];
		    AssociateTo[idom, bb -> candidate];
		    bb["setImmediateDominator", candidate]
		];
		fm["topologicalOrderScan", computeDominator];
		fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ImmediateDominator",
	"The pass computes the immediate dominator for each basic block in the function module.",
	"A node d is an immediate domainator of n (written d IDOM n) if d strictly dominates n " <>
	" and does not dominate any other node that strictly dominates n. This node is unique " <> 
	" and therefore generates a tree. This requires the basic blocks to be reindexed to be in topological order."
];

ImmediateDominatorPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"preserves" -> {TopologicalOrderRenumberPass},
	"traversalOrder" -> "reversePostOrder",
	"passClass" -> "Analysis"
|>];

RegisterPass[ImmediateDominatorPass]
]]

End[]
	
EndPackage[]
