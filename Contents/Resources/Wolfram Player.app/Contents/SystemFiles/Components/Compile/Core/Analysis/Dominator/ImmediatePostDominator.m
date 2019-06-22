
BeginPackage["Compile`Core`Analysis`Dominator`ImmediatePostDominator`"]

ImmediatePostDominatorPass;

Begin["`Private`"] 

Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["Compile`Core`Analysis`Dominator`Utilities`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_, opts_] :=
	fm["scanBasicBlocks", #["setProperty", "immediatePostDominator" -> None]&]

(* This assumes that the basic blocks are in postTopologicalOrder *)
run[fm_, opts_] :=
	Module[{computePostDominator, ipdom = <||>},
		computePostDominator[bb_] /; bb["id"] === fm["lastBasicBlock"]["id"] := (
			AssociateTo[ipdom, bb -> None];
			bb["setProperty", "immediatePostDominator" -> None]
		);
		computePostDominator[bb_] := Module[{succ, candidate = None, alt},
		    Do[
		        If[succ["id"] < bb["id"],
		            If[candidate === None,
		                candidate = succ,
			            alt = succ;
			            While[alt =!= candidate,
			                If[candidate["id"] > alt["id"],
			                    candidate = ipdom[candidate],
			                    alt = ipdom[alt] 
			                ]
			            ];
		            ]
		        ],
		        {succ, bb["getChildren"]}
		    ];
			AssociateTo[ipdom, bb -> candidate];
		    bb["setProperty", "immediatePostDominator" -> candidate]
		];
		fm["postTopologicalOrderScan", computePostDominator];
		fm
	]




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ImmediatePostDominator",
	"The pass computes the immediate post dominator for each basic block in the function module.",
	"A node z is said to post-dominate a node n if all paths to the exit node of the graph starting " <>
	"at n must go through z. Similarly, the immediate post-dominator of a node n is the postdominator " <>
	"of n that doesn't strictly postdominate any other strict postdominators of n."
];


(*
  This computes the PostTopologicalOrder and then restores the TopologicalOrder after.
  Perhaps it could compute the PostTopologicalOrder as some data and use that leaving the 
  TopologicalOrder alone.
*)
ImmediatePostDominatorPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"requires" -> {PostTopologicalOrderRenumberPass},
	"postPasses" -> {TopologicalOrderRenumberPass}
|>];

RegisterPass[ImmediatePostDominatorPass]
]]


End[]
	
EndPackage[]
