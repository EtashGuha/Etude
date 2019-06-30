BeginPackage["Compile`Core`Optimization`DeadJumpElimination`"]

DeadJumpEliminationPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)


run[bb_, opts_] :=
	Module[{reachedBranch = False},
		bb["scanInstructions",
			Function[{inst},
				Which[
					reachedBranch,
						inst["unlink"],
					BranchInstructionQ[inst],
						reachedBranch = True
				]
			]
		]
	];

run[args___] :=
	ThrowException[ {"The parameters to DeadJumpElimination are not valid.", {args}}]
	

	

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"DeadJumpElimination",
	"Deletes unreachable branches within a basic block. ",
	"This pass is targeted specifically to deal with code of the form \n\n" <>
	" branch bb1 \n" <>
	" ... \n" <>
	" branch bb2 \n\n" <>
	"the pass will delete everything after branch bb1. This code arises " <>
	"for the use of Continue[] and Break[] within user code."
];

DeadJumpEliminationPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[DeadJumpEliminationPass]
]]

End[] 

EndPackage[]
