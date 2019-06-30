BeginPackage["Compile`Core`Optimization`JumpThreading`"]

JumpThreadingPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]
Needs["Compile`Core`Optimization`DeadJumpElimination`"]


rewriteJump[inst_?BranchInstructionQ, oldTarget_, newTarget_] :=
	Module[{bb},
		inst["setOperands", (inst["operands"] /. {oldTarget -> newTarget})];
		bb = inst["basicBlock"];
		bb["addChild", newTarget]
	]

rewriteJump[inst_?InstructionQ, oldTarget_, newTarget_] :=
	Null

rewriteJump[args___] :=
    ThrowException[{"Unrecognized call to rewriteJump", {args}}]
    

run[fm_, opts_] :=
	Module[{changed = True},
		While[changed, (**< fold the branches until we reach a fixed point *)
			changed = False;
			fm["reversePostOrderScan", (**< we need to get this each time, since this pass deletes basic blocks *)
				Function[{bb},
					If[canBeFolded[bb],
						changed = branchFold[bb] || changed (**< We do not want short circuiting here *)
					]
				]
			];
		];
		fm
	]

(* A block can be folded when none of it's children contain Phi's *)
canBeFolded[bb_] :=
	NoneTrue[ (* None of the children ... *)
		bb["getChildren"],
		Function[{child},
			AnyTrue[ (* ... have any instructions which are Phi's *)
				child["getInstructions"],
				Function[{inst},
					Assert[
						If[PhiInstructionQ[inst],
							(* Assert that `bb` is present in this Phi's source list *)
							Module[{bbIsPhiSource},
								bbIsPhiSource = AnyTrue[
									inst["getSourceBasicBlocks"],
									#["sameQ", bb]&
								];
								(* Every parent MUST be present in the Phi sources *)
								bbIsPhiSource
							]
							,
							(* It's not a Phi, so this assertion defaults to True *)
							True
						]
					];
					PhiInstructionQ[inst]
				]
			]
		]
	]

branchFold[bb_] :=
	Module[{changed = False, target, newInst},
		Which[
		    bb["isEmpty"] && bb["hasChildren"] && bb["hasParents"],
			    (**< we delete empty basic blocks. A basic block is empty if it does not have any useful
			         instruction --- a useful instruction is anything other than a label *)
			    	Scan[
						#["scanInstructions", rewriteJump[#, bb, First[bb["getChildren"]]]&]&,
						bb["getParents"]
					];
			    	bb["unlink"];
			    	changed = True,
		    BranchInstructionQ[bb["firstNonLabelInstruction"]] && bb["firstNonLabelInstruction"]["isUnconditional"],
			    (**< If the first label is an unconditional jump to another basic block, then we forward
			         all basic blocks that enter the basic block to the target (jumped) basic block *)
			    target = First[bb["firstNonLabelInstruction"]["operands"]];
			    Scan[
					#["scanInstructions", rewriteJump[#, bb, target]&]&,
					bb["getParents"]
				];
				If[bb["functionModule"]["firstBasicBlock"] === bb,
		    		bb["functionModule"]["setFirstBasicBlock", target];
				];
				bb["unlink"];
		   	 	changed = True,
		    True,
			    (**< We now look for the final case ... conditional branch instructions were both targets 
			         are the same basic block. This may happen because of the previous two modifications 
			         to the basicblocks in this pass. *) 
			    	bb["scanInstructions",
			    		Function[{inst},
			    			If[BranchInstructionQ[inst] && inst["isConditional"] && 
			    				AllTrue[inst["operands"], inst["getOperand", 1]["id"] === #["id"]&],
			    				(**< is an conditional jump where the branches jump to the same target *)
			    				
								(** we now need to replace the instruction with an unconditional jump instruction.
								  * we use the same id so we do not invalidate other passes that reference
								  * the instruction id
								  *)
								newInst = CreateBranchInstruction[inst["getOperand", 1], inst["mexpr"]];
								newInst["moveAfter", inst];
								newInst["setId", inst["id"]];
								inst["unlink"];
			    				changed = True
			    			]
			    		]
			    	]
		];
		changed
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"JumpThreading",
		"The pass collapses unecessary branches in the IR. This pass is sometimes called BranchFolding also.",
		"The pass may delete basic blocks, so it would invalidate some analyis such as dominators. The pass looks "  <>
		"for 3 patterns: empty basic blocks, basic blocks where the only statement is an unconditional jump, " <>
		"and conditional branch instructions where the targets are the same (essentially an unconditional jump). " <>
		"The pass then rewrite and rewires the basic block tree to remove the extra basic blocks."
];

JumpThreadingPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"traversalOrder" -> "postOrder",
	"requires" -> {
		DeadJumpEliminationPass
	}
|>];

RegisterPass[JumpThreadingPass]
]]

	
End[]
EndPackage[]
