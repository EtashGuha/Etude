(**
 * Given the following code
 *     bb1 :  inst1
 *            inst2
 *            jump bb2
 *     bb2 :  inst3
 *            inst4
 *            jump bb3
 * 
 * And bb2 can only be reached through bb1. Then we merge bb1
 * and bb2 to produce the following code:
 *     bb1 :  inst1
 *            inst2
 *            inst3
 *            inst4
 *            jump bb3
 *)

BeginPackage["Compile`Core`Optimization`FuseBasicBlocks`"]


FuseBasicBlocksPass

Begin["`Private`"] 

Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`LabelInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["CompileUtilities`Callback`"]



run[fm_?FunctionModuleQ, opts_] :=
	Module[{changed = True},
		While[changed, (**< fold the branches until we reach a fixed point *)
			changed = False;
			fm["reversePreOrderScan", (**< we need to get this each time, since this pass deletes basic blocks *)
				Function[{bb},
					changed = fuse[bb] || changed (**< We do not want short circuiting here *)
				]
			]
		];
		fm
	]

(* 
   TODO: This will need to also rename the phi operands in the future,  
   should use the fixPhiInstructions method of BasicBlock
*)
fuse[bb_?BasicBlockQ] :=
	Module[{changed = False, children, child, lastInst, childChildren},
		children = bb["getChildren"];
		If[Length[children] === 1 && Length[First[children]["getParents"]] === 1,
			child = First[children];
			lastInst = bb["lastInstruction"];

			AssertThat["The last instruction should be a branch instruction",
				lastInst]["named", "lastInstruction"]["isA", BranchInstruction];
			AssertThat["The last instruction should be an unconditional branch instruction",
				lastInst]["named", "lastInstruction"]["satisfies", #["isUnconditional"]&];
			AssertThat["The last instruction jump target to the child",
				lastInst]["named", "lastInstruction"][
				"satisfies", (#["getOperand", 1]["id"] === child["id"])&];

			lastInst["unlink"];
			
			child["scanInstructions",
				Function[{inst},
					If[!LabelInstructionQ[inst], (**< Each basic block begins with a label *)
						lastInst = bb["lastInstruction"];
						inst["moveAfter", lastInst] (**< moveAfter, unlike setNext, performs extra checks
						                                 to maintain the internal graph. For example, it 
						                                 updates the lastInstruction field in the enclosing basic block
						                              *)
					]
				]
			];
			bb["removeChild", child];
			childChildren = child["getChildren"];
			#["removeParent", child]& /@ childChildren;
			bb["addChild", #]& /@ childChildren;
			(*  
			   Now for all of the children of the child we need to fix any of the  phi instructions 
			   which once pointed to child and now should point to bb.  This is exactly what fixPhiInstructions 
			   does.
			*)
			Map[ #["fixPhiInstructions", child, bb]&, childChildren];
			bb["functionModule"]["checkLast", bb, child];
			child["functionModule"]["unlinkBasicBlock", child];
			
			changed = True;
			
		];
		changed
	]

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"FuseBasicBlocks",
		"The pass collapses two basic blocks a -> b if basic block a falls through to b and b has no other parent.",
		"The pass may delete basic blocks, so it would invalidate some analyis such as dominators. The pass looks "  <>
		"for 1 pattern: two basic blocks where a is the only parent to b. It then collapses a and b into one basic block."
];

FuseBasicBlocksPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[FuseBasicBlocksPass]
]]

End[]
EndPackage[]
