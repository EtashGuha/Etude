BeginPackage["Compile`Core`Optimization`EmptyBlockRemoval`"]

EmptyBlockRemovalPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["CompileUtilities`Callback`"]



rewriteJump[bb_, oldTarget_, newTarget_] :=
	Module[{branchInstr},
		branchInstr = bb["getTerminator"];
		branchInstr["setOperands", (branchInstr["operands"] /. {oldTarget -> newTarget})];
		bb["addChild", newTarget]
	]
rewriteJump[bb_] :=
	Module[{parents, children, child},
		parents = bb["getParents"];
		children = bb["getChildren"];
		AssertThat["The number of children is 1", children
			]["named", "children"
			]["hasLengthOf", 1
		];
		child = First[children];
		rewriteJump[#, bb, child]& /@ parents
	]
	
removeBlock[bb_] := bb["unlink"]

run[fm_, opts_] :=
	Module[{toRemove},
		toRemove = Internal`Bag[];
		fm["scanBasicBlocks",
			Function[{bb},
				With[{branchInst = bb["firstNonLabelInstruction"]},
					Which[
						branchInst === None, (* BB is empty *)
							Internal`StuffBag[toRemove, bb],
						BranchInstructionQ[branchInst] && branchInst["isUnconditional"],
							Internal`StuffBag[toRemove, bb]
					]
				]
			]
		];
		rewriteJump /@ Internal`BagPart[toRemove, All];
		removeBlock /@ Internal`BagPart[toRemove, All];
	]

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"EmptyBlockRemoval",
	"Deletes basic blocks that are empty."
];

EmptyBlockRemovalPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[EmptyBlockRemovalPass]
]]

End[] 

EndPackage[]
