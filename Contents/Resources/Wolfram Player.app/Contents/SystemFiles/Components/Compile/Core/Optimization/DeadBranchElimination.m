BeginPackage["Compile`Core`Optimization`DeadBranchElimination`"]

DeadBranchEliminationPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Optimization`ConstantPropagation`"] 
Needs["Compile`Core`Optimization`JumpThreading`"]
Needs["Compile`Core`Optimization`FuseBasicBlocks`"]
Needs["Compile`Core`Optimization`CollectBasicBlocks`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]



operandsSameQ[inst_] :=
	Length[DeleteDuplicates[#["id"]& /@ inst["operands"]]] === 1


run[fm_, opts_] :=
	Module[{},
		CreateInstructionVisitor[
			<|
				"visitBranchInstruction" -> Function[{st, inst},
					Module[{changed = False, cond, then, else, bb, target, newInst},
						Which[
							inst["isConditional"] && ConstantValueQ[inst["condition"]],
								cond = inst["condition"];
								bb = inst["basicBlock"];
								AssertThat["The condition constant value must either be true or false",
									cond["value"]]["named", "condition"][
									"isMemberOf", {True, False}];
								then = inst["getOperand", 1];
								else = inst["getOperand", 2];
								target = If[cond["value"] === True,
									else["remove"];
									then
									, (* Else *)
									then["remove"];
									else
								];
								changed = True,
							inst["isConditional"] && operandsSameQ[inst],
								(** since both jump targets are the same, we can replace the
								  * instruction with an unconditional jump
								  *)
								target = inst["getOperand", 1];
								changed = True];
						If[ changed,
							(** we now need to replace the instruction with an unconditional jump instruction.
							  * we use the same id so we do not invalidate other passes that reference
							  * the instruction id
							  *)
							newInst = CreateBranchInstruction[target, inst["mexpr"]];
							newInst["moveAfter", inst];
							newInst["setId", inst["id"]];
							inst["unlink"]];
					]
				]
			|>,
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"DeadBranchElimination",
	"Deletes conditional branches if the condition is a constant.",
	"This is done by rewriting the conditional branch instruction to an unconditional branch " <>
	"and then running the jump threading pass on it. It is assumed that constant propagation " <>
	"has been run to propagate the constant to the condition."
];

DeadBranchEliminationPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		ConstantPropagationPass
	},
	"postPasses" -> {
		CollectBasicBlocksPass,
		JumpThreadingPass,
		FuseBasicBlocksPass
	}
|>];

RegisterPass[DeadBranchEliminationPass]
]]

End[] 

EndPackage[]
