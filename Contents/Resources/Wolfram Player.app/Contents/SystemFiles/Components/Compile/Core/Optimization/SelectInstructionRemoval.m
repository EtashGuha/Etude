BeginPackage["Compile`Core`Optimization`SelectInstructionRemoval`"]


SelectInstructionRemovalPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



run[bb_, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitSelectInstruction" -> Function[{st, inst},
				Module[{op1, op2, newInst},
					op1 = inst["getOperand", 1];
					op2 = inst["getOperand", 2];
					If[op1["id"] === op2["id"], (**< the two operands are the same *)
						(** we now need to replace the select instruction with a load instruction.
						  * we use the same id so we do not invalidate other passes that reference
						  * the instruction id
						  *)
						newInst = CreateCopyInstruction[
							inst["target"],
							op1,
							inst["mexpr"]
						];
						newInst["moveAfter", inst];
						newInst["setId", inst["id"]];
						inst["unlink"];
						
					]
				]
			]
		|>,
		bb,
		"IgnoreRequiredInstructions" -> True
	]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"SelectInstructionRemoval",
		"The pass removes unessary select instructions where the two operands are equal."
];

SelectInstructionRemovalPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[SelectInstructionRemovalPass]
]]


End[] 

EndPackage[]
