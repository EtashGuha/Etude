BeginPackage["Compile`Core`Transform`ConstantUnfolding`"]


ConstantUnfoldingPass; (**< TODO: not sure about the name just yet *)  

Begin["`Private`"] 

Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]




unfoldBinary[st_, inst_] :=
	Module[{operands, unfolder},
		operands = inst["getOperands"];
		unfolder[operand_, {idx_}] := Module[{newInst},
			If[ConstantValueQ[operand],
				newInst = CreateCopyInstruction[
					"C$" <> ToString[inst["id"]] <> "$" <> ToString[operand["value"]] <> "$" <> ToString[idx], (**< we need to create a unique name *)
					operand,
					operand["mexpr"]
				];
				newInst["moveBefore", inst];
				inst["setOperand", idx, newInst["target"]]
			]
		];
		MapIndexed[unfolder, operands]
	]
	
unfoldUnary[st_, inst_] := Module[{newInst},
	If[ConstantValueQ[inst["operand"]],
		newInst = CreateCopyInstruction[
			"C$" <> ToString[inst["id"]] <> "$" <> ToString[inst["operand"]["value"]] <> "$1", (**< we need to create a unique name *)
			inst["operand"],
			inst["operand"]["mexpr"]
		];
		newInst["moveBefore", inst];
		inst["setOperand", newInst["target"]]
	]
]

unfoldReturn[st_, inst_] := Module[{newInst},
	If[inst["hasValue"] && ConstantValueQ[inst["value"]],
		newInst = CreateCopyInstruction[
			"C$" <> ToString[inst["id"]] <> "$" <> ToString[inst["value"]] <> "$1", (**< we need to create a unique name *)
			inst["value"],
			inst["value"]["mexpr"]
		];
		newInst["moveBefore", inst];
		inst["setValue", newInst["target"]]
	]
]

run[bb_?BasicBlockQ, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitBinaryInstruction" -> unfoldBinary,
			"visitUnaryInstruction" -> unfoldUnary,
			"visitCompareInstruction" -> unfoldBinary,
			"visitReturnInstruction" -> unfoldReturn
		|>,
		bb,
		"IgnoreRequiredInstructions" -> True
	]
run[args___] :=
	ThrowException[{"Invalid argument to run ", args}]	



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"ConstantUnfolding",
		"The pass transforms immediate constant operands to an load constant instructions."
];

ConstantUnfoldingPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ConstantUnfoldingPass]
]]


End[] 

EndPackage[]
