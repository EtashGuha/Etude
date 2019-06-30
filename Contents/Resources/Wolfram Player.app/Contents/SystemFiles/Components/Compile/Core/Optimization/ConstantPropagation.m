BeginPackage["Compile`Core`Optimization`ConstantPropagation`"]
ConstantPropagationPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Optimization`EvaluateExpression`"]
Needs["Compile`Core`Optimization`CopyPropagation`"]
Needs["Compile`Core`Optimization`CoerceCast`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Variable`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]


propagateInstruction[state_, inst_?BranchInstructionQ] :=
	Module[ {cond},
		If[inst["isConditional"],
			cond = inst["condition"];
			If[ VariableQ[cond] && state["constants"]["keyExistsQ", cond["id"]],
				inst["setCondition", state["constants"]["lookup", cond["id"]]];
				state["changed"]["set", True]]];
		]

propagateInstruction[state_, inst_?StackAllocateInstructionQ] :=
	Module[ {size = inst["size"]},
	If[VariableQ[size] && state["constants"]["keyExistsQ", size["id"]],
		inst["setSize", state["constants"]["lookup", size["id"]]];
		state["changed"]["set", True]];
	]

propagateInstruction[state_, inst_] :=
	Module[{operands, operand, idx},
		If[inst["hasOperands"],
			operands = inst["operands"];
			Table[
				operand = inst["getOperand", idx];
				If[VariableQ[operand] && state["constants"]["keyExistsQ", operand["id"]],
					inst["setOperand", idx, state["constants"]["lookup", operand["id"]]];
					state["changed"]["set", True]
				],
				{idx, Length[operands]}
			]
		]
	]
propagateConstants[state_, bb_] :=
	CreateInstructionVisitor[
		state,
		<|
			"visitInstruction" -> propagateInstruction
		|>,
		bb
	]


createState[] :=
	<|"constants" -> CreateReference[<||>], "changed" -> CreateReference[False]|>

run[fm_, opts_] :=
	Module[{state, changed = True},
		While[changed === True,
			state = createState[];
			fm["reversePostOrderScan",
				Function[{bb},
					gen[state, bb];
					propagateConstants[state, bb]
				]
			];
			changed = state["changed"]["get"];
			If[changed,
				RunPass[EvaluateExpressionPass, fm];
				RunPass[CopyPropagationPass, fm]];
		];
		fm
	]

gen[state_, bb_] :=
	Module[{},
		CreateInstructionVisitor[
			<|
				"visitCopyInstruction" -> Function[{st, inst},
					If[ConstantValueQ[inst["source"]],
						state["constants"]["associateTo", inst["target"]["id"] -> inst["source"]];
					]
				]
			|>,
			bb,
			"IgnoreRequiredInstructions" -> True
		];
	]

	
(**********************************************************)
(**********************************************************)
(**********************************************************)


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ConstantPropagation",
	"Propagates constants and evaluate constant expressions.",
	"This pass removes the need for all load constant expressions " <>
	"and evaluates expressions (using the EvaluateExpressionPass) until a fixed point is reached."
];

ConstantPropagationPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		CoerceCastPass,
		CopyPropagationPass
	}
|>];

RegisterPass[ConstantPropagationPass]
]]


End[] 

EndPackage[]
