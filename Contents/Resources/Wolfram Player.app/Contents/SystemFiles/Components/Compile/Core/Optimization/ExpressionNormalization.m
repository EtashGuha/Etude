BeginPackage["Compile`Core`Optimization`ExpressionNormalization`"]


ExpressionNormalizationPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Utilities`Language`Attributes`"]
Needs["CompileUtilities`Callback`"]


orderlessSymbols := orderlessSymbols = 
	KeyMap[Symbol, Select[$SystemAttributes, MemberQ[#, Orderless]&]]

associativeQ[sym_] := KeyExistsQ[orderlessSymbols, sym]

inverseSymbol[sym_] := Head[Module[{x, y}, ! sym[x, y]]]

orderOperands[op1_, op2_] :=
	Which[
		ConstantValueQ[op1] && ConstantValueQ[op2] && op1["id"] > op2["id"],
			{op2, op1},
		VariableQ[op1] && ConstantValueQ[op2],
			{op2, op1},
		VariableQ[op1] && VariableQ[op2] && op1["id"] > op2["id"],
			{op2, op1},
		True,
			{op1, op2}
	]
normalizeBinary[st_, inst_] :=
	If[associativeQ[inst["operator"]],
		With[{
				op1 = inst["getOperand", 1],
				op2 = inst["getOperand", 2]
			},
			inst["setOperands", orderOperands[op1, op2]]
		]
	]
	
normalizeCompare[st_, inst_] :=
	With[{
			operator = inst["operator"],
			op1 = inst["getOperand", 1],
			op2 = inst["getOperand", 2]
		},
		Which[
			associativeQ[operator],
				inst["setOperands", orderOperands[op1, op2]],
			MemberQ[{LessEqual, GreaterEqual}, operator],
				inst["setOperator", inverseSymbol[operator]];
				inst["setOperands", {op2, op1}],
			True,
				Module[{orderedOps},
					orderedOps = orderOperands[op1, op2];
					If[!op1["sameQ", First[orderedOps]],
						inst["setOperator", inverseSymbol[operator]];
						inst["setOperands", orderedOps];
					]
				]
		]
	]

run[bb_, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitBinaryInstruction" -> normalizeBinary,
			"visitCompareInstruction" -> normalizeCompare
		|>,
		bb,
		"IgnoreRequiredInstructions" -> True
	]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"ExpressionNormalization",
		"The pass normalizes binary expressions so that constants are before variables and "
];

ExpressionNormalizationPass = CreateBasicBlockPass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ExpressionNormalizationPass]
]]


End[] 

EndPackage[]
